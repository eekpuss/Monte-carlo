from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, Http404
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST
from django.core.paginator import Paginator
from django.utils import timezone
import pandas as pd
import json
import io
import os
import logging
import time
import threading
from datetime import datetime

from .models import DataSession, ProcessingResult, UserActivity
from .forms import FileUploadForm, DataConfirmationForm, MonteCarloConfigForm, DataEditForm, ExportForm
from .utils import MonteCarloProcessor, handle_uploaded_file, generate_session_id, create_export_file, log_user_activity

logger = logging.getLogger('monte_carlo_app')

def home(request):
    """Home page with file upload"""
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Handle file upload
                uploaded_file = request.FILES['file']
                sheet_name = form.cleaned_data.get('sheet_name', '0')
                
                # Convert sheet_name
                try:
                    sheet_name = int(sheet_name)
                except (ValueError, TypeError):
                    pass  # Keep as string
                
                logger.info(f"Processing upload: {uploaded_file.name}, sheet: {sheet_name}")
                
                # Save file and create session
                file_path, session_id = handle_uploaded_file(uploaded_file)
                
                # Log activity
                log_user_activity(request, session_id, 'file_upload', f"Uploaded file: {uploaded_file.name}")
                
                # Analyze file
                processor = MonteCarloProcessor()
                analysis, df = processor.analyze_file(file_path, sheet_name)
                
                if not analysis['success']:
                    error_msg = f"Error analyzing file: {analysis.get('error', 'Unknown error')}"
                    messages.error(request, error_msg)
                    logger.error(f"File analysis failed: {error_msg}")
                    return render(request, 'monte_carlo_app/upload.html', {'form': form})
                
                # Create DataSession
                session = DataSession.objects.create(
                    session_id=session_id,
                    file_name=uploaded_file.name,
                    file_path=file_path,
                    total_rows=analysis['total_rows'],
                    total_columns=analysis['total_columns'],
                    file_size=uploaded_file.size,
                    sheet_name=str(sheet_name),
                    processing_status='analyzed'
                )
                
                # Store analysis results
                session.set_column_names(analysis['column_names'])
                session.set_numeric_columns(analysis['numeric_columns'])
                session.set_missing_values(analysis['missing_values'])
                session.set_data_types(analysis['data_types'])
                session.save()
                
                # Store DataFrame in session for preview
                request.session['current_session_id'] = session_id
                request.session['df_json'] = df.to_json(orient='records', date_format='iso')
                request.session['analysis'] = analysis
                
                messages.success(request, f"File '{uploaded_file.name}' uploaded and analyzed successfully!")
                return redirect('monte_carlo_app:preview_data', session_id=session_id)
                
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                messages.error(request, error_msg)
                logger.error(f"File processing error: {error_msg}")
                return render(request, 'monte_carlo_app/upload.html', {'form': form})
    else:
        form = FileUploadForm()
    
    # Get recent sessions for display
    recent_sessions = DataSession.objects.all()[:5]
    
    context = {
        'form': form,
        'recent_sessions': recent_sessions,
        'page_title': 'Monte Carlo Time Series Imputation'
    }
    
    return render(request, 'monte_carlo_app/upload.html', context)

def preview_data(request, session_id):
    """Preview data and ask for confirmation"""
    session = get_object_or_404(DataSession, session_id=session_id)
    
    # Get DataFrame from session
    df_json = request.session.get('df_json')
    analysis = request.session.get('analysis', {})
    
    if not df_json:
        messages.error(request, "Data not found. Please upload file again.")
        return redirect('monte_carlo_app:home')
    
    try:
        df = pd.read_json(df_json, orient='records')
        
        # Log activity
        log_user_activity(request, session_id, 'data_preview', "Viewing data preview")
        
        # Prepare data for template
        context = {
            'session': session,
            'column_names': session.get_column_names(),
            'numeric_columns': session.get_numeric_columns(),
            'missing_values': session.get_missing_values(),
            'data_types': session.get_data_types(),
            'data_preview': df.head(20).fillna('').to_dict('records'),
            'total_missing': session.get_total_missing_values(),
            'data_quality': analysis.get('data_quality', {}),
            'time_candidates': analysis.get('time_candidates', []),
            'numeric_summary': analysis.get('numeric_summary', {}),
            'page_title': f'Data Preview - {session.file_name}'
        }
        
        return render(request, 'monte_carlo_app/preview.html', context)
        
    except Exception as e:
        error_msg = f"Error loading data preview: {str(e)}"
        messages.error(request, error_msg)
        logger.error(f"Preview error: {error_msg}")
        return redirect('monte_carlo_app:home')

def edit_data(request, session_id):
    """Edit data page with inline editing capability"""
    session = get_object_or_404(DataSession, session_id=session_id)
    
    # Get DataFrame
    df_json = request.session.get('df_json')
    if not df_json:
        messages.error(request, "Data not found. Please upload file again.")
        return redirect('monte_carlo_app:home')
    
    try:
        df = pd.read_json(df_json, orient='records')
        
        if request.method == 'POST':
            form = DataEditForm(request.POST)
            if form.is_valid():
                try:
                    edited_data_json = request.POST.get('edited_data', '[]')
                    edited_data = json.loads(edited_data_json)
                    edit_notes = form.cleaned_data.get('edit_notes', '')
                    
                    # Log activity
                    log_user_activity(request, session_id, 'data_edit', f"Manual data editing. Notes: {edit_notes}")
                    
                    # Apply edits
                    changes_count = 0
                    for edit in edited_data:
                        row_idx = edit.get('row')
                        col_name = edit.get('column')
                        new_value = edit.get('value')
                        
                        if row_idx is not None and col_name in df.columns:
                            old_value = df.loc[row_idx, col_name]
                            
                            # Convert value to appropriate type
                            if pd.isna(new_value) or new_value == '' or new_value == 'null':
                                df.loc[row_idx, col_name] = pd.NA
                            else:
                                try:
                                    if col_name in session.get_numeric_columns():
                                        df.loc[row_idx, col_name] = float(new_value)
                                    else:
                                        df.loc[row_idx, col_name] = str(new_value)
                                    changes_count += 1
                                except (ValueError, TypeError):
                                    df.loc[row_idx, col_name] = str(new_value)
                    
                    # Update session data
                    request.session['df_json'] = df.to_json(orient='records', date_format='iso')
                    
                    # Update missing values count
                    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
                    missing_values = df[numeric_columns].isnull().sum().to_dict()
                    session.set_missing_values(missing_values)
                    session.save()
                    
                    messages.success(request, f"Data updated successfully! {changes_count} changes applied.")
                    return redirect('monte_carlo_app:confirm_data', session_id=session_id)
                    
                except Exception as e:
                    error_msg = f"Error updating data: {str(e)}"
                    messages.error(request, error_msg)
                    logger.error(f"Data edit error: {error_msg}")
        else:
            form = DataEditForm()
        
        # Prepare data for template with pagination
        page_size = 50
        total_rows = len(df)
        page = request.GET.get('page', 1)
        
        try:
            page = int(page)
        except (ValueError, TypeError):
            page = 1
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        data_page = df.iloc[start_idx:end_idx]
        
        # Create pagination info
        has_previous = page > 1
        has_next = end_idx < total_rows
        
        context = {
            'session': session,
            'form': form,
            'data': data_page.fillna('').to_dict('records'),
            'columns': list(df.columns),
            'numeric_columns': session.get_numeric_columns(),
            'page': page,
            'page_size': page_size,
            'total_rows': total_rows,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'has_previous': has_previous,
            'has_next': has_next,
            'previous_page': page - 1 if has_previous else None,
            'next_page': page + 1 if has_next else None,
            'page_title': f'Edit Data - {session.file_name}'
        }
        
        return render(request, 'monte_carlo_app/edit_data.html', context)
        
    except Exception as e:
        error_msg = f"Error in data editing: {str(e)}"
        messages.error(request, error_msg)
        logger.error(f"Edit data error: {error_msg}")
        return redirect('monte_carlo_app:preview_data', session_id=session_id)

def confirm_data(request, session_id):
    """Confirm data and proceed to configuration"""
    session = get_object_or_404(DataSession, session_id=session_id)
    
    # Get analysis data
    analysis = request.session.get('analysis', {})
    time_candidates = analysis.get('time_candidates', [])
    time_column_choices = [tc['column'] for tc in time_candidates]
    
    if request.method == 'POST':
        form = DataConfirmationForm(request.POST, time_column_choices=time_column_choices)
        if form.is_valid():
            time_column = form.cleaned_data.get('time_column', '')
            missing_data_handling = form.cleaned_data.get('missing_data_handling', 'impute')
            
            # Update session
            session.time_column = time_column
            session.is_data_confirmed = True
            session.processing_status = 'confirmed'
            session.save()
            
            # Log activity
            log_user_activity(request, session_id, 'data_confirm', 
                            f"Data confirmed. Time column: {time_column}, Handling: {missing_data_handling}")
            
            messages.success(request, "Data confirmed! Proceeding to configuration.")
            
            if missing_data_handling == 'skip':
                messages.info(request, "Note: Rows with missing values will be skipped during processing.")
            elif missing_data_handling == 'proceed':
                messages.info(request, "Note: Data will be processed as-is without imputation.")
            
            return redirect('monte_carlo_app:configure', session_id=session_id)
    else:
        # Pre-select best time column candidate
        initial_time_column = ''
        if time_candidates:
            # Sort by confidence and pick best
            best_candidate = max(time_candidates, key=lambda x: x.get('confidence', 0))
            initial_time_column = best_candidate['column']
        
        form = DataConfirmationForm(
            initial={'time_column': initial_time_column},
            time_column_choices=time_column_choices
        )
    
    # Get summary statistics
    df_json = request.session.get('df_json')
    summary_stats = {}
    if df_json:
        try:
            df = pd.read_json(df_json, orient='records')
            numeric_cols = session.get_numeric_columns()
            if numeric_cols:
                summary_stats = df[numeric_cols].describe().to_dict()
        except Exception as e:
            logger.warning(f"Error calculating summary stats: {str(e)}")
    
    context = {
        'session': session,
        'form': form,
        'time_candidates': time_candidates,
        'data_quality': analysis.get('data_quality', {}),
        'summary_stats': summary_stats,
        'page_title': f'Confirm Data - {session.file_name}'
    }
    
    return render(request, 'monte_carlo_app/confirm.html', context)

def configure(request, session_id):
    """Configure Monte Carlo parameters"""
    session = get_object_or_404(DataSession, session_id=session_id)
    
    if not session.is_data_confirmed:
        messages.warning(request, "Please confirm your data first.")
        return redirect('monte_carlo_app:confirm_data', session_id=session_id)
    
    if request.method == 'POST':
        form = MonteCarloConfigForm(request.POST)
        if form.is_valid():
            # Update session configuration
            session.n_simulations = form.cleaned_data['n_simulations']
            session.data_frequency = form.cleaned_data['data_frequency']
            session.imputation_method = form.cleaned_data['imputation_method']
            session.processing_status = 'configured'
            session.save()
            
            # Log activity
            config_details = f"Simulations: {session.n_simulations}, Method: {session.imputation_method}"
            log_user_activity(request, session_id, 'configure', f"Configuration set. {config_details}")
            
            messages.success(request, "Configuration saved! Ready to process data.")
            return redirect('monte_carlo_app:process_data', session_id=session_id)
    else:
        # Initialize with current session values
        initial_data = {
            'n_simulations': session.n_simulations,
            'data_frequency': session.data_frequency,
            'imputation_method': session.imputation_method,
        }
        form = MonteCarloConfigForm(initial=initial_data)
    
    # Get data characteristics for recommendations
    df_json = request.session.get('df_json')
    recommendations = []
    
    if df_json:
        try:
            df = pd.read_json(df_json, orient='records')
            missing_count = session.get_total_missing_values()
            total_values = session.total_rows * len(session.get_numeric_columns())
            
            if missing_count > 0:
                missing_pct = (missing_count / total_values) * 100
                
                if missing_pct < 5:
                    recommendations.append("Low missing data percentage - Simple or Adaptive method recommended")
                elif missing_pct < 20:
                    recommendations.append("Moderate missing data - Adaptive method with higher simulations recommended")
                else:
                    recommendations.append("High missing data percentage - Consider data quality before proceeding")
                
                if session.time_column:
                    recommendations.append("Time column detected - Interpolation method may work well")
                
                if session.total_rows > 1000:
                    recommendations.append("Large dataset - Consider reducing simulations for faster processing")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
    
    context = {
        'session': session,
        'form': form,
        'recommendations': recommendations,
        'page_title': f'Configure Processing - {session.file_name}'
    }
    
    return render(request, 'monte_carlo_app/configure.html', context)
def process_data(request, session_id):
    """Process data with Monte Carlo imputation"""
    session = get_object_or_404(DataSession, session_id=session_id)
    
    if session.processing_status == 'processing':
        # Already processing, show status page
        return render(request, 'monte_carlo_app/processing.html', {
            'session': session,
            'page_title': f'Processing - {session.file_name}'
        })
    
    if request.method == 'POST':
        # Start processing
        try:
            # Get DataFrame
            df_json = request.session.get('df_json')
            if not df_json:
                messages.error(request, "Data not found. Please start over.")
                return redirect('monte_carlo_app:home')
            
            df = pd.read_json(df_json, orient='records')
            
            # Update status
            session.processing_status = 'processing'
            session.save()
            
            # Log activity
            log_user_activity(request, session_id, 'process_start', "Started Monte Carlo processing")
            
            # Start background processing
            def background_process():
                try:
                    processor = MonteCarloProcessor(
                        n_simulations=session.n_simulations,
                        confidence_level=0.95
                    )
                    
                    # Perform imputation
                    result_df, validation = processor.monte_carlo_imputation(
                        df=df,
                        time_column=session.time_column if session.time_column else None,
                        freq=session.data_frequency,
                        method=session.imputation_method,
                        preserve_patterns=True
                    )
                    
                    # Create ProcessingResult
                    processing_result, created = ProcessingResult.objects.get_or_create(session=session)
                    
                    # Store results
                    processing_result.set_original_data_sample(df)
                    processing_result.set_imputed_data_sample(result_df)
                    processing_result.set_validation_results(validation)
                    processing_result.set_imputation_log(processor.imputation_log)
                    
                    # Calculate metrics
                    processing_result.processing_time = validation.get('processing_time', 0)
                    processing_result.imputed_values_count = validation.get('imputed_count', 0)
                    processing_result.imputation_quality_score = validation.get('overall_quality', 0)
                    
                    # Create export file
                    export_path = create_export_file(result_df, session, 'xlsx', True)
                    processing_result.result_file_path = export_path
                    
                    processing_result.save()
                    
                    # Update session status
                    session.processing_status = 'completed'
                    session.is_processed = True
                    session.save()
                    
                    # Store result in session for immediate access
                    request.session['result_df_json'] = result_df.to_json(orient='records', date_format='iso')
                    
                    logger.info(f"Processing completed for session {session_id}")
                    
                except Exception as e:
                    logger.error(f"Processing failed for session {session_id}: {str(e)}")
                    session.processing_status = 'error'
                    session.save()
            
            # Start background thread
            thread = threading.Thread(target=background_process)
            thread.daemon = True
            thread.start()
            
            messages.success(request, "Processing started! This may take a few minutes.")
            return render(request, 'monte_carlo_app/processing.html', {
                'session': session,
                'page_title': f'Processing - {session.file_name}'
            })
            
        except Exception as e:
            error_msg = f"Error starting processing: {str(e)}"
            messages.error(request, error_msg)
            logger.error(f"Process start error: {error_msg}")
            session.processing_status = 'error'
            session.save()
    
    # Show processing confirmation page
    context = {
        'session': session,
        'estimated_time': max(1, session.n_simulations // 200),  # Rough estimate
        'page_title': f'Ready to Process - {session.file_name}'
    }
    
    return render(request, 'monte_carlo_app/process.html', context)

def view_results(request, session_id):
    """View processing results"""
    session = get_object_or_404(DataSession, session_id=session_id)
    
    if not session.is_processed:
        if session.processing_status == 'processing':
            return redirect('monte_carlo_app:process_data', session_id=session_id)
        else:
            messages.warning(request, "Data has not been processed yet.")
            return redirect('monte_carlo_app:configure', session_id=session_id)
    
    try:
        # Get processing result
        result = get_object_or_404(ProcessingResult, session=session)
        
        # Get result DataFrame
        result_df_json = request.session.get('result_df_json')
        if result_df_json:
            result_df = pd.read_json(result_df_json, orient='records')
        else:
            result_df = result.get_imputed_data_sample()
        
        # Prepare context
        validation_results = result.get_validation_results()
        imputation_log = result.get_imputation_log()
        
        # Calculate summary statistics
        imputed_mask = result_df['Status'] == 'Imputed' if 'Status' in result_df.columns else pd.Series([])
        imputed_rows = result_df[imputed_mask] if imputed_mask.any() else pd.DataFrame()
        
        summary = {
            'total_rows': len(result_df),
            'imputed_rows': len(imputed_rows),
            'imputed_percentage': (len(imputed_rows) / len(result_df)) * 100 if len(result_df) > 0 else 0,
            'processing_time': result.processing_time,
            'quality_score': result.imputation_quality_score or 0,
            'simulations_used': session.n_simulations,
            'method_used': session.imputation_method
        }
        
        # Log activity
        log_user_activity(request, session_id, 'view_results', "Viewing processing results")
        
        context = {
            'session': session,
            'result': result,
            'summary': summary,
            'validation_results': validation_results,
            'data_preview': result_df.head(20).fillna('').to_dict('records'),
            'columns': list(result_df.columns),
            'quality_level': get_quality_level(summary['quality_score']),
            'page_title': f'Results - {session.file_name}'
        }
        
        return render(request, 'monte_carlo_app/results.html', context)
        
    except Exception as e:
        error_msg = f"Error loading results: {str(e)}"
        messages.error(request, error_msg)
        logger.error(f"Results view error: {error_msg}")
        return redirect('monte_carlo_app:configure', session_id=session_id)

def visualization(request, session_id):
    """Create visualizations of results"""
    session = get_object_or_404(DataSession, session_id=session_id)
    
    if not session.is_processed:
        messages.warning(request, "Please process the data first.")
        return redirect('monte_carlo_app:view_results', session_id=session_id)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        import base64
        
        # Get result data
        result_df_json = request.session.get('result_df_json')
        if not result_df_json:
            result = get_object_or_404(ProcessingResult, session=session)
            result_df = result.get_imputed_data_sample()
        else:
            result_df = pd.read_json(result_df_json, orient='records')
        
        # Create visualizations
        visualizations = create_visualizations(result_df, session)
        
        # Log activity
        log_user_activity(request, session_id, 'view_visualization', "Viewing data visualizations")
        
        context = {
            'session': session,
            'visualizations': visualizations,
            'page_title': f'Visualizations - {session.file_name}'
        }
        
        return render(request, 'monte_carlo_app/visualization.html', context)
        
    except Exception as e:
        error_msg = f"Error creating visualizations: {str(e)}"
        messages.error(request, error_msg)
        logger.error(f"Visualization error: {error_msg}")
        return redirect('monte_carlo_app:view_results', session_id=session_id)

# AJAX Views
@require_POST
def validate_file_ajax(request):
    """AJAX endpoint to validate uploaded file"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({'valid': False, 'error': 'No file uploaded'})
        
        uploaded_file = request.FILES['file']
        
        # Basic validation
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB
            return JsonResponse({'valid': False, 'error': 'File too large (max 50MB)'})
        
        allowed_extensions = ['.xlsx', '.xls', '.ods']
        file_extension = '.' + uploaded_file.name.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            return JsonResponse({'valid': False, 'error': 'Invalid file type'})
        
        return JsonResponse({
            'valid': True,
            'filename': uploaded_file.name,
            'size': uploaded_file.size,
            'type': file_extension
        })
        
    except Exception as e:
        return JsonResponse({'valid': False, 'error': str(e)})

@require_POST
def update_data_ajax(request):
    """AJAX endpoint to update data values"""
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        edits = data.get('edits', [])
        
        if not session_id:
            return JsonResponse({'success': False, 'error': 'No session ID provided'})
        
        # Get session
        session = get_object_or_404(DataSession, session_id=session_id)
        
        # Get current DataFrame
        df_json = request.session.get('df_json')
        if not df_json:
            return JsonResponse({'success': False, 'error': 'Data not found'})
        
        df = pd.read_json(df_json, orient='records')
        
        # Apply edits
        for edit in edits:
            row_idx = edit.get('row')
            col_name = edit.get('column')
            new_value = edit.get('value')
            
            if row_idx