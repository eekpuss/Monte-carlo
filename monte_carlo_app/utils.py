import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import uuid
import os
import logging
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('monte_carlo_app')

class MonteCarloProcessor:
    """Advanced Monte Carlo Time Series Imputation Processor"""
    
    def __init__(self, n_simulations=1000, confidence_level=0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        self.imputation_log = {}
        self.processing_start_time = None
        self.quality_metrics = {}
    
    def analyze_file(self, file_path, sheet_name=0):
        """
        Analyze uploaded file and return comprehensive structure info
        """
        logger.info(f"Analyzing file: {file_path}")
        
        try:
            # Load file with appropriate engine
            if file_path.lower().endswith('.ods'):
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='odf')
                except ImportError:
                    logger.warning("ODF engine not available, trying default")
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            logger.info(f"File loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Comprehensive data analysis
            analysis = self._perform_data_analysis(df)
            analysis['success'] = True
            
            return analysis, df
            
        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }, None
    
    def _perform_data_analysis(self, df):
        """Perform comprehensive data analysis"""
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
        }
        
        # Identify column types
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        analysis.update({
            'numeric_columns': numeric_columns,
            'text_columns': text_columns,
            'datetime_columns': datetime_columns,
        })
        
        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
        
        analysis.update({
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'total_missing': sum(missing_values.values()),
            'missing_rows': df.isnull().any(axis=1).sum(),
        })
        
        # Data quality assessment
        analysis['data_quality'] = self._assess_data_quality(df)
        
        # Data preview (first 15 rows)
        analysis['data_preview'] = df.head(15).fillna('').to_dict('records')
        
        # Detect potential time columns
        analysis['time_candidates'] = self._detect_time_columns(df)
        
        # Statistical summary for numeric columns
        if numeric_columns:
            analysis['numeric_summary'] = df[numeric_columns].describe().to_dict()
        
        return analysis
    
    def _assess_data_quality(self, df):
        """Assess overall data quality"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        quality_score = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Quality categories
        if quality_score >= 0.95:
            quality_level = 'Excellent'
        elif quality_score >= 0.85:
            quality_level = 'Good'
        elif quality_score >= 0.70:
            quality_level = 'Fair'
        else:
            quality_level = 'Poor'
        
        return {
            'score': quality_score,
            'level': quality_level,
            'missing_percentage': (missing_cells / total_cells) * 100,
            'total_cells': total_cells,
            'missing_cells': missing_cells,
        }
    
    def _detect_time_columns(self, df):
        """Detect potential time/date columns"""
        time_candidates = []
        time_keywords = ['time', 'waktu', 'tanggal', 'date', 'timestamp', 'datetime', 'jam', 'hour']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check column name
            if any(keyword in col_lower for keyword in time_keywords):
                time_candidates.append({
                    'column': col,
                    'reason': 'Column name contains time keyword',
                    'confidence': 0.8
                })
                continue
            
            # Check data content
            sample_data = df[col].dropna().head(10)
            if len(sample_data) > 0:
                time_like_count = 0
                for value in sample_data:
                    if self._is_time_like(value):
                        time_like_count += 1
                
                if time_like_count / len(sample_data) > 0.5:
                    time_candidates.append({
                        'column': col,
                        'reason': 'Column contains time-like values',
                        'confidence': time_like_count / len(sample_data)
                    })
        
        return time_candidates
    
    def _is_time_like(self, value):
        """Check if a value looks like time/date"""
        if pd.isna(value):
            return False
        
        value_str = str(value)
        
        # Common time patterns
        time_patterns = [
            r'\d{1,2}:\d{2}',  # HH:MM
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        import re
        for pattern in time_patterns:
            if re.search(pattern, value_str):
                return True
        
        # Try pandas datetime parsing
        try:
            pd.to_datetime(value)
            return True
        except:
            return False
    
    def clean_data(self, df, time_column=None):
        """Clean and prepare data for processing"""
        logger.info("Starting data cleaning process")
        df_clean = df.copy()
        
        if time_column and time_column in df_clean.columns:
            logger.info(f"Cleaning time column: {time_column}")
            df_clean = self._clean_time_column(df_clean, time_column)
        
        # Clean numeric columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            df_clean = self._clean_numeric_column(df_clean, col)
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def _clean_time_column(self, df, time_column):
        """Clean time column with multiple format support"""
        def standardize_time_value(val):
            if pd.isna(val):
                return val
            elif hasattr(val, 'hour'):  # datetime.time object
                return f"{val.hour:02d}:{val.minute:02d}:{val.second:02d}"
            elif isinstance(val, str):
                return val.strip()
            else:
                return str(val)
        
        df[time_column] = df[time_column].apply(standardize_time_value)
        return df
    
    def _clean_numeric_column(self, df, column):
        """Clean numeric column by handling outliers and invalid values"""
        # Remove infinite values
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        
        # Optional: Handle extreme outliers (disabled by default)
        # Q1 = df[column].quantile(0.25)
        # Q3 = df[column].quantile(0.75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - 3 * IQR
        # upper_bound = Q3 + 3 * IQR
        # df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def monte_carlo_imputation(self, df, time_column=None, freq='1min', method='adaptive', preserve_patterns=True):
        """
        Perform advanced Monte Carlo imputation with pattern preservation
        """
        logger.info("Starting Monte Carlo imputation process")
        self.processing_start_time = time.time()
        
        try:
            # Clean data first
            df_processed = self.clean_data(df, time_column)
            
            # Check for missing values
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            total_nan = df_processed[numeric_columns].isnull().sum().sum()
            
            logger.info(f"Found {total_nan} missing values in {len(numeric_columns)} numeric columns")
            
            if total_nan == 0:
                logger.info("No missing values found")
                df_result = df_processed.copy()
                df_result['Status'] = 'Original'
                return df_result, {'message': 'No missing values found', 'imputed_count': 0}
            
            # Perform imputation based on strategy
            if time_column and self._is_time_series_suitable(df_processed, time_column):
                logger.info("Using time-series aware imputation")
                df_result = self._time_series_imputation(df_processed, time_column, freq, method, preserve_patterns)
            else:
                logger.info("Using general missing value imputation")
                df_result = self._general_imputation(df_processed, numeric_columns, method, preserve_patterns)
            
            # Validate results
            validation = self._validate_imputation(df_processed, df_result, numeric_columns)
            
            # Calculate processing metrics
            processing_time = time.time() - self.processing_start_time
            validation['processing_time'] = processing_time
            validation['imputed_count'] = total_nan
            
            logger.info(f"Imputation completed in {processing_time:.2f} seconds")
            
            return df_result, validation
            
        except Exception as e:
            logger.error(f"Imputation failed: {str(e)}")
            raise Exception(f"Imputation failed: {str(e)}")
    
    def _is_time_series_suitable(self, df, time_column):
        """Check if data is suitable for time-series imputation"""
        if not time_column or time_column not in df.columns:
            return False
        
        try:
            # Try to sort by time column
            df_sorted = df.sort_values(time_column)
            return True
        except Exception:
            return False
    
    def _time_series_imputation(self, df, time_column, freq, method, preserve_patterns):
        """Time-series aware imputation"""
        df_result = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Sort by time
        df_result = df_result.sort_values(time_column)
        
        for col in numeric_columns:
            missing_mask = df_result[col].isnull()
            if not missing_mask.any():
                continue
            
            logger.info(f"Imputing time-series column: {col}")
            
            for idx in df_result[missing_mask].index:
                imputed_value = self._impute_time_series_value(
                    df_result, col, idx, time_column, method, preserve_patterns
                )
                df_result.loc[idx, col] = imputed_value
        
        # Add status column
        df_result['Status'] = 'Original'
        for col in numeric_columns:
            missing_mask = df[col].isnull()
            df_result.loc[missing_mask, 'Status'] = 'Imputed'
        
        return df_result
    
    def _general_imputation(self, df, numeric_columns, method, preserve_patterns):
        """General missing value imputation"""
        df_result = df.copy()
        self.imputation_log = {}
        
        for col in numeric_columns:
            missing_mask = df_result[col].isnull()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                continue
            
            logger.info(f"Imputing column {col}: {missing_count} missing values")
            
            # Calculate distribution parameters
            params = self._calculate_distribution_params(df_result, col, method)
            
            # Impute each missing value
            for idx in df_result[missing_mask].index:
                # Context-aware imputation
                local_params = self._get_local_context(df_result, col, idx, preserve_patterns)
                
                # Combine global and local parameters
                if local_params:
                    combined_mean = 0.3 * params['mean'] + 0.7 * local_params['mean']
                    combined_std = max(params['std'], local_params['std']) * 0.8
                else:
                    combined_mean = params['mean']
                    combined_std = params['std']
                
                # Monte Carlo simulation
                simulated_values = np.random.normal(combined_mean, combined_std, self.n_simulations)
                
                # Apply constraints if preserve_patterns is True
                if preserve_patterns:
                    simulated_values = self._apply_pattern_constraints(
                        simulated_values, df_result, col, idx
                    )
                
                imputed_value = np.mean(simulated_values)
                df_result.loc[idx, col] = imputed_value
                
                # Log imputation details
                self._log_imputation(idx, col, imputed_value, combined_mean, combined_std, simulated_values, method)
        
        # Add status column
        df_result['Status'] = 'Original'
        for col in numeric_columns:
            missing_mask = df[col].isnull()
            df_result.loc[missing_mask, 'Status'] = 'Imputed'
        
        return df_result
    
    def _impute_time_series_value(self, df, column, index, time_column, method, preserve_patterns):
        """Impute single time-series value with context awareness"""
        # Get temporal context
        current_time = df.loc[index, time_column]
        
        # Find nearby values
        before_mask = df[time_column] < current_time
        after_mask = df[time_column] > current_time
        
        before_values = df[before_mask][column].dropna()
        after_values = df[after_mask][column].dropna()
        
        # Determine imputation strategy
        if len(before_values) > 0 and len(after_values) > 0:
            # Interpolation
            before_val = before_values.iloc[-1]
            after_val = after_values.iloc[0]
            base_value = (before_val + after_val) / 2
            std_dev = df[column].std() * 0.1  # Small variation
            method_used = 'interpolation'
        elif len(before_values) > 0:
            # Forward fill with trend
            recent_values = before_values.tail(5)
            if len(recent_values) > 1:
                trend = (recent_values.iloc[-1] - recent_values.iloc[0]) / len(recent_values)
                base_value = recent_values.iloc[-1] + trend
            else:
                base_value = recent_values.iloc[-1]
            std_dev = recent_values.std() if len(recent_values) > 1 else df[column].std() * 0.2
            method_used = 'forward_trend'
        elif len(after_values) > 0:
            # Backward fill
            base_value = after_values.iloc[0]
            std_dev = df[column].std() * 0.2
            method_used = 'backward_fill'
        else:
            # Global mean
            base_value = df[column].mean()
            std_dev = df[column].std()
            method_used = 'global_mean'
        
        # Monte Carlo simulation
        simulated_values = np.random.normal(base_value, std_dev, self.n_simulations)
        
        # Apply constraints
        if preserve_patterns:
            simulated_values = self._apply_pattern_constraints(simulated_values, df, column, index)
        
        return np.mean(simulated_values)
    
    def _get_local_context(self, df, column, index, preserve_patterns, window_size=10):
        """Get local context for imputation"""
        try:
            start_idx = max(0, index - window_size)
            end_idx = min(len(df), index + window_size + 1)
            
            local_data = df[column].iloc[start_idx:end_idx].dropna()
            
            if len(local_data) < 2:
                return None
            
            return {
                'mean': local_data.mean(),
                'std': local_data.std() if local_data.std() > 0 else df[column].std() * 0.1,
                'trend': self._calculate_trend(local_data) if len(local_data) > 2 else 0
            }
        except:
            return None
    
    def _calculate_trend(self, data):
        """Calculate simple linear trend"""
        if len(data) < 3:
            return 0
        
        try:
            x = np.arange(len(data))
            slope, _, _, _, _ = stats.linregress(x, data)
            return slope
        except:
            return 0
    
    def _apply_pattern_constraints(self, simulated_values, df, column, index):
        """Apply pattern-based constraints to simulated values"""
        # Get column statistics for reasonable bounds
        col_data = df[column].dropna()
        if len(col_data) == 0:
            return simulated_values
        
        # Set reasonable bounds (mean ± 3 standard deviations)
        mean_val = col_data.mean()
        std_val = col_data.std()
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        
        # Apply bounds
        simulated_values = np.clip(simulated_values, lower_bound, upper_bound)
        
        return simulated_values
    
    def _calculate_distribution_params(self, df, column, method='adaptive'):
        """Calculate distribution parameters for Monte Carlo simulation"""
        clean_data = df[column].dropna()
        
        if len(clean_data) < 2:
            return {'mean': 0, 'std': 1, 'method': 'default'}
        
        if method == 'simple':
            mean_val = clean_data.mean()
            std_val = max(clean_data.std(), 0.01)  # Minimum std to avoid zero
        elif method == 'rolling':
            window_size = min(10, len(clean_data) // 2)
            if window_size > 1:
                rolling_mean = clean_data.rolling(window=window_size, center=True).mean()
                rolling_std = clean_data.rolling(window=window_size, center=True).std()
                mean_val = rolling_mean.dropna().iloc[-1] if not rolling_mean.dropna().empty else clean_data.mean()
                std_val = rolling_std.dropna().iloc[-1] if not rolling_std.dropna().empty else clean_data.std()
            else:
                mean_val = clean_data.mean()
                std_val = clean_data.std()
        else:  # adaptive
            global_mean = clean_data.mean()
            global_std = max(clean_data.std(), 0.01)
            
            if len(clean_data) > 10:
                # Use recent trend
                recent_data = clean_data.tail(10)
                trend_mean = recent_data.mean()
                trend_std = max(recent_data.std(), global_std * 0.5)
                
                # Weighted combination
                weight = 0.7
                mean_val = weight * trend_mean + (1 - weight) * global_mean
                std_val = max(trend_std, global_std * 0.3)
            else:
                mean_val = global_mean
                std_val = global_std
        
        return {
            'mean': mean_val,
            'std': max(std_val, 0.01),  # Ensure minimum std
            'method': method,
            'data_points': len(clean_data)
        }
    
    def _log_imputation(self, index, column, imputed_value, mean_val, std_val, simulated_values, method):
        """Log imputation details"""
        if index not in self.imputation_log:
            self.imputation_log[index] = {}
        
        self.imputation_log[index][column] = {
            'imputed_value': float(imputed_value),
            'distribution_mean': float(mean_val),
            'distribution_std': float(std_val),
            'method': method,
            'confidence_interval': [
                float(np.percentile(simulated_values, (1 - self.confidence_level) / 2 * 100)),
                float(np.percentile(simulated_values, (1 + self.confidence_level) / 2 * 100))
            ],
            'simulation_stats': {
                'min': float(np.min(simulated_values)),
                'max': float(np.max(simulated_values)),
                'std': float(np.std(simulated_values))
            }
        }
    
    def _validate_imputation(self, original_df, imputed_df, numeric_columns):
        """Comprehensive imputation validation"""
        validation_results = {
            'overall_quality': 0,
            'column_quality': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        quality_scores = []
        
        for col in numeric_columns:
            try:
                col_validation = self._validate_column_imputation(original_df, imputed_df, col)
                validation_results['column_quality'][col] = col_validation
                quality_scores.append(col_validation.get('quality_score', 0))
            except Exception as e:
                logger.warning(f"Validation failed for column {col}: {str(e)}")
                continue
        
        # Overall quality score
        if quality_scores:
            validation_results['overall_quality'] = np.mean(quality_scores)
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        return validation_results
    
    def _validate_column_imputation(self, original_df, imputed_df, column):
        """Validate imputation for single column"""
        original_clean = original_df[column].dropna()
        
        if len(original_clean) < 2:
            return {'quality_score': 0, 'message': 'Insufficient original data'}
        
        # Get imputed values
        if 'Status' in imputed_df.columns:
            imputed_mask = imputed_df['Status'] == 'Imputed'
            imputed_values = imputed_df[imputed_mask][column].dropna()
        else:
            original_nan_mask = original_df[column].isnull()
            imputed_values = imputed_df.loc[original_nan_mask, column].dropna()
        
        if len(imputed_values) == 0:
            return {'quality_score': 1, 'message': 'No values were imputed'}
        
        # Statistical comparison
        original_mean = original_clean.mean()
        original_std = original_clean.std()
        imputed_mean = imputed_values.mean()
        imputed_std = imputed_values.std()
        
        # Quality metrics
        mean_diff_pct = abs(imputed_mean - original_mean) / original_mean * 100 if original_mean != 0 else 0
        std_ratio = imputed_std / original_std if original_std > 0 else 1
        
        # Quality score calculation
        mean_score = max(0, 1 - mean_diff_pct / 50)  # Penalize >50% difference
        std_score = max(0, 1 - abs(1 - std_ratio))  # Penalize std deviation from 1
        
        quality_score = (mean_score + std_score) / 2
        
        return {
            'original_mean': float(original_mean),
            'original_std': float(original_std),
            'imputed_mean': float(imputed_mean),
            'imputed_std': float(imputed_std),
            'mean_difference_pct': float(mean_diff_pct),
            'std_ratio': float(std_ratio),
            'quality_score': float(quality_score),
            'n_imputed': len(imputed_values),
            'n_original': len(original_clean)
        }
    
    def _generate_recommendations(self, validation_results):
        """Generate recommendations based on validation results"""
        recommendations = []
        overall_quality = validation_results.get('overall_quality', 0)
        
        if overall_quality >= 0.8:
            recommendations.append("✅ Imputation quality is excellent. Results are reliable.")
        elif overall_quality >= 0.6:
            recommendations.append("⚠️ Imputation quality is good but could be improved.")
            recommendations.append("💡 Consider using different imputation method or parameters.")
        else:
            recommendations.append("❌ Imputation quality is poor. Results may not be reliable.")
            recommendations.append("💡 Try different imputation strategy or collect more data.")
        
        # Column-specific recommendations
        poor_columns = []
        for col, metrics in validation_results.get('column_quality', {}).items():
            if metrics.get('quality_score', 0) < 0.5:
                poor_columns.append(col)
        
        if poor_columns:
            recommendations.append(f"⚠️ Poor imputation quality in columns: {', '.join(poor_columns)}")
        
        return recommendations

# Utility functions
def generate_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())

def handle_uploaded_file(uploaded_file):
    """Handle file upload and return file path"""
    session_id = generate_session_id()
    upload_dir = f'media/uploads/{session_id}'
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    return file_path, session_id

def create_export_file(df, session, export_format='xlsx', include_metadata=True):
    """Create export file with results"""
    export_dir = f'media/exports/{session.session_id}'
    os.makedirs(export_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{session.file_name.split('.')[0]}_imputed_{timestamp}"
    
    if export_format == 'xlsx':
        file_path = os.path.join(export_dir, f"{base_filename}.xlsx")
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Imputed_Data', index=False)
            
            if include_metadata and hasattr(session, 'result'):
                # Add metadata sheet
                metadata = {
                    'Processing_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Original_File': [session.file_name],
                    'Simulations': [session.n_simulations],
                    'Method': [session.imputation_method],
                    'Total_Rows': [session.total_rows],
                    'Imputed_Values': [session.result.imputed_values_count if hasattr(session.result, 'imputed_values_count') else 0]
                }
                metadata_df = pd.DataFrame(metadata)
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    elif export_format == 'csv':
        file_path = os.path.join(export_dir, f"{base_filename}.csv")
        df.to_csv(file_path, index=False)
    
    elif export_format == 'json':
        file_path = os.path.join(export_dir, f"{base_filename}.json")
        df.to_json(file_path, orient='records', date_format='iso', indent=2)
    
    return file_path

def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def log_user_activity(request, session_id, activity_type, description):
    """Log user activity"""
    from .models import UserActivity
    
    try:
        UserActivity.objects.create(
            session_id=session_id,
            activity_type=activity_type,
            activity_description=description,
            ip_address=get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', '')[:500]
        )
    except Exception as e:
        logger.warning(f"Failed to log user activity: {str(e)}")