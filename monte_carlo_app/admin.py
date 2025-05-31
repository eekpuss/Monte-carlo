from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from .models import DataSession, ProcessingResult, UserActivity

@admin.register(DataSession)
class DataSessionAdmin(admin.ModelAdmin):
    list_display = [
        'file_name', 'session_id_short', 'processing_status_badge', 
        'total_rows', 'total_columns', 'total_missing_values',
        'uploaded_at', 'is_processed'
    ]
    list_filter = [
        'processing_status', 'is_processed', 'is_data_confirmed',
        'imputation_method', 'uploaded_at'
    ]
    search_fields = ['file_name', 'session_id']
    readonly_fields = [
        'session_id', 'uploaded_at', 'updated_at', 'file_size',
        'column_info', 'missing_values_summary'
    ]
    
    fieldsets = (
        ('File Information', {
            'fields': ('session_id', 'file_name', 'file_path', 'file_size', 'sheet_name')
        }),
        ('Data Structure', {
            'fields': ('total_rows', 'total_columns', 'column_info', 'missing_values_summary')
        }),
        ('Processing Configuration', {
            'fields': ('time_column', 'data_frequency', 'n_simulations', 'imputation_method')
        }),
        ('Status', {
            'fields': ('processing_status', 'is_data_confirmed', 'is_processed')
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def session_id_short(self, obj):
        return f"{obj.session_id[:8]}..."
    session_id_short.short_description = 'Session ID'
    
    def processing_status_badge(self, obj):
        colors = {
            'uploaded': 'secondary',
            'analyzed': 'info',
            'confirmed': 'primary',
            'configured': 'warning',
            'processing': 'warning',
            'completed': 'success',
            'error': 'danger'
        }
        color = colors.get(obj.processing_status, 'secondary')
        return format_html(
            '<span class="badge bg-{}">{}</span>',
            color,
            obj.get_processing_status_display()
        )
    processing_status_badge.short_description = 'Status'
    
    def total_missing_values(self, obj):
        return obj.get_total_missing_values()
    total_missing_values.short_description = 'Missing Values'
    
    def column_info(self, obj):
        columns = obj.get_column_names()
        numeric_columns = obj.get_numeric_columns()
        
        html = '<div class="column-info">'
        html += f'<p><strong>All Columns ({len(columns)}):</strong></p>'
        html += '<ul>'
        for col in columns:
            if col in numeric_columns:
                html += f'<li>{col} <span class="badge bg-primary">Numeric</span></li>'
            else:
                html += f'<li>{col}</li>'
        html += '</ul>'
        html += '</div>'
        
        return format_html(html)
    column_info.short_description = 'Columns'
    
    def missing_values_summary(self, obj):
        missing_values = obj.get_missing_values()
        if not missing_values:
            return "No missing values"
        
        html = '<div class="missing-values-summary">'
        for col, count in missing_values.items():
            if count > 0:
                percentage = (count / obj.total_rows) * 100 if obj.total_rows > 0 else 0
                html += f'<div>{col}: {count} ({percentage:.1f}%)</div>'
        html += '</div>'
        
        return format_html(html)
    missing_values_summary.short_description = 'Missing Values Details'
    
    actions = ['mark_as_error', 'reset_processing_status']
    
    def mark_as_error(self, request, queryset):
        queryset.update(processing_status='error')
        self.message_user(request, f"{queryset.count()} sessions marked as error.")
    mark_as_error.short_description = "Mark selected sessions as error"
    
    def reset_processing_status(self, request, queryset):
        queryset.update(processing_status='uploaded', is_processed=False)
        self.message_user(request, f"{queryset.count()} sessions reset to uploaded status.")
    reset_processing_status.short_description = "Reset processing status"

@admin.register(ProcessingResult)
class ProcessingResultAdmin(admin.ModelAdmin):
    list_display = [
        'session_file_name', 'processing_time', 'imputed_values_count',
        'quality_score_badge', 'created_at'
    ]
    list_filter = ['created_at', 'imputation_quality_score']
    search_fields = ['session__file_name', 'session__session_id']
    readonly_fields = [
        'created_at', 'updated_at', 'processing_summary_display',
        'validation_results_display'
    ]
    
    fieldsets = (
        ('Session Information', {
            'fields': ('session',)
        }),
        ('Processing Metrics', {
            'fields': ('processing_time', 'imputed_values_count', 'imputation_quality_score')
        }),
        ('Results', {
            'fields': ('result_file_path', 'visualization_path')
        }),
        ('Details', {
            'fields': ('processing_summary_display', 'validation_results_display'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def session_file_name(self, obj):
        return obj.session.file_name
    session_file_name.short_description = 'File Name'
    
    def quality_score_badge(self, obj):
        score = obj.imputation_quality_score or 0
        if score >= 0.9:
            color = 'success'
            level = 'Excellent'
        elif score >= 0.7:
            color = 'info'
            level = 'Good'
        elif score >= 0.5:
            color = 'warning'
            level = 'Fair'
        else:
            color = 'danger'
            level = 'Poor'
            
        return format_html(
            '<span class="badge bg-{}">{:.2f} ({})</span>',
            color, score, level
        )
    quality_score_badge.short_description = 'Quality Score'
    
    def processing_summary_display(self, obj):
        summary = obj.get_processing_summary()
        if not summary:
            return "No summary available"
        
        html = '<div class="processing-summary">'
        for key, value in summary.items():
            html += f'<div><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        html += '</div>'
        
        return format_html(html)
    processing_summary_display.short_description = 'Processing Summary'
    
    def validation_results_display(self, obj):
        validation = obj.get_validation_results()
        if not validation:
            return "No validation results"
        
        html = '<div class="validation-results">'
        
        # Overall quality
        overall_quality = validation.get('overall_quality', 0)
        html += f'<div><strong>Overall Quality:</strong> {overall_quality:.3f}</div><br>'
        
        # Column quality
        column_quality = validation.get('column_quality', {})
        if column_quality:
            html += '<strong>Column Quality:</strong><ul>'
            for col, metrics in column_quality.items():
                quality_score = metrics.get('quality_score', 0)
                html += f'<li>{col}: {quality_score:.3f}</li>'
            html += '</ul>'
        
        # Recommendations
        recommendations = validation.get('recommendations', [])
        if recommendations:
            html += '<strong>Recommendations:</strong><ul>'
            for rec in recommendations:
                html += f'<li>{rec}</li>'
            html += '</ul>'
        
        html += '</div>'
        
        return format_html(html)
    validation_results_display.short_description = 'Validation Results'

@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = [
        'session_id_short', 'activity_type', 'activity_description_short',
        'ip_address', 'timestamp'
    ]
    list_filter = ['activity_type', 'timestamp']
    search_fields = ['session_id', 'activity_description', 'ip_address']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'
    
    def session_id_short(self, obj):
        return f"{obj.session_id[:8]}..."
    session_id_short.short_description = 'Session ID'
    
    def activity_description_short(self, obj):
        return obj.activity_description[:50] + "..." if len(obj.activity_description) > 50 else obj.activity_description
    activity_description_short.short_description = 'Description'
    
    def has_add_permission(self, request):
        return False  # Prevent manual addition of activity logs
    
    def has_change_permission(self, request, obj=None):
        return False  # Make read-only

# Custom admin site configuration
admin.site.site_header = "Monte Carlo Imputation Admin"
admin.site.site_title = "Monte Carlo Admin"
admin.site.index_title = "Welcome to Monte Carlo Imputation Administration"

# Add custom CSS for admin interface
class AdminConfig:
    def __init__(self):
        pass
    
    @staticmethod
    def get_admin_css():
        return """
        <style>
        .column-info ul { margin: 0; padding-left: 20px; }
        .column-info .badge { font-size: 0.7em; margin-left: 5px; }
        .missing-values-summary div { margin-bottom: 2px; }
        .processing-summary div, .validation-results div { margin-bottom: 5px; }
        .validation-results ul { margin: 5px 0; padding-left: 20px; }
        .badge { 
            display: inline-block; 
            padding: 0.25em 0.4em; 
            font-size: 0.75em; 
            font-weight: 700; 
            line-height: 1; 
            text-align: center; 
            white-space: nowrap; 
            vertical-align: baseline; 
            border-radius: 0.25rem; 
        }
        .bg-primary { background-color: #007bff !important; color: white; }
        .bg-success { background-color: #28a745 !important; color: white; }
        .bg-warning { background-color: #ffc107 !important; color: black; }
        .bg-danger { background-color: #dc3545 !important; color: white; }
        .bg-info { background-color: #17a2b8 !important; color: white; }
        .bg-secondary { background-color: #6c757d !important; color: white; }
        </style>
        """

# Inject custom CSS into admin
def admin_media_css():
    from django.utils.safestring import mark_safe
    return mark_safe(AdminConfig.get_admin_css())

# Register the CSS injection
admin.site.media_css = admin_media_css