from django.db import models
import json
from django.utils import timezone

class DataSession(models.Model):
    """Model untuk menyimpan session data processing"""
    session_id = models.CharField(max_length=100, unique=True, db_index=True)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Data structure info
    total_rows = models.IntegerField(default=0)
    total_columns = models.IntegerField(default=0)
    column_names = models.TextField(blank=True)  # JSON string
    numeric_columns = models.TextField(blank=True)  # JSON string
    missing_values = models.TextField(blank=True)  # JSON string
    data_types = models.TextField(blank=True)  # JSON string
    
    # Processing status
    is_data_confirmed = models.BooleanField(default=False)
    is_processed = models.BooleanField(default=False)
    processing_status = models.CharField(max_length=50, default='uploaded')
    
    # Configuration
    time_column = models.CharField(max_length=100, blank=True)
    data_frequency = models.CharField(max_length=20, default='1min')
    n_simulations = models.IntegerField(default=1000)
    imputation_method = models.CharField(max_length=20, default='adaptive')
    
    # Metadata
    file_size = models.BigIntegerField(default=0)
    sheet_name = models.CharField(max_length=100, default='0')
    
    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Data Session'
        verbose_name_plural = 'Data Sessions'
    
    def __str__(self):
        return f"{self.file_name} - {self.session_id[:8]}..."
    
    def set_column_names(self, columns):
        """Set column names as JSON"""
        self.column_names = json.dumps(list(columns))
    
    def get_column_names(self):
        """Get column names from JSON"""
        try:
            return json.loads(self.column_names) if self.column_names else []
        except json.JSONDecodeError:
            return []
    
    def set_numeric_columns(self, columns):
        """Set numeric columns as JSON"""
        self.numeric_columns = json.dumps(list(columns))
    
    def get_numeric_columns(self):
        """Get numeric columns from JSON"""
        try:
            return json.loads(self.numeric_columns) if self.numeric_columns else []
        except json.JSONDecodeError:
            return []
    
    def set_missing_values(self, missing_dict):
        """Set missing values dictionary as JSON"""
        self.missing_values = json.dumps(missing_dict)
    
    def get_missing_values(self):
        """Get missing values dictionary from JSON"""
        try:
            return json.loads(self.missing_values) if self.missing_values else {}
        except json.JSONDecodeError:
            return {}
    
    def set_data_types(self, types_dict):
        """Set data types dictionary as JSON"""
        self.data_types = json.dumps(types_dict)
    
    def get_data_types(self):
        """Get data types dictionary from JSON"""
        try:
            return json.loads(self.data_types) if self.data_types else {}
        except json.JSONDecodeError:
            return {}
    
    def get_total_missing_values(self):
        """Get total count of missing values"""
        missing_values = self.get_missing_values()
        return sum(missing_values.values()) if missing_values else 0
    
    def get_processing_status_display(self):
        """Get human-readable processing status"""
        status_map = {
            'uploaded': 'File Uploaded',
            'analyzed': 'Data Analyzed',
            'confirmed': 'Data Confirmed',
            'configured': 'Parameters Configured',
            'processing': 'Processing...',
            'completed': 'Completed',
            'error': 'Error Occurred'
        }
        return status_map.get(self.processing_status, self.processing_status.title())

class ProcessingResult(models.Model):
    """Model untuk menyimpan hasil processing"""
    session = models.OneToOneField(DataSession, on_delete=models.CASCADE, related_name='result')
    
    # Data storage (JSON format)
    original_data_sample = models.TextField(blank=True)  # Sample of original data
    imputed_data_sample = models.TextField(blank=True)   # Sample of imputed data
    validation_results = models.TextField(blank=True)    # Validation metrics
    imputation_log = models.TextField(blank=True)        # Detailed imputation log
    processing_summary = models.TextField(blank=True)    # Processing summary
    
    # Processing metrics
    processing_time = models.FloatField(default=0.0)  # Processing time in seconds
    imputed_values_count = models.IntegerField(default=0)
    imputation_quality_score = models.FloatField(null=True, blank=True)
    
    # File exports
    result_file_path = models.CharField(max_length=500, blank=True)
    visualization_path = models.CharField(max_length=500, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Processing Result'
        verbose_name_plural = 'Processing Results'
    
    def __str__(self):
        return f"Result for {self.session.file_name}"
    
    def set_original_data_sample(self, data, max_rows=100):
        """Set sample of original data as JSON"""
        if hasattr(data, 'head'):
            sample_data = data.head(max_rows)
            self.original_data_sample = sample_data.to_json(orient='records', date_format='iso')
        else:
            self.original_data_sample = json.dumps(data[:max_rows] if len(data) > max_rows else data)
    
    def get_original_data_sample(self):
        """Get original data sample from JSON"""
        try:
            import pandas as pd
            if self.original_data_sample:
                return pd.read_json(self.original_data_sample, orient='records')
            return pd.DataFrame()
        except:
            return {}
    
    def set_imputed_data_sample(self, data, max_rows=100):
        """Set sample of imputed data as JSON"""
        if hasattr(data, 'head'):
            sample_data = data.head(max_rows)
            self.imputed_data_sample = sample_data.to_json(orient='records', date_format='iso')
        else:
            self.imputed_data_sample = json.dumps(data[:max_rows] if len(data) > max_rows else data)
    
    def get_imputed_data_sample(self):
        """Get imputed data sample from JSON"""
        try:
            import pandas as pd
            if self.imputed_data_sample:
                return pd.read_json(self.imputed_data_sample, orient='records')
            return pd.DataFrame()
        except:
            return {}
    
    def set_validation_results(self, validation_dict):
        """Set validation results as JSON"""
        self.validation_results = json.dumps(validation_dict)
    
    def get_validation_results(self):
        """Get validation results from JSON"""
        try:
            return json.loads(self.validation_results) if self.validation_results else {}
        except json.JSONDecodeError:
            return {}
    
    def set_imputation_log(self, log_dict):
        """Set imputation log as JSON"""
        self.imputation_log = json.dumps(log_dict)
    
    def get_imputation_log(self):
        """Get imputation log from JSON"""
        try:
            return json.loads(self.imputation_log) if self.imputation_log else {}
        except json.JSONDecodeError:
            return {}
    
    def set_processing_summary(self, summary_dict):
        """Set processing summary as JSON"""
        self.processing_summary = json.dumps(summary_dict)
    
    def get_processing_summary(self):
        """Get processing summary from JSON"""
        try:
            return json.loads(self.processing_summary) if self.processing_summary else {}
        except json.JSONDecodeError:
            return {}

class UserActivity(models.Model):
    """Model untuk tracking user activity"""
    session_id = models.CharField(max_length=100, db_index=True)
    activity_type = models.CharField(max_length=50)
    activity_description = models.TextField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'User Activity'
        verbose_name_plural = 'User Activities'
    
    def __str__(self):
        return f"{self.activity_type} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"