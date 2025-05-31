from django import forms
from django.core.validators import FileExtensionValidator

class FileUploadForm(forms.Form):
    """Form untuk upload file Excel/ODS"""
    file = forms.FileField(
        label='Select Excel or ODS file',
        help_text='Supported formats: .xlsx, .xls, .ods (Max size: 50MB)',
        validators=[
            FileExtensionValidator(
                allowed_extensions=['xlsx', 'xls', 'ods'],
                message='Please upload a valid Excel or ODS file.'
            )
        ],
        widget=forms.FileInput(attrs={
            'class': 'form-control-file',
            'accept': '.xlsx,.xls,.ods',
            'id': 'fileInput',
            'onchange': 'updateFileName(this)'
        })
    )
    
    sheet_name = forms.CharField(
        label='Sheet Name/Index',
        initial='0',
        help_text='Enter sheet name or index (0 for first sheet)',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '0 (first sheet)',
            'id': 'sheetName'
        }),
        required=False
    )
    
    def clean_file(self):
        """Validate uploaded file"""
        file = self.cleaned_data.get('file')
        if file:
            # Check file size (50MB limit)
            if file.size > 50 * 1024 * 1024:
                raise forms.ValidationError('File size cannot exceed 50MB.')
            
            # Check file extension
            allowed_extensions = ['.xlsx', '.xls', '.ods']
            file_extension = '.' + file.name.split('.')[-1].lower()
            if file_extension not in allowed_extensions:
                raise forms.ValidationError('Please upload a valid Excel or ODS file.')
        
        return file
    
    def clean_sheet_name(self):
        """Validate sheet name/index"""
        sheet_name = self.cleaned_data.get('sheet_name', '0').strip()
        if not sheet_name:
            return '0'
        return sheet_name

class DataConfirmationForm(forms.Form):
    """Form untuk konfirmasi data yang sudah diload"""
    time_column = forms.ChoiceField(
        label='Time/Date Column',
        help_text='Select which column contains time/timestamp data (optional)',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'timeColumn'
        }),
        required=False
    )
    
    data_quality_check = forms.BooleanField(
        label='Data quality looks good',
        help_text='Check this if the data preview looks correct',
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'dataQualityCheck'
        })
    )
    
    missing_data_handling = forms.ChoiceField(
        label='Missing Data Handling',
        choices=[
            ('impute', 'Impute missing values using Monte Carlo'),
            ('skip', 'Skip rows with missing values'),
            ('proceed', 'Proceed with current data as-is'),
        ],
        initial='impute',
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        help_text='Choose how to handle missing values in your dataset'
    )
    
    confirm_data = forms.BooleanField(
        label='I confirm this data is correct and ready for processing',
        required=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'confirmData'
        })
    )
    
    def __init__(self, *args, time_column_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if time_column_choices:
            choices = [('', 'No time column')] + [(col, col) for col in time_column_choices]
            self.fields['time_column'].choices = choices

class MonteCarloConfigForm(forms.Form):
    """Form untuk konfigurasi Monte Carlo parameters"""
    n_simulations = forms.IntegerField(
        label='Number of Simulations',
        initial=1000,
        min_value=100,
        max_value=10000,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '100',
            'id': 'nSimulations',
            'oninput': 'updateSimulationDisplay(this.value)'
        }),
        help_text='Number of Monte Carlo simulations (100-10,000). Higher values = more accurate but slower.'
    )
    
    data_frequency = forms.ChoiceField(
        label='Expected Data Frequency',
        choices=[
            ('1min', '1 Minute'),
            ('5min', '5 Minutes'),
            ('10min', '10 Minutes'),
            ('15min', '15 Minutes'),
            ('30min', '30 Minutes'),
            ('1H', '1 Hour'),
            ('2H', '2 Hours'),
            ('6H', '6 Hours'),
            ('12H', '12 Hours'),
            ('1D', '1 Day'),
            ('1W', '1 Week'),
        ],
        initial='1min',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'dataFrequency'
        }),
        help_text='Expected frequency of your time series data (only used if time column is specified)'
    )
    
    imputation_method = forms.ChoiceField(
        label='Imputation Strategy',
        choices=[
            ('adaptive', 'Adaptive (Recommended) - Context-aware imputation'),
            ('simple', 'Simple Global - Use overall statistics'),
            ('rolling', 'Rolling Window - Use recent data trends'),
            ('interpolation', 'Linear Interpolation - Between known values'),
        ],
        initial='adaptive',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'imputationMethod',
            'onchange': 'updateMethodDescription(this.value)'
        }),
        help_text='Method for calculating distribution parameters for missing values'
    )
    
    confidence_level = forms.ChoiceField(
        label='Confidence Level',
        choices=[
            ('90', '90% Confidence'),
            ('95', '95% Confidence (Recommended)'),
            ('99', '99% Confidence'),
        ],
        initial='95',
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'confidenceLevel'
        }),
        help_text='Confidence level for imputation quality assessment'
    )
    
    preserve_patterns = forms.BooleanField(
        label='Preserve Data Patterns',
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'preservePatterns'
        }),
        help_text='Try to maintain temporal patterns and trends in imputed values'
    )
    
    quality_threshold = forms.FloatField(
        label='Quality Threshold',
        initial=0.8,
        min_value=0.1,
        max_value=1.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.1',
            'max': '1.0',
            'id': 'qualityThreshold'
        }),
        help_text='Minimum quality score to accept imputation results (0.1-1.0)'
    )
    
    def clean_n_simulations(self):
        """Validate number of simulations"""
        n_simulations = self.cleaned_data.get('n_simulations')
        if n_simulations and (n_simulations < 100 or n_simulations > 10000):
            raise forms.ValidationError('Number of simulations must be between 100 and 10,000.')
        return n_simulations
    
    def clean_quality_threshold(self):
        """Validate quality threshold"""
        threshold = self.cleaned_data.get('quality_threshold')
        if threshold and (threshold < 0.1 or threshold > 1.0):
            raise forms.ValidationError('Quality threshold must be between 0.1 and 1.0.')
        return threshold

class DataEditForm(forms.Form):
    """Form untuk editing data yang sudah diload"""
    edited_data = forms.CharField(
        widget=forms.HiddenInput(),
        required=False
    )
    
    edit_notes = forms.CharField(
        label='Edit Notes (Optional)',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Describe any manual changes you made to the data...'
        }),
        required=False,
        help_text='Optional notes about your data modifications'
    )

class ExportForm(forms.Form):
    """Form untuk export hasil"""
    export_format = forms.ChoiceField(
        label='Export Format',
        choices=[
            ('xlsx', 'Excel (.xlsx)'),
            ('csv', 'CSV (.csv)'),
            ('json', 'JSON (.json)'),
        ],
        initial='xlsx',
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        help_text='Choose format for downloading results'
    )
    
    include_original = forms.BooleanField(
        label='Include original data',
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Include original data alongside imputed values'
    )
    
    include_metadata = forms.BooleanField(
        label='Include processing metadata',
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Include imputation log and validation results'
    )
    
    separate_sheets = forms.BooleanField(
        label='Use separate sheets/files',
        initial=False,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Export original and imputed data in separate sheets (Excel) or files'
    )