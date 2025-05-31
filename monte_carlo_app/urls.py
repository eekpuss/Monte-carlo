from django.urls import path
from . import views

app_name = 'monte_carlo_app'

urlpatterns = [
    # Main workflow
    path('', views.home, name='home'),
    path('upload/', views.home, name='upload'),
    path('preview/<str:session_id>/', views.preview_data, name='preview_data'),
    path('edit/<str:session_id>/', views.edit_data, name='edit_data'),
    path('confirm/<str:session_id>/', views.confirm_data, name='confirm_data'),
    path('configure/<str:session_id>/', views.configure, name='configure'),
    path('process/<str:session_id>/', views.process_data, name='process_data'),
    path('results/<str:session_id>/', views.view_results, name='view_results'),
    path('visualization/<str:session_id>/', views.visualization, name='visualization'),
    
    # AJAX endpoints
    path('api/validate-file/', views.validate_file_ajax, name='validate_file_ajax'),
    path('api/update-data/', views.update_data_ajax, name='update_data_ajax'),
    path('api/get-column-info/<str:session_id>/', views.get_column_info_ajax, name='get_column_info_ajax'),
    path('api/preview-imputation/<str:session_id>/', views.preview_imputation_ajax, name='preview_imputation_ajax'),
    path('api/process-status/<str:session_id>/', views.process_status_ajax, name='process_status_ajax'),
    
    # Export and download
    path('export/<str:session_id>/', views.export_results, name='export_results'),
    path('download/<str:session_id>/<str:file_type>/', views.download_file, name='download_file'),
    
    # Utility endpoints
    path('session-list/', views.session_list, name='session_list'),
    path('delete-session/<str:session_id>/', views.delete_session, name='delete_session'),
    path('help/', views.help_page, name='help'),
    path('about/', views.about_page, name='about'),
]