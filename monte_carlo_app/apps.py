from django.apps import AppConfig

class MonteCarloAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monte_carlo_app'
    verbose_name = 'Monte Carlo Time Series Imputation'
    
    def ready(self):
        """Called when Django starts."""
        # Import signals or perform app initialization here
        pass