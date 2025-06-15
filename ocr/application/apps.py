from django.apps import AppConfig
from django.db.utils import OperationalError, ProgrammingError


class ApplicationConfig(AppConfig):
    """
    Django application configuration for the application app.
    Initializes Firestore database connection when the application starts.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "application"

    def ready(self):
        try:
            from application.services import setup_firestore_db

            setup_firestore_db()

        except (OperationalError, ProgrammingError) as e:
            print(f"Error: {e}")
