from django.apps import AppConfig
from django.db.utils import OperationalError, ProgrammingError


class ApplicationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "application"

    def ready(self):
        try:
            from application.services import setup_firestore_db

            setup_firestore_db()

        except (OperationalError, ProgrammingError) as e:
            print(f"Error: {e}")
