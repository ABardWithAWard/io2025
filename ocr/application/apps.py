from django.apps import AppConfig
from django.db.utils import OperationalError, ProgrammingError


class ApplicationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "application"

    def ready(self):

        try:
            from application.models import dataLimit, fileLimit

            if not dataLimit.objects.exists():
                dataLimit.objects.create(value=0)

            if not fileLimit.objects.exists():
                fileLimit.objects.create(value=0)

        except (OperationalError, ProgrammingError) as e:
            print(f"Błąd podczas tworzenia rekordów: {e}")