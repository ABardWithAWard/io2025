from django.apps import AppConfig
from django.db.utils import OperationalError, ProgrammingError
from django.db.utils import OperationalError
from django.contrib.auth import get_user_model


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

            User = get_user_model()
            if not User.objects.filter(username="admin").exists():
                User.objects.create_superuser("admin", "admin@example.com", "admin")

        except (OperationalError, ProgrammingError) as e:
            print(f"Error: {e}")