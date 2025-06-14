from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from application.services import get_limits, set_data_limit, set_file_limit


class Command(BaseCommand):
    """
    Django management command to set up initial application data including limits and admin user.
    """

    help = "Set up initial data for the application"

    def handle(self, *args, **options):
        # Set up limits
        limits = get_limits()
        if not limits:
            set_data_limit(300)
            set_file_limit(10)
            self.stdout.write(self.style.SUCCESS("Successfully set initial limits"))

        # Create admin user
        User = get_user_model()
        if not User.objects.filter(username="admin@example.com").exists():
            User.objects.create_superuser(
                "admin@example.com", "admin@example.com", "admin"
            )
            self.stdout.write(self.style.SUCCESS("Successfully created admin user"))
        else:
            self.stdout.write("Admin user already exists")
