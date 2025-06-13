from django.apps import AppConfig
from django.db.utils import OperationalError, ProgrammingError
from django.contrib.auth import get_user_model
from .services import get_limits, set_data_limit, set_file_limit


class ApplicationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "application"

    def ready(self):
        self.unregister_admin_models()

        try:
            from application.services import setup_firestore_db

            setup_firestore_db()
        except (OperationalError, ProgrammingError) as e:
            print(f"Error: {e}")

    def unregister_admin_models(self):
        from django.contrib import admin
        from django.contrib.auth.models import Group, User

        try:
            admin.site.unregister(Group)
        except admin.sites.NotRegistered:
            pass

        try:
            admin.site.unregister(User)
        except admin.sites.NotRegistered:
            pass

        try:
            from social_django.models import Association, Nonce, UserSocialAuth

            try:
                admin.site.unregister(Association)
            except admin.sites.NotRegistered:
                pass

            try:
                admin.site.unregister(Nonce)
            except admin.sites.NotRegistered:
                pass

            try:
                admin.site.unregister(UserSocialAuth)
            except admin.sites.NotRegistered:
                pass

        except ImportError:
            pass
