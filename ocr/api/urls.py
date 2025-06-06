from django.urls import path, include, re_path
from rest_framework.routers import DefaultRouter
from django.contrib import admin
from .views import (
    UploadedFileViewSet, SupportTicketViewSet, CSRFView,
    ContactAPIView, LoginAPIView, GoogleAuthAPIView, LogoutAPIView,
    RegisterAPIView, AuthStatusAPIView, ReactAppView, GlobalSettingsAPIView
)

# Create a router and register our viewsets with it
router = DefaultRouter()
router.register(r'files', UploadedFileViewSet, basename='list_files')
router.register(r'support-tickets', SupportTicketViewSet)
router.register(r'upload', UploadedFileViewSet, basename='upload')

app_name = 'application'

urlpatterns = [
    path('', include(router.urls)),
    path('csrf-token/', CSRFView.as_view(), name='get_csrf_token'),
    path('contact/', ContactAPIView.as_view(), name='contact_api'),
    path('login/', LoginAPIView.as_view(), name='login_api'),
    path('register/', RegisterAPIView.as_view(), name='register_api'),
    path('google-auth/', GoogleAuthAPIView.as_view(), name='google_auth_api'),
    path('logout/', LogoutAPIView.as_view(), name='logout_api'),
    path('auth-status/', AuthStatusAPIView.as_view(), name='auth_status_api'),
    path('global-settings/', GlobalSettingsAPIView.as_view(), name='global_settings_api'),
    path('admin/', admin.site.urls),  # Django admin
    re_path(r'^(?!api/|admin/|media/).*$', ReactAppView.as_view(), name='react_app'),  # All other routes go to React
] 