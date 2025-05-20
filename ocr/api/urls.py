from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .services import handle_uploaded_file
from .views import (
    UploadedFileViewSet, DataLimitViewSet, FileLimitViewSet, 
    BlockListViewSet, SupportTicketViewSet, CSRFView,
    ContactAPIView, LoginAPIView, GoogleAuthAPIView, LogoutAPIView,
    RegisterAPIView, AuthStatusAPIView
)

# Create a router and register our viewsets with it
router = DefaultRouter()
router.register(r'files', UploadedFileViewSet, basename='list_files')
router.register(r'data-limits', DataLimitViewSet)
router.register(r'file-limits', FileLimitViewSet)
router.register(r'block-lists', BlockListViewSet)
router.register(r'support-tickets', SupportTicketViewSet)
router.register(r'upload', UploadedFileViewSet, basename='upload')

urlpatterns = [
    path('', include(router.urls)),
    path('csrf-token/', CSRFView.as_view(), name='get_csrf_token'),
    path('contact/', ContactAPIView.as_view(), name='contact_api'),
    path('login/', LoginAPIView.as_view(), name='login_api'),
    path('register/', RegisterAPIView.as_view(), name='register_api'),
    path('google-auth/', GoogleAuthAPIView.as_view(), name='google_auth_api'),
    path('logout/', LogoutAPIView.as_view(), name='logout_api'),
    path('auth-status/', AuthStatusAPIView.as_view(), name='auth_status_api'),
] 