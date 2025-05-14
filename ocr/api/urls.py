from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .services import handle_uploaded_file
from .views import UploadedFileViewSet, DataLimitViewSet, FileLimitViewSet, BlockListViewSet, SupportTicketViewSet, CSRFView
from . import views

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
    path('contact/', views.ContactAPIView.as_view(), name='contact_api'),
] 