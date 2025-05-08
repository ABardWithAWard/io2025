from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DataLimitViewSet, FileLimitViewSet, BlockListViewSet,
    UploadedFileViewSet, SupportTicketViewSet, CSRFView
)

# Create a router and register our viewsets with it
router = DefaultRouter()
router.register(r'data-limits', DataLimitViewSet)
router.register(r'file-limits', FileLimitViewSet)
router.register(r'block-list', BlockListViewSet)
router.register(r'files', UploadedFileViewSet, basename='file')
router.register(r'support-tickets', SupportTicketViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('csrf-token/', CSRFView.as_view(), name='get_csrf_token'),
] 