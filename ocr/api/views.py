import os

from django.db.migrations import serializer
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
from django.utils.decorators import method_decorator
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie
from application.models import dataLimit, fileLimit, blockList, UploadedFile, SupportTicket
from application.forms import UploadFileForm, SubmitTicketForm
from .services import handle_uploaded_file, get_files
from .serializers import (
    DataLimitSerializer, FileLimitSerializer, BlockListSerializer,
    UploadedFileSerializer, SupportTicketSerializer
)
from django.http import JsonResponse

class DataLimitViewSet(viewsets.ModelViewSet):
    queryset = dataLimit.objects.all()
    serializer_class = DataLimitSerializer

class FileLimitViewSet(viewsets.ModelViewSet):
    queryset = fileLimit.objects.all()
    serializer_class = FileLimitSerializer

class BlockListViewSet(viewsets.ModelViewSet):
    queryset = blockList.objects.all()
    serializer_class = BlockListSerializer

class UploadedFileViewSet(viewsets.ModelViewSet):
    queryset = UploadedFile.objects.all()
    serializer_class = UploadedFileSerializer
    permission_classes = [AllowAny]

    @action(detail=False, methods=['post'])
    def upload(self, request):
        print("FILES:", request.FILES)
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file_instance = handle_uploaded_file(request.FILES["file"])
            serializer = self.get_serializer(file_instance)
            return Response({'status': 'success', 'file': serializer.data})
        return Response({'status': 'error', 'errors': form.errors}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def list_files(self, request):
        files = get_files()
        return Response(files)

class SupportTicketViewSet(viewsets.ModelViewSet):
    queryset = SupportTicket.objects.all()
    serializer_class = SupportTicketSerializer

    def create(self, request):
        form = SubmitTicketForm(request.data)
        if form.is_valid():
            ticket = form.save()
            return Response({'status': 'success'})
        return Response({'status': 'error', 'errors': form.errors}, status=status.HTTP_400_BAD_REQUEST)

class CSRFView(APIView):
    permission_classes = [AllowAny]
    @method_decorator(ensure_csrf_cookie)
    def get(self, request):
        return JsonResponse({'csrf_token': get_token(request)}) 