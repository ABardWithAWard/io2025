from rest_framework import serializers
from application.models import UploadedFile, SupportTicket

class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ['id', 'file', 'uploaded_at']
        read_only_fields = ['uploaded_at']

class SupportTicketSerializer(serializers.ModelSerializer):
    class Meta:
        model = SupportTicket
        fields = ['id', 'name', 'email', 'message'] 