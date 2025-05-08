from rest_framework import serializers
from application.models import dataLimit, fileLimit, blockList, UploadedFile, SupportTicket

class DataLimitSerializer(serializers.ModelSerializer):
    class Meta:
        model = dataLimit
        fields = ['id', 'value']

class FileLimitSerializer(serializers.ModelSerializer):
    class Meta:
        model = fileLimit
        fields = ['id', 'value']

class BlockListSerializer(serializers.ModelSerializer):
    class Meta:
        model = blockList
        fields = ['id', 'ip_address']

class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ['id', 'file', 'uploaded_at']
        read_only_fields = ['uploaded_at']

class SupportTicketSerializer(serializers.ModelSerializer):
    class Meta:
        model = SupportTicket
        fields = ['id', 'name', 'email', 'message'] 