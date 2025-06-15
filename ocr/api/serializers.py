from rest_framework import serializers
from application.models import UploadedFile, SupportTicket


class UploadedFileSerializer(serializers.ModelSerializer):
    """
    Django REST Framework serializer for UploadedFile model instances.
    """

    class Meta:
        model = UploadedFile
        fields = ["id", "file", "uploaded_at"]
        read_only_fields = ["uploaded_at"]


class SupportTicketSerializer(serializers.ModelSerializer):
    """
    Django REST Framework serializer for SupportTicket model instances.
    """

    class Meta:
        model = SupportTicket
        fields = ["id", "name", "email", "message"]
