from django.db import models


class FirebaseDataLimit(models.Model):
    """
    Django model representing Firebase data limit settings.
    """

    class Meta:
        managed = False
        verbose_name = "Limit danych"
        verbose_name_plural = "Limit danych"


class FirebaseFileLimit(models.Model):
    """
    Django model representing Firebase file limit settings.
    """

    class Meta:
        managed = False
        verbose_name = "Limit plików"
        verbose_name_plural = "Limit plików"


class UploadedFile(models.Model):
    """
    Django model to store information about uploaded files.
    """

    file = models.FileField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)


class SupportTicket(models.Model):
    """
    Django model to store support ticket information submitted by users.
    """

    name = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    message = models.TextField()
