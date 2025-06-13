from django.db import models


class FirebaseDataLimit(models.Model):
    class Meta:
        managed = False
        verbose_name = "Limit danych"
        verbose_name_plural = "Limit danych"


class FirebaseFileLimit(models.Model):
    class Meta:
        managed = False
        verbose_name = "Limit plików"
        verbose_name_plural = "Limit plików"


# Model to store uploaded files
class UploadedFile(models.Model):
    file = models.FileField(upload_to="uploads/")
    uploaded_at = models.DateTimeField(auto_now_add=True)


class SupportTicket(models.Model):
    name = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    message = models.TextField()
