from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.core.exceptions import ValidationError
import ipaddress

# Model dataLimit
class dataLimit(models.Model):
    value = models.IntegerField(default=0, help_text="Wartość limitu danych")

    def __str__(self):
        return f"Data limit: {self.value}"

# Model fileLimit
class fileLimit(models.Model):
    value = models.IntegerField(default=0, help_text="Wartość limitu plików")

    def __str__(self):
        return f"File limit: {self.value}"

# Model blockList
class blockList(models.Model):
    ip_address = models.GenericIPAddressField(unique=True, help_text="Adres IP do zablokowania")

    def clean(self):
        try:
            ipaddress.ip_address(self.ip_address)  # Sprawdzenie poprawności adresu IP
        except ValueError:
            raise ValidationError(f"{self.ip_address} nie jest poprawnym adresem IP.")

    def __str__(self):
        return f"Zablokowany IP: {self.ip_address}"

# Model to store uploaded files
class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class SupportTicket(models.Model):
    name = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    message = models.TextField()