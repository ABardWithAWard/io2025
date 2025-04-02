from django import forms
from django.core.exceptions import ValidationError
import os

def validate_image_file(value):
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    ext = os.path.splitext(value.name)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError('Only image files are allowed.')

class UploadFileForm(forms.Form):
    file = forms.FileField(
        validators=[validate_image_file],
        widget=forms.FileInput(attrs={
            'accept': '.png,.jpg,.jpeg,.bmp,.gif,.tiff,.webp'
        })
    )