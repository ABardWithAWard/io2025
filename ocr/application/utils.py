import os
from django.core.exceptions import ValidationError


def validate_image_file(value):
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    ext = os.path.splitext(value.name)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError('Only image files are allowed.')
