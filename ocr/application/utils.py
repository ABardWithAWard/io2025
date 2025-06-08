import os
import numpy as np
from PIL import Image
from django.core.exceptions import ValidationError


def validate_image_file(value):
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    ext = os.path.splitext(value.name)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError('Only image files are allowed.')

def validate_image_brightness(image: Image, dark_threshold: int = 30, bright_threshold: int = 250):
    """
    Check if a PIL image has usable brightness (not too dark or too bright).

    Args:
        image (PIL.Image): Input PIL image
        dark_threshold (int): Below this average lightness, image is too dark (0-255)
        bright_threshold (int): Above this average lightness, image is too bright (0-255)

    Returns:
        str: "good" if image has usable brightness, "dark" if too dark and "bright" if too bright
    """
    # Convert to grayscale to get lightness values
    grayscale = image.convert('L')

    # Convert to numpy array for efficient calculation
    pixels = np.array(grayscale)

    # Calculate average lightness
    average_lightness = np.mean(pixels)

    # Check if within usable range
    if average_lightness < dark_threshold:
        return "dark"
    elif average_lightness > bright_threshold:
        return "bright"
    return "good"