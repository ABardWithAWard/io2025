import os
import numpy as np
from PIL import Image
from django.core.exceptions import ValidationError


def validate_image_file(value):
    """
    Validate that an uploaded file has an allowed image extension.
    :param value: The uploaded file object to validate.
    """
    allowed_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]

    ext = os.path.splitext(value.name)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError("Only image files are allowed.")


def validate_image_brightness(image_path, dark_threshold=30, bright_threshold=250):
    """
    Check if an image has usable brightness by analyzing its average lightness values.
    :param image_path: Path to the image file to analyze.
    :param dark_threshold: Below this average lightness value, image is considered too dark (0-255).
    :param bright_threshold: Above this average lightness value, image is considered too bright (0-255).
    :return: True if image has usable brightness, False if too dark or too bright.
    """
    image = Image.open(image_path)

    # Convert to grayscale to get lightness values
    grayscale = image.convert("L")

    # Convert to numpy array for efficient calculation
    pixels = np.array(grayscale)

    # Calculate average lightness
    average_lightness = np.mean(pixels)

    # Check if within usable range
    return dark_threshold <= average_lightness <= bright_threshold
