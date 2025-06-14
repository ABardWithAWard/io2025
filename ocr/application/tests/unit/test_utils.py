import tempfile
from PIL import Image
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError

from application.utils import validate_image_file, validate_image_brightness


class ValidateImageFileTests(TestCase):
    def test_valid_extensions(self):
        valid_filenames = [
            "image.jpg",
            "image.png",
            "image.jpeg",
            "image.bmp",
            "image.gif",
            "image.tiff",
            "image.webp",
        ]
        for filename in valid_filenames:
            file = SimpleUploadedFile(
                filename, b"somecontent", content_type="image/jpeg"
            )
            try:
                validate_image_file(file)
            except ValidationError:
                self.fail(
                    f"validate_image_file raised ValidationError for valid file: {filename}"
                )

    def test_invalid_extension(self):
        file = SimpleUploadedFile(
            "document.txt", b"text content", content_type="text/plain"
        )
        with self.assertRaises(ValidationError):
            validate_image_file(file)


class ValidateImageBrightnessTests(TestCase):
    def create_temp_image(self, color, size=(10, 10)):
        image = Image.new("RGB", size, color)
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image.save(temp_file.name)
        return temp_file.name

    def test_dark_image(self):
        image_path = self.create_temp_image((10, 10, 10))  # very dark
        self.assertFalse(validate_image_brightness(image_path))

    def test_bright_image_too_bright(self):
        image_path = self.create_temp_image((251, 251, 251))  # very bright
        self.assertFalse(validate_image_brightness(image_path))

    def test_bright_image_limit(self):
        image_path = self.create_temp_image((250, 250, 250))  # very bright
        self.assertTrue(validate_image_brightness(image_path))

    def test_normal_brightness_image(self):
        image_path = self.create_temp_image((120, 120, 120))  # mid-brightness
        self.assertTrue(validate_image_brightness(image_path))
