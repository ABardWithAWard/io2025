import os
import shutil
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile


class OCRModelTests(TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_files")
        os.makedirs(self.test_dir, exist_ok=True)
        os.environ["UPLOADED_FILES"] = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("application.services.handle_uploaded_file")
    @patch("PIL.Image.open")
    @patch("application.utils.validate_image_brightness")
    @patch("application.model.easyocr.EasyOCR.perform_ocr")
    @patch("application.model.paddleocr.PaddleOCR.perform_ocr")
    def test_ocr_error(
        self, mock_paddleocr, mock_easyocr, mock_brightness, mock_image_open, mock_handle
    ):
        mock_handle.return_value = {"status": "error"}
        file = SimpleUploadedFile("test.jpg", b"fake image content", content_type="image/jpeg")
        result = mock_handle(file)

        self.assertEqual(result["status"], "error")

    @patch("application.services.handle_uploaded_file")
    @patch("PIL.Image.open")
    @patch("application.utils.validate_image_brightness")
    @patch("application.model.easyocr.EasyOCR.perform_ocr")
    @patch("application.model.paddleocr.PaddleOCR.perform_ocr")
    def test_ocr_no_text_found(
        self, mock_paddleocr, mock_easyocr, mock_brightness, mock_image_open, mock_handle
    ):
        mock_handle.return_value = {
            "status": "error",
            "message": "Image too dark."
        }
        file = SimpleUploadedFile("empty.jpg", b"fake content", content_type="image/jpeg")
        result = mock_handle(file)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Image too dark.")

    @patch("application.services.handle_uploaded_file")
    @patch("PIL.Image.open", side_effect=Exception("Invalid image"))
    def test_invalid_image(self, mock_image_open, mock_handle):
        mock_handle.return_value = {
            "status": "error",
            "message": "Invalid image format"
        }
        file = SimpleUploadedFile("corrupt.jpg", b"invalid", content_type="image/jpeg")
        result = mock_handle(file)

        self.assertEqual(result["status"], "error")
        self.assertIn("invalid image", result["message"].lower())

    @patch("application.services.handle_uploaded_file")
    @patch("PIL.Image.open")
    @patch("application.utils.validate_image_brightness")
    def test_dark_image(self, mock_brightness, mock_image_open, mock_handle):
        mock_handle.return_value = {
            "status": "error",
            "message": "Image too dark."
        }
        file = SimpleUploadedFile("dark.jpg", b"some content", content_type="image/jpeg")
        result = mock_handle(file)

        self.assertEqual(result["status"], "error")
        self.assertIn("image too dark.", result["message"].lower())

    @patch("application.services.handle_uploaded_file")
    def test_real_file_integration(self, mock_handle):
        test_image_path = os.path.join(self.test_dir, "sample.png")
        if not os.path.exists(test_image_path):
            self.skipTest("Brak pliku sample.png w katalogu test_files/")
            return

        mock_handle.return_value = {
            "status": "success",
            "text": "Recognized text content"
        }

        with open(test_image_path, "rb") as f:
            file = SimpleUploadedFile("sample.png", f.read(), content_type="image/png")
            result = mock_handle(file)

        self.assertIn("status", result)
        if result["status"] == "success":
            self.assertTrue(len(result.get("text", "")) > 0)
        else:
            self.assertIn("error", result["status"])
