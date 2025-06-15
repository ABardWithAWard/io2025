import os
import shutil
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from application.services import handle_uploaded_file


class OCRServiceTests(TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_files")
        os.makedirs(self.test_dir, exist_ok=True)
        os.environ["UPLOADED_FILES"] = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("application.model.easyocr.EasyOCR.perform_ocr")
    @patch("application.model.paddleocr.PaddleOCR.perform_ocr")
    @patch("application.services.validate_image_brightness")
    def test_ocr_success(self, mock_brightness, mock_paddle_ocr, mock_easy_ocr):
        mock_brightness.return_value = True
        mock_paddle_ocr.return_value = {"text_predictions": ["Paddle", "OCR"]}
        mock_easy_ocr.return_value = {"text_predictions": ["Easy", "OCR"]}

        file = SimpleUploadedFile(
            "document.jpg", b"fake image content", content_type="image/jpeg"
        )
        result = handle_uploaded_file(file)

        self.assertEqual(result["status"], "success")
        self.assertIn("text", result)
        self.assertIn("Paddle", result["text"])
        self.assertIn("Easy", result["text"])

    @patch("application.model.easyocr.EasyOCR.perform_ocr")
    @patch("application.model.paddleocr.PaddleOCR.perform_ocr")
    @patch("application.services.validate_image_brightness")
    def test_ocr_empty_results(self, mock_brightness, mock_paddle_ocr, mock_easy_ocr):
        mock_brightness.return_value = True
        mock_paddle_ocr.return_value = {"text_predictions": []}
        mock_easy_ocr.return_value = {"text_predictions": []}

        file = SimpleUploadedFile(
            "empty.jpg", b"fake image content", content_type="image/jpeg"
        )
        result = handle_uploaded_file(file)

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["message"], "No text could be extracted from the image."
        )

    @patch("application.services.validate_image_brightness")
    def test_dark_image(self, mock_brightness):
        mock_brightness.return_value = False

        file = SimpleUploadedFile(
            "dark.jpg", b"fake image content", content_type="image/jpeg"
        )
        result = handle_uploaded_file(file)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Image too dark.")

    @patch("application.services.validate_image_brightness")
    def test_invalid_image(self, mock_brightness):
        mock_brightness.side_effect = Exception("Invalid image")

        file = SimpleUploadedFile(
            "corrupt.jpg", b"not an image", content_type="image/jpeg"
        )
        result = handle_uploaded_file(file)

        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid image", result["message"])
