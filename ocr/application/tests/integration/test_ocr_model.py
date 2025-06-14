import os
import shutil
from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from application.services import handle_uploaded_file


class OCRModelTests(TestCase):
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_files')
        os.makedirs(self.test_dir, exist_ok=True)
        os.environ['UPLOADED_FILES'] = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    def test_ocr_success(self, mock_image_open, mock_ocr):
        mock_ocr.return_value = "Test OCR Result"
        mock_img = MagicMock()
        mock_image_open.return_value = mock_img

        file = SimpleUploadedFile("document.jpg", b"fake image content", content_type="image/jpeg")
        result = handle_uploaded_file(file)

        self.assertEqual(result['status'], 'success')
        self.assertIn('text', result)
        self.assertEqual(result['text'], "Test OCR Result")

    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    def test_ocr_empty_text(self, mock_image_open, mock_ocr):
        mock_ocr.return_value = ""
        mock_img = MagicMock()
        mock_image_open.return_value = mock_img

        file = SimpleUploadedFile("empty.jpg", b"fake image content", content_type="image/jpeg")
        result = handle_uploaded_file(file)

        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['message'], 'No text could be extracted from the image.')

    @patch('PIL.Image.open')
    def test_ocr_invalid_image(self, mock_image_open):
        mock_image_open.side_effect = Exception("Invalid image")

        file = SimpleUploadedFile("corrupt.jpg", b"invalid", content_type="image/jpeg")
        result = handle_uploaded_file(file)

        self.assertEqual(result['status'], 'error')
        self.assertIn('Invalid image', result['message'])

    def test_ocr_real_file(self):
        test_image_path = os.path.join(self.test_dir, 'sample.png')
        if not os.path.exists(test_image_path):
            self.skipTest("Brak pliku sample.png w test_files/")
            return

        with open(test_image_path, 'rb') as f:
            file = SimpleUploadedFile("sample.png", f.read(), content_type="image/png")
            result = handle_uploaded_file(file)

        self.assertEqual(result['status'], 'success')
        self.assertTrue(len(result['text']) > 0)
