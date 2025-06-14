import unittest
from unittest.mock import patch, MagicMock
from application.model.paddleocr import PaddleOCR
from application.model.easyocr import EasyOCR


class TestPaddleOCR(unittest.TestCase):

    @patch("application.model.paddleocr.externalPaddleOCR")
    def test_perform_ocr_success(self, mock_external_paddleocr):
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [
            {"rec_texts": ["Hello", "World"], "rec_scores": [0.99, 0.95]}
        ]
        mock_external_paddleocr.return_value = mock_model_instance

        model = PaddleOCR()
        result = model.perform_ocr("dummy_path.jpg")

        self.assertEqual(result["text_predictions"], ["Hello", "World"])
        self.assertEqual(result["confidence_scores"], [0.99, 0.95])

    @patch("application.model.paddleocr.externalPaddleOCR")
    def test_model_initialization(self, mock_external_paddleocr):
        model = PaddleOCR()
        mock_external_paddleocr.assert_called_once()
        self.assertIsNotNone(model.model)

    @patch("application.model.paddleocr.externalPaddleOCR")
    def test_perform_ocr_empty_result(self, mock_external_paddleocr):
        mock_model_instance = MagicMock()
        mock_model_instance.predict.return_value = [{"rec_texts": [], "rec_scores": []}]
        mock_external_paddleocr.return_value = mock_model_instance

        model = PaddleOCR()
        result = model.perform_ocr("dummy_path.jpg")

        self.assertEqual(result["text_predictions"], [])
        self.assertEqual(result["confidence_scores"], [])


class TestEasyOCR(unittest.TestCase):

    @patch("application.model.easyocr.easyocr.Reader")
    def test_perform_ocr_success(self, mock_reader):
        # Tworzymy mock instancję
        mock_model_instance = MagicMock()
        mock_model_instance.readtext.return_value = [
            ([(0, 0), (1, 0), (1, 1), (0, 1)], "Hello", 0.99),
            ([(0, 1), (1, 1), (1, 2), (0, 2)], "World", 0.95),
        ]
        # Ustawiamy, żeby Reader() zwracał naszą mock instancję
        mock_reader.return_value = mock_model_instance

        model = EasyOCR()
        result = model.perform_ocr("dummy_path.jpg")

        self.assertEqual(result["text_predictions"], ["Hello", "World"])
        self.assertEqual(result["confidence_scores"], [0.99, 0.95])

    @patch("application.model.easyocr.easyocr.Reader")
    def test_model_initialization(self, mock_reader):
        model = EasyOCR()
        mock_reader.assert_called_once_with(["en"])
        self.assertIsNotNone(model.reader)

    @patch("application.model.easyocr.easyocr.Reader")
    def test_perform_ocr_empty_result(self, mock_reader):
        mock_model_instance = MagicMock()
        mock_model_instance.readtext.return_value = []
        mock_reader.return_value = mock_model_instance

        model = EasyOCR()
        result = model.perform_ocr("dummy_path.jpg")

        self.assertEqual(result["text_predictions"], [])
        self.assertEqual(result["confidence_scores"], [])
