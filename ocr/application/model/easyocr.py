import easyocr
from application.model.modelbase import ModelBase

from typing import List


class EasyOCR(ModelBase):
    """
    OCR model implementation using EasyOCR for text recognition.
    """

    def __init__(self):
        """
        Initialize the EasyOCR model with English language support.
        """

        super().__init__("EasyOCR")
        self.reader = easyocr.Reader(["pl"])

    def perform_ocr(self, input_path) -> dict[str, List]:
        """
        Perform OCR on an image using EasyOCR and return text predictions with confidence scores.
        :param input_path: Path to the image file on which to perform OCR.
        :return: Dictionary containing "text_predictions" list of recognized text strings and "confidence_scores" list of confidence values.
        """

        model_return_data = self.reader.readtext(input_path)

        return_dict = {
            "text_predictions": [
                internal_list[1] for internal_list in model_return_data
            ],
            "confidence_scores": [
                internal_list[2] for internal_list in model_return_data
            ],
        }

        return return_dict
