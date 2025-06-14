from pathlib import Path
from PIL import Image
from typing import List

import logging
import warnings

# cache requires zlib >=1.2.11,<1.3.0a0, paddle requires zlib ==1.3.1
warnings.filterwarnings("ignore", message="No ccache found.*")

from paddleocr import PaddleOCR as externalPaddleOCR

logging.getLogger("paddlex").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)

from application.model.modelbase import ModelBase


class PaddleOCR(ModelBase):
    """
    OCR model implementation using PaddleOCR for text recognition.
    """

    def __init__(self):
        """
        Initialize the PaddleOCR model and prepare it for text recognition tasks.
        """
        super().__init__("PaddleOCR")
        self.model = externalPaddleOCR()

    def _preprocess(self, dataset_dir):
        """
        Preprocess dataset directory into the format required by the model.
        :param dataset_dir: Directory path containing the dataset to preprocess.
        """

    def perform_ocr(self, input_path) -> dict[str, List]:
        """
        Perform OCR on an image using PaddleOCR and return text predictions with confidence scores.
        :param input_path: Path to the image file on which to perform OCR.
        :return: Dictionary containing "text_predictions" list of recognized text strings and "confidence_scores" list of confidence values.
        """

        model_return_data = self.model.predict(input_path)

        return_dict = {
            "text_predictions": [text for text in model_return_data[0]["rec_texts"]],
            "confidence_scores": [
                score for score in model_return_data[0]["rec_scores"]
            ],
        }

        return return_dict
