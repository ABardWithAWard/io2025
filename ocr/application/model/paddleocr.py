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
    def __init__(self):
        """
        Constructor which initializes the model and gets everything ready for running after upload.
        """
        super().__init__("PaddleOCR")
        self.model = externalPaddleOCR()

    def _preprocess(self, dataset_dir):
        """
        Function that takes in the directory of the dataset and outputs the format the model requires.
        """

    def perform_ocr(self, input_path) -> dict[str, List]:
        """
        Function that takes in the directory of the dataset and outputs the recognized text to a directory.
        Args
            input_path: Path to the image on which to perform OCR.
        Returns:
            Prediction if the prediction was successful, Error otherwise.
        """
        model_return_data = self.model.predict(input_path)

        return_dict = {
            "text_predictions": [text for text in model_return_data[0]["rec_texts"]],
            "confidence_scores": [
                score for score in model_return_data[0]["rec_scores"]
            ],
        }

        return return_dict
