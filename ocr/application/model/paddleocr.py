from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR as externalPaddleOCR

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

    def perform_ocr(self, input_path):
        """
        Function that takes in the directory of the dataset and outputs the recognized text to a directory.
        Args
            input_path: Path to the image on which to perform OCR.
        Returns:
            Prediction if the prediction was successful, Error otherwise.
        """
        result = self.model.predict(input_path)
        return result