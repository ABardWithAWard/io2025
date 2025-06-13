from abc import ABC, abstractmethod
from typing import List


class ModelBase(ABC):
    def __init__(self, name):
        """
        Constructor which initializes the model and gets everything ready for running after upload.
        """
        self.name = name

    @abstractmethod
    def perform_ocr(self, input_path) -> dict[str, List]:
        """
        Function that takes in the path of an image file and outputs a pair of predictions and confidence results.
        Inputs:
            input_path: Path to the image on which to perform OCR.
            kwargs: Additional arguments dictionary.
        Returns:
            A dict with keys "text_predictions" and "confidence_scores" containing two lists:
                "text_predictions" - List[str]: the word/text sequence predictions
                "confidence_scores" - List[float]: the corresponding confidence values
        """
