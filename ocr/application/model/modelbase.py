from abc import ABC, abstractmethod
from typing import List


class ModelBase(ABC):
    """
    Abstract base class for OCR model implementations.
    """

    def __init__(self, name):
        """
        Initialize the model with a given name.
        :param name: String identifier for the model type.
        """
        self.name = name

    @abstractmethod
    def perform_ocr(self, input_path) -> dict[str, List]:
        """
        Perform OCR on an image file and return text predictions with confidence scores.
        :param input_path: Path to the image file on which to perform OCR.
        :return: Dictionary containing "text_predictions" list of strings and "confidence_scores" list of floats.
        """
