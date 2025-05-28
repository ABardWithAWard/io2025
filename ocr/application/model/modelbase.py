from abc import ABC, abstractmethod

class ModelBase(ABC):
    def __init__(self, name):
        """
        Constructor which initializes the model and gets everything ready for running after upload.
        """
        self.name = name

    @abstractmethod
    def perform_ocr(self, input_path):
        """
        Function that takes in the directory of the dataset and outputs the recognized text to a directory.
        Inputs:
            input_path: Path to the image on which to perform OCR.
            kwargs: Additional arguments dictionary.
        Returns:
            Prediction if the prediction was successful, Error otherwise.
        """