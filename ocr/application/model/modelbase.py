from abc import ABC, abstractmethod

class ModelBase(ABC):
    def __init__(self, name):
        """
        Constructor which initializes the model and gets everything ready for running after upload.
        """
        self.name = name

    @abstractmethod
    def perform_ocr(self, input_dir, output_dir):
        """
        Function that takes in the directory of the dataset and outputs the recognized text to a directory.
        Inputs:
            input_dir: Directory of the dataset with some predefined structure.
            output_dir: Directory where the text will be saved.
        Returns:
            True if the prediction was successful, False otherwise.
        """