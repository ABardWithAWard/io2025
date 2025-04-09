from abc import ABC, abstractmethod

class ModelBase(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def _preprocess(self, dataset_dir):
        """
        Function that takes in the directory of the dataset and outputs the format the model requires.
        """

    @abstractmethod
    def run_inference(self, input_dir, output_dir):
        """
        Function that takes in the directory of the dataset and outputs the recognized text to a directory.
        """