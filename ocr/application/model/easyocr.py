import easyocr
from application.model.modelbase import ModelBase

from typing import List

class EasyOCR(ModelBase):
    def __init__(self):
        super().__init__("EasyOCR")
        self.reader = easyocr.Reader(['en'])

    def perform_ocr(self, input_path) -> dict[str, List]:
        model_return_data = self.reader.readtext(input_path)

        return_dict = {"text_predictions": [internal_list[1] for internal_list in model_return_data],
                       "confidence_scores": [internal_list[2] for internal_list in model_return_data]}

        return return_dict
