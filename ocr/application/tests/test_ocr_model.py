from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from pathlib import Path
from io import StringIO
import sys
from application.model.modelMatthew.model import Model
from application.model.trocr import TrOCR


model = TrOCR()
modelMatthew = Model()


class SimpleOCRTestCase(TestCase):

    def get_ocr_result(self, filename):
        full_path = Path(__file__).parent / 'ocrTestFiles' / filename

        print("Full path:", full_path)
        print("File exists:", full_path.exists())

        # modelMatthew._preprocess(full_path)
        result = model.perform_ocr(full_path)

        print("OCR result:", result)
        return result


    def test_invoice(self):
        expected = "Warszawa"
        actual = self.get_ocr_result("Krakow.png")

        print(f"Expected: {expected}")
        print(f"Got:      {actual}")
