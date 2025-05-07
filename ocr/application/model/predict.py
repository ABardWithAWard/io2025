"""
Temporary script used for running exploratory tests on models without running Django.
Execute as "Current File" if using the PyCharm IDE.
"""
from trocr import TrOCR

# Initialise the model
model = TrOCR()

# Test the model
model.perform_ocr("datasets")