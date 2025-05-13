"""
Temporary script used for running exploratory tests on models without running Django.
Execute as "Current File" if using the PyCharm IDE.
"""
from trocr import TrOCR
from modelMatthew.model import Model

# Initialise the model
model = Model()

# Test the model
model.perform_ocr("stringent-1.jpg", "")
