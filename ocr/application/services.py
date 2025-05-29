import os
from pathlib import Path
from django.core.files.storage import FileSystemStorage

from application.model.paddleocr import PaddleOCR
from application.model.easyocr import EasyOCR

paddle_model = PaddleOCR()
easy_model = EasyOCR()

def prepare_file_hierarchy (file):
    """Takes uploaded file and returns directory where it is saved and its detected content"""
    upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])
    print(f"Upload directory: {upload_dir}")

    # Save the uploaded file first
    storage = FileSystemStorage(location=upload_dir)
    file_path = storage.save(file.name, file)
    full_path = storage.path(file_path)
    print(f"Saved file to: {full_path}")

    # Possibly legacy?
    # output_dir = os.path.join(upload_dir, 'processed_text')
    # os.makedirs(output_dir, exist_ok=True)

    return full_path

def handle_uploaded_file(file):
    """Takes file uploaded in form and calls helper function to manage file and its contents"""
    full_path = prepare_file_hierarchy(file)

    paddle_result_list = paddle_model.perform_ocr(input_path=full_path)
    print("PaddleOCR results:")
    print(" ".join([result for result in paddle_result_list[0]["rec_texts"]]))

    easy_result_list = easy_model.perform_ocr(input_path=full_path)
    print("EasyOCR results:")
    print(" ".join([result[1] for result in easy_result_list]))
