import os
import time
from django.core.files.storage import FileSystemStorage

from application.model.trocr import TrOCR
from model import trocr

model = TrOCR()

def handle_uploaded_file(file):
    # Get the absolute path from environment variable
    upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])

    # Save the uploaded file first
    storage = FileSystemStorage(location=upload_dir)
    file_path = storage.save(file.name, file)
    full_path = storage.path(file_path)

    # Wait for file to be fully written
    while not os.path.exists(full_path):
        time.sleep(0.1)

    time.sleep(0.5)

    # Create reversed_images directory if it doesn't exist
    reversed_dir = os.path.join(upload_dir, 'reversed_images')
    os.makedirs(reversed_dir, exist_ok=True)

    try:
        # Create output filename
        output_filename = f"{os.path.splitext(file.name)[0]}_reversed{os.path.splitext(file.name)[1]}"
        output_path = os.path.join(reversed_dir, output_filename)

        # Process the single uploaded file
        if model.perform_ocr(full_path, output_path):
            print("Image processing completed successfully")
        else:
            print("Failed to process image")

    except Exception as e:
        print(f"Unexpected error: {e}")