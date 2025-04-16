import os
import time
from PIL import Image
from PIL.ImagePath import Path
from django.core.files.storage import FileSystemStorage

from application.model.trocr import TrOCR

model = TrOCR()

def handle_uploaded_file(file):
    # Get the absolute path from environment variable
    upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])
    print(f"Upload directory: {upload_dir}")

    # Save the uploaded file first
    storage = FileSystemStorage(location=upload_dir)
    file_path = storage.save(file.name, file)
    full_path = storage.path(file_path)
    print(f"Saved file to: {full_path}")

    # Wait for file to be fully written
    while not os.path.exists(full_path):
        print("Not yet!")
        time.sleep(0.1)

    time.sleep(0.5)

    # Create reversed_images directory if it doesn't exist
    reversed_dir = os.path.join(upload_dir, 'reversed_images')
    os.makedirs(reversed_dir, exist_ok=True)
    print(f"Created reversed images directory: {reversed_dir}")
    # Create output directory if it doesn't exist
    output_dir = os.path.join(upload_dir, 'processed_text')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Create output filename
        output_filename = f"{os.path.splitext(file.name)[0]}_out.txt"
        output_path = os.path.join(output_dir, output_filename)

        # Process the single uploaded file
        if model.perform_ocr(full_path, output_path):
            print("Image processing completed successfully")
        else:
            print("Failed to process image")

    except Exception as e:
        print(f"Unexpected error: {e}")


def reverse_colors(input_path, output_path=None):
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size
            pixels = img.load()
            reversed_img = Image.new('RGB', (width, height))
            reversed_pixels = reversed_img.load()

            for x in range(width):
                for y in range(height):
                    r, g, b = pixels[x, y]
                    reversed_pixels[x, y] = (255 - r, 255 - g, 255 - b)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            reversed_img.save(output_path)
            print(f"Successfully saved reversed image to: {output_path}")

    except Exception as e:
        print(f"Error processing image {input_path}: {str(e)}")
        return False
    return True


def process_directory(input_dir, output_dir):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

    for file in os.listdir(input_dir):
        if Path(file).suffix.lower() in image_extensions:
            input_path = os.path.join(input_dir, file)
            output_filename = f"{Path(file).stem}_reversed{Path(file).suffix}"
            output_path = os.path.join(output_dir, output_filename)

            if reverse_colors(input_path, output_path):
                print("Success")
            else:
                print("Failed")