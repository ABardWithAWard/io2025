from PIL import Image
import sys
import os
from pathlib import Path


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


# Prevents misuse by running it via import
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python color_reverse.py input_directory output_directory")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"Error: {output_dir} is not a valid directory")
        sys.exit(1)

    process_directory(input_dir, output_dir)
