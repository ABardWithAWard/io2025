from PIL import Image
import sys

def reverse_colors(input_path, output_path=None):
    try:
        with Image.open(input_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            pixels = img.load()
            
            # Create a new image with the same size
            reversed_img = Image.new('RGB', (width, height))
            reversed_pixels = reversed_img.load()
            
            # Reverse each pixel's color
            for x in range(width):
                for y in range(height):
                    r, g, b = pixels[x, y]
                    # Reverse each color channel (255 - value)
                    reversed_pixels[x, y] = (255 - r, 255 - g, 255 - b)
            
            # Generate output path if not provided
            if output_path is None:
                input_name = input_path.rsplit('.', 1)[0]
                output_path = f"{input_name}_reversed.png"
            
            # Save the reversed image
            reversed_img.save(output_path)
            print(f"Successfully created reversed image: {output_path}")
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python color_reverse.py input.png [output.png]")
    sys.exit(1)
    
input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else None
reverse_colors(input_path, output_path) 