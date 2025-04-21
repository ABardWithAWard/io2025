import glob
import re
import cv2
import numpy as np
import os


# Might be highly unnecessary and I could just implement it in a different way
# Code largely stolen from stack overflow. Does not work very well for small text and words which are
# rising their height on a screen. Fix: just write like a human being
def xAxisKernel(size):
    size = size if size % 2 else size + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    kernel[center - 1:center + 2, :] = 1  # Vertical kernel
    return kernel


def find_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, xAxisKernel(21))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = sorted([cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100],
                   key=lambda x: x[1])  # Sort by Y coordinate
    return lines


def find_words(line_image):
    padded = cv2.copyMakeBorder(line_image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, xAxisKernel(9))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = sorted([(x - 15, y - 15, w, h) for (x, y, w, h) in
                    [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 50]],
                   key=lambda x: x[0])  # Sort by X coordinate
    return words


def preprocessWords():
    # Create output directory
    os.makedirs('UploadedFiles/words', exist_ok=True)

    # Get and sort all input files
    files = sorted(glob.glob('UploadedFiles/roiEdited*.png'),
                   key=lambda x: int(re.search(r'roiEdited(\d+)\.png$', x).group(1)))

    if not files:
        print("No files found matching pattern: UploadedFiles/roiEdited*.png")
        return

    word_counter = 1

    for file_path in files:
        print(f"Processing {file_path}...")
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not read {file_path}")
            continue

        try:
            # Get image dimensions
            h, w = img.shape[:2]

            # Find and process lines
            lines = find_lines(img)
            for lx, ly, lw, lh in lines:
                line_roi = img[ly:ly + lh, lx:lx + lw]
                words = find_words(line_roi)

                # Process words in line
                for wx, wy, ww, wh in words:
                    abs_x = lx + wx
                    abs_y = ly + wy

                    # Validate coordinates
                    x1 = max(0, abs_x)
                    y1 = max(0, abs_y)
                    x2 = min(w, x1 + ww)
                    y2 = min(h, y1 + wh)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Save word image
                    word_img = img[y1:y2, x1:x2]
                    output_path = f'UploadedFiles/words/text{word_counter:04d}.png'
                    cv2.imwrite(output_path, word_img)
                    word_counter += 1

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print(f"Processed {word_counter - 1} words total")

