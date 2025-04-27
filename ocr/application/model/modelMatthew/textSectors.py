import cv2
import numpy as np
import glob
import os
import re

from application.model.modelMatthew.findingWords import preprocessWords


# Sometimes works. Sometimes does not. Needs to be reworked as it works on my CV, but it does not in case of an old book
# https://github.com/wjbmattingly/ocr_python_textbook/blob/main/data/index_02.JPG
# This one ^
# Maybe we should write unit tests to those functions?
def split_image_on_lines(image, lines, width, height):
    horizontal_cuts = []
    vertical_cuts = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y2 - y1) < 5 and abs(x2 - x1) > 0.9 * width:
                y = (y1 + y2) // 2
                if 10 < y < height - 10:
                    horizontal_cuts.append(y)
            elif abs(x2 - x1) < 5 and abs(y2 - y1) > 0.9 * height:
                x = (x1 + x2) // 2
                if 10 < x < width - 10:
                    vertical_cuts.append(x)

    horizontal_cuts = sorted(set(horizontal_cuts))
    vertical_cuts = sorted(set(vertical_cuts))

    print(f"  Horizontal cuts: {horizontal_cuts}")
    print(f"  Vertical cuts: {vertical_cuts}")

    segments = [image]

    if horizontal_cuts:
        segments = []
        prev = 0
        for cut in horizontal_cuts:
            segments.append(image[prev:cut, :])
            prev = cut
        segments.append(image[prev:, :])

    if vertical_cuts:
        new_segments = []
        for seg in segments:
            w = seg.shape[1]
            prev = 0
            for cut in vertical_cuts:
                new_segments.append(seg[:, prev:cut])
                prev = cut
            new_segments.append(seg[:, prev:])
        segments = new_segments

    return segments

def process_images(input_pattern=f'{os.environ['UPLOADED_FILES']}/roi*.png'):
    files = sorted(glob.glob(input_pattern), key=lambda x: int(re.search(r'roi(\d+)', x).group(1)))
    counter = 1
    generated_files = []

    for file_path in files:
        print(f"Processing {file_path}...")
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, threshold1=30, threshold2=120, apertureSize=3)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        height, width = image.shape[:2]
        lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, 50, 10, 200)

        if lines is not None:
            segments = split_image_on_lines(image, lines, width, height)
        else:
            print("  No lines found. Keeping original.")
            segments = [image]

        base_dir = os.path.dirname(file_path)

        if segments:
            for seg in segments:
                if seg.shape[0] < 10 or seg.shape[1] < 10:
                    continue
                new_filename = f"roiEdited{counter}.png"
                output_path = os.path.join(base_dir, new_filename)
                if cv2.imwrite(output_path, seg):
                    print(f"  Saved: {output_path}")
                    generated_files.append(output_path)
                    counter += 1
                else:
                    print(f"  Failed to save: {output_path}")

    if not generated_files:
        print("No output images were generated.")