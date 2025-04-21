import cv2
import numpy as np
import glob
import os
import re

from application.model.modelMatthew.findingWords import preprocessWords


# Sometimes works. Sometimes does not. Needs to be reworked as it works on my CV, but it does not in case of an old book
# https://github.com/wjbmattingly/ocr_python_textbook/blob/main/data/index_02.JPG
# This one ^
# Maybe we should write unit tests to those functions?to
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

def process_images(input_pattern='UploadedFiles/roi*.png'):
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



def preprocess(image):
    imageLoad = cv2.imread(image)
    gray = cv2.cvtColor(imageLoad, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("UploadedFiles/gray.png", gray)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    #cv2.imwrite("UploadedFiles/gray_blurred.png", blur)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #cv2.imwrite("UploadedFiles/thresh.png", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imwrite("UploadedFiles/dilate.png", dilate)

    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 1 and cv2.boundingRect(c)[3] > 1]
    img_height = imageLoad.shape[0]
    tolerance = int(0.10 * img_height)

    filtered_boxes = []
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        inside = False
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i != j:
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    inside = True
                    break
        if not inside:
            filtered_boxes.append((x1, y1, w1, h1))

    def boxes_are_close(b1, b2, thresh=15):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 + thresh < x2 or x2 + w2 + thresh < x1 or y1 + h1 + thresh < y2 or y2 + h2 + thresh < y1)

    def merge_boxes(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        x = min(x1, x2)
        y = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)
        return (x, y, x_max - x, y_max - y)

    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip = set()
        for i in range(len(filtered_boxes)):
            if i in skip:
                continue
            box1 = filtered_boxes[i]
            for j in range(i + 1, len(filtered_boxes)):
                if j in skip:
                    continue
                box2 = filtered_boxes[j]
                if boxes_are_close(box1, box2):
                    box1 = merge_boxes(box1, box2)
                    skip.add(j)
                    merged = True
            new_boxes.append(box1)
        filtered_boxes = new_boxes

    def sort_key(box):
        return (box[1] // tolerance, box[0])

    sorted_boxes = sorted(filtered_boxes, key=sort_key)

    img_w, img_h = imageLoad.shape[1], imageLoad.shape[0]

    def is_horizontal_line(box):
        x, y, w, h = box
        aspect_ratio = w / h if h > 0 else 0
        return h <= 15 and aspect_ratio > 10

    final_boxes = [(x, y, w, h) for (x, y, w, h) in sorted_boxes if w >= 15 and h >= 15]

    # Split boxes containing horizontal lines
    line_boxes = [box for box in final_boxes if is_horizontal_line(box)]
    other_boxes = [box for box in final_boxes if not is_horizontal_line(box)]
    used_lines = []
    new_other_boxes = []

    for other_box in other_boxes:
        ox, oy, ow, oh = other_box
        split_lines = []
        for line_box in line_boxes:
            lx, ly, lw, lh = line_box
            if (lx >= ox and ly >= oy and (lx + lw) <= (ox + ow) and (ly + lh) <= (oy + oh)):
                if lw >= 0.8 * ow:
                    split_lines.append(line_box)
                    used_lines.append(line_box)
        split_lines.sort(key=lambda lb: lb[1])
        current_y = oy
        remaining_height = oh
        for line in split_lines:
            ly = line[1]
            lh_line = line[3]
            upper_height = ly - current_y
            if upper_height >= 15:
                new_other_boxes.append((ox, current_y, ow, upper_height))
            current_y = ly + lh_line
            remaining_height = oh - (current_y - oy)
        if remaining_height >= 15:
            new_other_boxes.append((ox, current_y, ow, remaining_height))

    remaining_line_boxes = [lb for lb in line_boxes if lb not in used_lines]
    final_boxes = new_other_boxes + remaining_line_boxes
    final_boxes = sorted(final_boxes, key=sort_key)  # Re-sort after splitting

    for idx, (x, y, w, h) in enumerate(final_boxes, start=1):
        roi = imageLoad[y:y + h, x:x + w]
        color = (0, 0, 255) if is_horizontal_line((x, y, w, h)) else (36, 255, 12)
        cv2.imwrite(f"UploadedFiles/roi{idx}.png", roi)
        cv2.rectangle(imageLoad, (x, y), (x + w, y + h), color, 2)

    cv2.imwrite("UploadedFiles/boxed.png", imageLoad)
    process_images()
    preprocessWords()