import cv2
import numpy as np
import glob
import os
import re

DEBUG_MODE = False

# Sometimes works. Sometimes does not. Needs to be reworked as it works on my CV, but it does not in case of an old book
# https://github.com/wjbmattingly/ocr_python_textbook/blob/main/data/index_02.JPG
# This one ^
# Maybe we should write unit tests to those functions?
def split_image_on_lines(image, lines, width, height):
    #Splits image along lines drawn
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

    if DEBUG_MODE:
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

def adjust_for_outliers(roi_idx_list):
    """
    Function tries to find stray dividers if text is positioned too close to it
    Input:
        roi_idx_list: ???
    Returns:
        roi_counter_list: ???
    """
    regions_of_interest = sorted(roi_idx_list, key=lambda x: x[1])
    counter = 1
    roi_counter_list = []

    for roi, idx in regions_of_interest:
        if DEBUG_MODE:
            print(f"Processing region of interest with id {idx}...")

        # We have a smaller roi now, so we need lower blur to be able to detect lines and silence noise.
        # It works similarly to the model.py _preprocessing function
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, threshold1=30, threshold2=120, apertureSize=3)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        height, width = roi.shape[:2]

        # Detects long lines which should be dividers according to these parameters.
        lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, 50, 10, 200)

        if lines is not None:
            segments = split_image_on_lines(roi, lines, width, height)
        else:
            if DEBUG_MODE:
                print("  No lines found. Keeping original.")
            segments = [roi]

        #base_dir = os.path.dirname(file_path)

        # If we found a line we rename files in such a way that we keep sorted order
        if segments:
            for seg in segments:
                if not (seg.shape[0] < 10 or seg.shape[1] < 10):
                    roi_counter_list.append([seg, counter])

    if len(roi_counter_list) == 0:
        print("No output images were generated.")

    return roi_counter_list