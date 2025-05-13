import html

import pandas as pd
import torch
import cv2
import os

from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
from tqdm import tqdm
from evaluate import load
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Modified version of Craft from harshanck
# https://github.com/harshanck/trocr-multiline
from application.model.craft_text_detector import Craft
from application.model.modelbase import ModelBase

from application.model.modelMatthew.findingWords import extract_wordlike_sectors
from application.model.modelMatthew.textSectors import adjust_for_outliers

DEBUG_MODE = False

def get_contour_precedence(box, cols):
    tolerance_factor = 10
    x, y = box[0][0], box[0][1]
    return ((y // tolerance_factor) * tolerance_factor) * cols + x

def kaggle_dataset_preprocess(self, dataset_dir):
    """
    Preprocess the Kaggle IAM dataset from https://www.kaggle.com/datasets/ngkinwang/iam-dataset for TrOCR.
    Input:
        dataset_dir: Path to the directory containing the IAM dataset from Kaggle.
    Output:
        out_df: Pandas DataFrame containing two columns, 'file_name' and 'text'. File names all end in .png.
    """

    out_df = pd.DataFrame({"file_name": [], "text": []})
    with open(f"{dataset_dir}/linux_gt.txt", "r") as f:
        lines = f.readlines()
        lines = [[line.split()[0], line.split()[1]] for line in lines]

    previous_file_piece = lines[0][0].split("/")[3]
    previous_file_components = previous_file_piece.split("-")
    previous_file_components.pop(-1)
    previous_file = "-".join(previous_file_components) + ".png"

    resulting_text = ""

    for line in lines:
        file_path = line[0]
        string = line[1]

        current_file_piece = file_path.split("/")[3]
        current_file_components = current_file_piece.split("-")
        current_file_components.pop(-1)
        current_file = "-".join(current_file_components) + ".png"

        if current_file == previous_file:
            if resulting_text != "":
                resulting_text += " "
            resulting_text += html.unescape(str(string))
        else:
            out_df.loc[len(out_df)] = [previous_file, resulting_text]
            previous_file = current_file
            resulting_text = html.unescape(str(string))

    return out_df

# Needed for PyTorch, from NielsRogge's tutorial
class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]

        # extract kaggle dataset dir structure from filename
        dir_components = file_name.split("-")
        lines_archive_dir = f"{dir_components[0]}/{dir_components[0]}-{dir_components[1]}/"

        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + lines_archive_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

class TrOCR(ModelBase):
    def __init__(self):
        super().__init__("TrOCR")
        self.text_detector = Craft(output_dir=None, crop_type="box", cuda=True)
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    def _preprocess(self, input_file_path, tolerance_coeff=0.10):
        """
        Function which detects text sectors and tries to cut them into single lines or words.
        Argument image is path to image which will be preprocessed.
        Does not return anything, but saves cut images to directories created by it.
        """
        try:
            input_dir = os.environ['UPLOADED_FILES']
            imageLoad = cv2.imread(input_file_path)
            gray = cv2.cvtColor(imageLoad, cv2.COLOR_BGR2GRAY)
            if DEBUG_MODE:
                cv2.imwrite("UploadedFiles/gray.png", gray)

            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            if DEBUG_MODE:
                cv2.imwrite("UploadedFiles/gray_blurred.png", blur)

            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            if DEBUG_MODE:
                cv2.imwrite("UploadedFiles/thresh.png", thresh)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 13))
            dilate = cv2.dilate(thresh, kernel, iterations=1)
            if DEBUG_MODE:
                cv2.imwrite(f"{input_dir}/dilate.png", dilate)

            # Everything above this line prepares for text sectors detection,
            # we do things like blurring the image, graying it out to reduce noice
            # and then dilate the rest to extract text sectors
            # then we write boxes on text sectors and we splinter original file according to them

            contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 1 and cv2.boundingRect(c)[3] > 1]
            img_height = imageLoad.shape[0]
            tolerance = int(tolerance_coeff * img_height)

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
                # Checks if boxes are close enough to merge them.
                # Threshold is in pixels
                x1, y1, w1, h1 = b1
                x2, y2, w2, h2 = b2
                return not (
                            x1 + w1 + thresh < x2 or x2 + w2 + thresh < x1 or y1 + h1 + thresh < y2 or y2 + h2 + thresh < y1)

            def merge_boxes(b1, b2):
                # Merges close boxes
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
                # Sorts boxes vertically and horizontally. We sort vertically according to a tolerance, as curved text skews results
                return (box[1] // tolerance), box[0]

            sorted_boxes = sorted(filtered_boxes, key=sort_key)

            img_w, img_h = imageLoad.shape[1], imageLoad.shape[0]

            def is_horizontal_line(box):
                # Checks if we detected a divider as text.
                # Reduces noise overall when document is partitioned
                x, y, w, h = box
                aspect_ratio = w / h if h > 0 else 0
                return h <= 15 and aspect_ratio > 10

            # We filter out very small boxes which is likely noise
            # w and h can be adjusted to smaller/bigger values
            final_boxes = [(x, y, w, h) for (x, y, w, h) in sorted_boxes if w >= 25 and h >= 25]

            # Split boxes containing horizontal lines
            line_boxes = [box for box in final_boxes if is_horizontal_line(box)]
            other_boxes = [box for box in final_boxes if not is_horizontal_line(box)]
            used_lines = []
            new_other_boxes = []

            for other_box in other_boxes:
                # Splintering file according to boxes
                ox, oy, ow, oh = other_box
                split_lines = []
                for line_box in line_boxes:
                    lx, ly, lw, lh = line_box
                    if lx >= ox and ly >= oy and (lx + lw) <= (ox + ow) and (ly + lh) <= (oy + oh):
                        if lw >= 0.8 * ow:
                            split_lines.append(line_box)
                            used_lines.append(line_box)

                split_lines.sort(key=lambda lb: lb[1])
                current_y = oy
                remaining_height = oh

                # There is lower tolerance since it is much less likely to have noise slightly out of box
                # Than have a random stray box, however upper_height and remaining_height can be adjusted if needed
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
            roi_idx_list = []

            for idx, (x, y, w, h) in enumerate(final_boxes, start=1):
                # Appending cut boxes to list in sorted order for next steps.
                roi = imageLoad[y: y + h, x: x + w]
                roi_idx_list.append([roi, idx])

                # Draw a rectangle on the image which will be saved if debugging
                if DEBUG_MODE:
                    color = (0, 0, 255) if is_horizontal_line((x, y, w, h)) else (36, 255, 12)
                    cv2.rectangle(imageLoad, (x, y), (x + w, y + h), color, 2)

            if DEBUG_MODE:
                # We can check what we have drawn there
                cv2.imwrite(f"{input_dir}/boxed.png", imageLoad)

            # Find stray dividers and lastly hone in on final sectors
            segment_counter_list = adjust_for_outliers(roi_idx_list)
            wordlike_sector_list = extract_wordlike_sectors(segment_counter_list)

            return wordlike_sector_list
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def _finetune(self, dataset_dir):
        """
        [WIP] Function which fine-tunes the TrOCR model using the Polish handwritten letters dataset located at
        https://www.kaggle.com/datasets/westedcrean/phcd-polish-handwritten-characters-database
        """
        pass

    def _evaluate(self, dataset_dir, output_dir):
        """
        Function reflects NielsRogge's TrOCR tutorial located at
        https://github.com/NielsRogge/Transformers-Tutorials/tree/master/TrOCR
        """
        df = self._preprocess("datasets/iam")

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        test_dataset = IAMDataset(root_dir="datasets/iam/",
                                  df=df,
                                  processor=processor)

        test_dataloader = DataLoader(test_dataset, batch_size=8)
        batch = next(iter(test_dataloader))
        for k, v in batch.items():
            print(k, v.shape)

        labels = batch["labels"]
        labels[labels == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels, skip_special_tokens=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.to(device)

        cer = load("cer")

        print("Running evaluation...")

        for batch in tqdm(test_dataloader):
            # predict using generate
            pixel_values = batch["pixel_values"].to(device)
            outputs = model.generate(pixel_values)

            # decode
            pred_str = processor.batch_decode(outputs, skip_special_tokens=True)
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(labels, skip_special_tokens=True)

            # add batch to metric
            cer.add_batch(predictions=pred_str, references=label_str)

        final_score = cer.compute()
        print(final_score)

    def perform_ocr(self, input_path, **kwargs):
        """
        Function which runs inference on an image and outputs a text file.
        """
        # Test if everything works using _evaluate().
        #self._evaluate(dataset_dir, output_dir)

        try:
            # image = Image.open(input_path).convert("RGB")
            # result = self.text_detector.detect_text(input_path)
            # boxes = result["boxes"]
            # boxes = sorted(boxes, key=lambda x: get_contour_precedence(x, image.width))
            cv2_sector_list = self._preprocess(input_path)
            texts = []
            for cv2_sector in cv2_sector_list:
                #crop = image.crop((box[0][0], box[0][1], box[2][0], box[2][1]))
                # cv2 image array word_img is contained in sector[0], converting to PIL
                cv2_img_array = cv2_sector[0]
                rgb_img_array = cv2.cvtColor(cv2_img_array, cv2.COLOR_BGR2RGB)
                converted_image = Image.fromarray(rgb_img_array)
                pixel_values = self.processor(converted_image, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                texts.append(text)

            '''
            chars_in_current_line = 0
            max_line_width = 80
            with open(output_path, "w") as f:
                for seq in texts:
                    chars_in_current_line += len(seq)
                    if chars_in_current_line > max_line_width:
                        f.writelines(f"{seq}\n")
                        chars_in_current_line = 0
                    else:
                        f.writelines(f"{seq} ")
            return output_path
            '''

            return texts

        except Exception as e:
            print(f"Unexpected error in perform_ocr: {e}")
            return False