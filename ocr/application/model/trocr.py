import html
import pandas as pd
import torch

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


def get_contour_precedence(box, cols):
    tolerance_factor = 10
    x, y = box[0][0], box[0][1]
    return ((y // tolerance_factor) * tolerance_factor) * cols + x

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

    def _preprocess(self, dataset_dir):
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

    def perform_ocr(self, input_path, output_path):
        """
        Function which runs inference on an image and outputs a text file.
        """
        # Test if everything works using _evaluate().
        #self._evaluate(dataset_dir, output_dir)

        try:
            image = Image.open(input_path).convert("RGB")
            result = self.text_detector.detect_text(input_path)
            boxes = result["boxes"]
            boxes = sorted(boxes, key=lambda x: get_contour_precedence(x, image.width))
            texts = []
            for box in boxes:
                crop = image.crop((box[0][0], box[0][1], box[2][0], box[2][1]))
                pixel_values = self.processor(crop, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                texts.append(text)

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

            return True
        except Exception:
            return False