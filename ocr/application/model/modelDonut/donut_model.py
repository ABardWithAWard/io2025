import os
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import default_data_collator

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments

from application.model.modelbase import ModelBase

model_path = "./application/model/donut_model"


class DonutDataset(Dataset):
    def __init__(self, df, image_dir, processor, prompt="<s_docvqa><s_question>What does the document say?</s_question><s_answer>"):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.prompt = prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir,
                                  row["file_name"])  # Zakłada, że kolumna 'file_name' zawiera nazwę pliku
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        target_text = f"{self.prompt}{row['text']}</s>"
        labels = self.processor.tokenizer(target_text, padding="max_length", max_length=512,
                                          return_tensors="pt").input_ids.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs.pop("num_items_in_batch", None)
        return super().compute_loss(model, inputs, return_outputs)


class DonutModel(ModelBase):
    def __init__(self):
        super().__init__("Donut")

    def _preprocess(self, dataset_dir):
        labels_path = os.path.join(dataset_dir, "labels.csv")
        df = pd.read_csv(labels_path)
        return df

    def _finetune(self, dataset_dir):
        df = self._preprocess(dataset_dir)
        image_dir = os.path.join(dataset_dir, "pages")

        if os.path.exists(model_path):
            processor = DonutProcessor.from_pretrained(model_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_path)
        else:
            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", use_fast=True)
            model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

        dataset = DonutDataset(df, image_dir, processor)

        train_args = Seq2SeqTrainingArguments(
            output_dir="../UploadedFiles/processed_text",
            eval_strategy="no",
            per_device_train_batch_size=2,
            num_train_epochs=3,
            logging_dir="./logs",
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            tokenizer=processor.tokenizer,
            data_collator=default_data_collator,
        )
        print("Początek treningu")
        trainer.train()
        print("Koniec treningu")

        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)

    def perform_ocr(self, input_path, output_path):
        try:
            if not os.path.exists(model_path):
                print("Model folder not found, loading from Hugging Face model...")
                processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa", use_fast=True)
                model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
            else:
                processor = DonutProcessor.from_pretrained(model_path)
                model = VisionEncoderDecoderModel.from_pretrained(model_path)

            model.eval()

            image = Image.open(input_path).convert("RGB")
            task_prompt = "<s_synthdog>"
            pixel_values = processor(image, return_tensors="pt").pixel_values
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            pixel_values = pixel_values.to(device)
            decoder_input_ids = decoder_input_ids.to(device)

            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    num_beams=1,
                )
            result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            return True
        except Exception as e:
            print(f"Błąd: {e} HALLO?")
            return False

