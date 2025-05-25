import os
from docx import Document
from docx.shared import Pt
from typing import List

from pathlib import Path
from django.core.files.storage import FileSystemStorage

#from application.model.modelMatthew.model import Model
#from application.model.trocr import TrOCR
from application.model.paddleocr import PaddleOCR
from application.model.easyocr import EasyOCR

paddle_model = PaddleOCR()
easy_model = EasyOCR()

def prepare_file_hierarchy (file):
    """Takes uploaded file and returns directory where it is saved and its detected content"""
    upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])
    print(f"Upload directory: {upload_dir}")

    # Save the uploaded file first
    storage = FileSystemStorage(location=upload_dir)
    file_path = storage.save(file.name, file)
    full_path = storage.path(file_path)
    print(f"Saved file to: {full_path}")

    # Possibly legacy?
    # output_dir = os.path.join(upload_dir, 'processed_text')
    # os.makedirs(output_dir, exist_ok=True)

    return full_path

def handle_uploaded_file(file):
    """Takes file uploaded in form and calls helper function to manage file and its contents"""
    full_path = prepare_file_hierarchy(file)

    # using protected function like this because model is still above 0.02 loss and doesnt
    # predict well
    #modelMatthew._preprocess(full_path)
    # function used in different model than trocr, for more details go to implementation

    # Process the single uploaded file
    # Now, we catch errors in trocr.py file since we did it anyway, no need for doing this twice
    paddle_result_list = paddle_model.perform_ocr(input_path=full_path)
    print("PaddleOCR results:")
    print(" ".join([result for result in paddle_result_list[0]["rec_texts"]]))

    easy_result_list = easy_model.perform_ocr(input_path=full_path)
    print("EasyOCR results:")
    print(" ".join([result[1] for result in easy_result_list]))

    #TODO: Somehow save to cloud user input and model input (or just user input? Depends on pricing)

def output_processed_as_txt(word_list: List[str], output_path: str, line_width=80):
    """
    Output a list of words (strings) extracted from an image into a .txt file at output_path.
    :param word_list: List of strings which are extracted words.
    :param output_path: A string representing the output_path for the .txt file to be generated at.
    :param line_width: The limit in characters after which a full new word cannot be written (but a part of the word may).
    """
    chars_in_current_line = 0
    max_line_width = line_width
    with open(output_path, "w") as f:
        for word in word_list:
            chars_in_current_line += len(word)
            # If limit exceeded after writing this word, write
            if chars_in_current_line > max_line_width:
                # But write with a newline and reset next line to the beginning
                f.writelines(f"{word}\n")
                chars_in_current_line = 0
            else:
                # Otherwise simply write
                f.writelines(f"{word} ")

def output_processed_as_docx(word_list: List[str], output_path: str, font_size=11):
    """
    Output a list of words (strings) extracted from an image into a .docx file at output_path.
    :param word_list: List of strings which are extracted words.
    :param output_path: A string representing the output_path for the .txt file to be generated at.
    :param font_size: The size of the font of the text which will take up the entire width at all times.
    """
    document = Document()

    # Add_run is the most general "generate text" method. It gives access to a Font obj through .font
    font = document.add_paragraph().add_run(" ".join(word_list)).font

    # Through the Font obj we can modify the appearance of the text
    font.name = "Arial"
    font.size = Pt(font_size)

    # The appearance will be saved
    document.save(output_path)

# Unit test
if __name__ == "__main__":
    word_list = ["God", "save", "our", "gracious", "Queen,", "Long", "live", "our", "noble", "Queen,",
                 "God", "save", "the", "queen!", "Send", "her", "victorious,", "Happy", "and", "Glorious",
                 "Long", "to", "reign", "over", "us;", "God", "save", "the", "Queen!"]

    abs_output_dir = "/home/km/PycharmProjects/io2025/test_output"

    output_processed_as_txt(word_list, abs_output_dir + "/output.txt")
    output_processed_as_docx(word_list, abs_output_dir + "/output.docx")