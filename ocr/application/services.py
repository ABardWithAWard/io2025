import os
from docx import Document
from docx.shared import Pt

from django.core.files.storage import FileSystemStorage

# from application.model.modelMatthew.model import Model
# from application.model.trocr import TrOCR
#
# model = TrOCR()
# modelMatthew = Model()

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
    modelMatthew._preprocess(full_path)
    # function used in different model than trocr, for more details go to implementation

    # Process the single uploaded file
    # Now, we catch errors in trocr.py file since we did it anyway, no need for doing this twice
    print(model.perform_ocr(full_path))

    #TODO: Somehow save to cloud user input and model input (or just user input? Depends on pricing)

def output_processed_as_txt(word_list, output_path, line_width=80):
    chars_in_current_line = 0
    max_line_width = line_width
    with open(output_path, "w") as f:
        for word in word_list:
            chars_in_current_line += len(word)
            if chars_in_current_line > max_line_width:
                f.writelines(f"{word}\n")
                chars_in_current_line = 0
            else:
                f.writelines(f"{word} ")

def output_processed_as_docx(word_list, output_path, font_size=11):
    document = Document()
    font = document.add_paragraph().add_run(" ".join(word_list)).font
    font.name = "Arial"
    font.size = Pt(font_size)
    document.save(output_path)

if __name__ == "__main__":
    word_list = ["God", "save", "our", "gracious", "Queen,", "Long", "live", "our", "noble", "Queen,",
                 "God", "save", "the", "queen!", "Send", "her", "victorious,", "Happy", "and", "Glorious",
                 "Long", "to", "reign", "over", "us;", "God", "save", "the", "Queen!"]

    abs_output_dir = "/home/km/PycharmProjects/io2025/test_output"

    output_processed_as_txt(word_list, abs_output_dir + "/output.txt")
    output_processed_as_docx(word_list, abs_output_dir + "/output.docx")