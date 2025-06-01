import os
import base64
from docx import Document
from docx.shared import Pt
from typing import List

from django.core.files.storage import FileSystemStorage

import firebase_admin
from firebase_admin import credentials, firestore
from application.services import paddle_model, easy_model

from application.model.paddleocr import PaddleOCR
from application.model.easyocr import EasyOCR
from application.utils import validate_image_brightness

paddle_model = PaddleOCR()
easy_model = EasyOCR()

def get_files():
    """Get list of files from the upload directory"""
    directory = os.environ.get('UPLOADED_FILES')
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return ['empty']

def prepare_file_hierarchy(file):
    """Takes uploaded file and returns directory where it is saved and its detected content"""
    upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])
    print(f"Upload directory: {upload_dir}")

    # Save the uploaded file first
    storage = FileSystemStorage(location=upload_dir)
    file_path = storage.save(file.name, file)
    full_path = storage.path(file_path)
    print(f"Saved file to: {full_path}")

    return full_path

def handle_uploaded_file(file, user_uid=None):
    """Takes file uploaded in form and calls helper function to manage file and its contents"""
    full_path = prepare_file_hierarchy(file)

    if validate_image_brightness(full_path):
        paddle_result = paddle_model.perform_ocr(input_path=full_path)
        print("PaddleOCR results:")
        print(" ".join([result for result in paddle_result["text_predictions"]]))

        easy_result = easy_model.perform_ocr(input_path=full_path)
        print("EasyOCR results:")
        print(" ".join([result for result in easy_result["text_predictions"]]))

        # Initialize Firebase if not already initialized
        if not firebase_admin._apps:
            cred_path = os.environ['FIREBASE_KEY']
            if not os.path.exists(cred_path):
                raise Exception('Firebase credentials not found')
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)

        # Get Firestore client
        db = firestore.client()

        # Read and encode the image
        with open(full_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        # Add image to Firestore with userUid
        image_ref = db.collection('images').document()
        image_ref.set({
            'image_data': encoded_string,
            'filename': file.name,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'userUid': user_uid  # This will be null for unauthenticated users
        })

        # We can look up images using https://base64.guru/converter/decode/image
        # As they are saved as base64 strings

        return full_path

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