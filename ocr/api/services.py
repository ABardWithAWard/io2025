import os
import base64
from django.core.files.storage import FileSystemStorage
import firebase_admin
from firebase_admin import credentials, firestore

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
        paddle_result_list = paddle_model.perform_ocr(input_path=full_path)
        print("PaddleOCR results:")
        print(" ".join([result for result in paddle_result_list[0]["rec_texts"]]))

        easy_result_list = easy_model.perform_ocr(input_path=full_path)
        print("EasyOCR results:")
        print(" ".join([result[1] for result in easy_result_list]))

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