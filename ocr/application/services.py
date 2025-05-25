import os
from pathlib import Path
import base64
from io import BytesIO

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter

from PIL import Image
from typing import List
from django.core.files.storage import FileSystemStorage

from application.model.paddleocr import PaddleOCR
from application.model.easyocr import EasyOCR
from application.utils import validate_image_brightness

paddle_model = PaddleOCR()
easy_model = EasyOCR()

firestore_db = None

def prepare_file_hierarchy(file):
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

    if validate_image_brightness(full_path):
        paddle_result = paddle_model.perform_ocr(input_path=full_path)
        print("PaddleOCR results:")
        print(" ".join([result for result in paddle_result["text_predictions"]]))

        easy_result = easy_model.perform_ocr(input_path=full_path)
        print("EasyOCR results:")
        print(" ".join([result for result in easy_result["text_predictions"]]))

def setup_firestore_db():
    # There may be a more elegant solution, but for now I am using a global variable
    global firestore_db

    # The environment variable is pulled from rootdir/ocr/.env
    credentials_obj = credentials.Certificate(os.environ["FIREBASE_KEY"])
    firebase_admin.initialize_app(credentials_obj)
    firestore_db = firestore.client()

def retrieve_pictures_using_uid(desired_uid: str) -> List[Image]:
    """
    Retrieve a list of a user's PIL.Image objects from the Firestore database based on his UID.
    Depends on the global (services.py) variable firebase_db being initialized. This variable is
    initialized with setup_firestore_db (services.py) called within ApplicationConfig (apps.py).

    :param desired_uid: The UID of the user for which pictures are to be retrieved from the db.
    :return: A list of Pillow image objects generated from parsing image data stored in the db.
    """

    # A list of DocumentSnapshot objects meeting the userUid == desired_uid criterion
    user_images_snapshot_list = (
        firestore_db.collection("images")
        .where(filter=FieldFilter("userUid", "==", desired_uid))
        .stream()
    )

    # A list of user images encoded in b64 string format
    user_images_b64_list = [user_image_snapshot.get("image_data") for user_image_snapshot in user_images_snapshot_list]

    # A list of PIL.Image objects generated from decoding the b64 string format
    user_images_list = [Image.open(BytesIO(base64.b64decode(user_image_b64))) for user_image_b64 in user_images_b64_list]

    return user_images_list
