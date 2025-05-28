import os
import base64
from django.core.files.storage import FileSystemStorage
import firebase_admin
from firebase_admin import credentials, firestore

from application.model.modelMatthew.model import Model
from application.model.trocr import TrOCR

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
    from application.model.modelMatthew.model import Model
    from application.model.trocr import TrOCR
    print("Attempted ocr")
    model = TrOCR()
    modelMatthew = Model()
    
    full_path = prepare_file_hierarchy(file)

    # using protected function like this because model is still above 0.02 loss and doesnt
    # predict well
    modelMatthew._preprocess(full_path)
    # function used in different model than trocr, for more details go to implementation

    # Process the single uploaded file
    # Now, we catch errors in trocr.py file since we did it anyway, no need for doing this twice
    print(model.perform_ocr(full_path))

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