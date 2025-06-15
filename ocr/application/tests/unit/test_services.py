import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
from unittest import mock, TestCase
from django.core.files.uploadedfile import SimpleUploadedFile

from application.services import (
    prepare_file_hierarchy,
    output_processed_as_txt,
    output_processed_as_docx,
    get_db,
    retrieve_pictures_using_uid,
    set_file_limit,
    set_data_limit,
    get_limits
)


class ServicesTests(TestCase):

    @mock.patch.dict(os.environ, {"UPLOADED_FILES": tempfile.gettempdir()})
    def test_prepare_file_hierarchy_saves_file(self):
        content = b"test image data"
        test_file = SimpleUploadedFile("test.png", content, content_type="image/png")
        saved_path = prepare_file_hierarchy(test_file)
        self.assertTrue(os.path.exists(saved_path))
        with open(saved_path, "rb") as f:
            self.assertEqual(f.read(), content)
        os.remove(saved_path)

    def test_output_processed_as_txt(self):
        words = ["This", "is", "a", "test", "file", "for", "OCR", "output."]
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            output_processed_as_txt(words, tmp.name, line_width=20)
            tmp.seek(0)
            content = tmp.read().decode()
        self.assertIn("This is a test", content)
        os.remove(tmp.name)

    def test_output_processed_as_docx(self):
        words = ["This", "is", "a", "test", "file", "for", "OCR", "output."]
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            output_processed_as_docx(words, tmp.name, font_size=12)
        self.assertTrue(os.path.exists(tmp.name))
        os.remove(tmp.name)

    @mock.patch("application.services.firebase_admin")
    @mock.patch("application.services.credentials")
    @mock.patch("application.services.firestore")
    @mock.patch("os.path.exists", return_value=True)
    @mock.patch.dict(os.environ, {"FIREBASE_KEY": "/fake/path/firebase.json"})
    def test_get_db_initializes_when_not_initialized(
        self, mock_exists, mock_firestore, mock_credentials, mock_firebase_admin
    ):
        mock_firebase_admin._apps = {}

        mock_cred = mock.MagicMock()
        mock_credentials.Certificate.return_value = mock_cred

        mock_client = mock.MagicMock()
        mock_firestore.client.return_value = mock_client

        db = get_db()

        mock_credentials.Certificate.assert_called_once_with("/fake/path/firebase.json")
        mock_firebase_admin.initialize_app.assert_called_once_with(mock_cred)
        mock_firestore.client.assert_called_once()
        self.assertEqual(db, mock_client)

    @mock.patch("application.services.firebase_admin")
    @mock.patch("application.services.firestore")
    def test_get_db_returns_client_if_already_initialized(
        self, mock_firestore, mock_firebase_admin
    ):
        mock_firebase_admin._apps = {"default": "exists"}
        mock_client = mock.MagicMock()
        mock_firestore.client.return_value = mock_client

        db = get_db()

        mock_firebase_admin.initialize_app.assert_not_called()
        mock_firestore.client.assert_called_once()
        self.assertEqual(db, mock_client)

    @mock.patch("application.services.firebase_admin")
    @mock.patch("application.services.credentials")
    @mock.patch("os.path.exists", return_value=False)
    @mock.patch.dict(os.environ, {"FIREBASE_KEY": "/missing/path.json"})
    def test_get_db_raises_exception_when_cred_missing(
        self, mock_exists, mock_credentials, mock_firebase_admin
    ):
        mock_firebase_admin._apps = {}
        with self.assertRaises(Exception) as context:
            get_db()
        self.assertIn("Firebase credentials not found", str(context.exception))

    @mock.patch("application.services.firestore_db")
    def test_retrieve_pictures_using_uid(self, mock_firestore_db):
        # Przygotowanie zakodowanego obrazu jako base64
        image = Image.new("RGB", (100, 100), color="blue")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Mock Firestore snapshot
        mock_snapshot = mock.MagicMock()
        mock_snapshot.get.return_value = encoded_image

        # Mock Firestore query
        mock_firestore_db.collection.return_value.where.return_value.stream.return_value = [
            mock_snapshot
        ]

        result = retrieve_pictures_using_uid("test_uid")
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Image.Image)
        self.assertEqual(result[0].size, (100, 100))

    @mock.patch("application.services.get_db")
    def test_set_data_limit(self, mock_get_db):
        mock_db = mock.MagicMock()
        mock_get_db.return_value = mock_db

        set_data_limit(123)

        mock_db.collection.assert_called_once_with("global_settings")
        mock_db.collection().document.assert_called_once_with("limits")
        mock_db.collection().document().set.assert_called_once_with(
            {"dataLimit": 123}, merge=True
        )

    @mock.patch("application.services.get_db")
    def test_set_file_limit(self, mock_get_db):
        mock_db = mock.MagicMock()
        mock_get_db.return_value = mock_db

        set_file_limit(10)

        mock_db.collection.assert_called_once_with("global_settings")
        mock_db.collection().document.assert_called_once_with("limits")
        mock_db.collection().document().set.assert_called_once_with(
            {"fileLimit": 10}, merge=True
        )

    @mock.patch("application.services.get_db")
    def test_get_limits_returns_data(self, mock_get_db):
        mock_doc = mock.MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {"dataLimit": 123, "fileLimit": 10}

        mock_db = mock.MagicMock()
        mock_db.collection().document().get.return_value = mock_doc
        mock_get_db.return_value = mock_db

        result = get_limits()
        self.assertEqual(result, {"dataLimit": 123, "fileLimit": 10})

    @mock.patch("application.services.get_db")
    def test_get_limits_returns_empty_if_not_exists(self, mock_get_db):
        mock_doc = mock.MagicMock()
        mock_doc.exists = False

        mock_db = mock.MagicMock()
        mock_db.collection().document().get.return_value = mock_doc
        mock_get_db.return_value = mock_db

        result = get_limits()
        self.assertEqual(result, {})