import os
import base64
import tempfile
import json
from unittest import TestCase, mock
from io import BytesIO
from PIL import Image
from django.core.files.uploadedfile import SimpleUploadedFile

from api import services

# I dont know why there are errors
# It doesnt make any sense
# ngl


class TestServices(TestCase):

    def setUp(self):
        services.DEBUG_MODE = False
        services.POLISH_MODE = False

    def tearDown(self):
        services.DEBUG_MODE = False
        services.POLISH_MODE = False

    @mock.patch("os.listdir")
    def test_get_files_success(self, mock_listdir):
        mock_listdir.return_value = ["file1.png", "file2.jpg"]
        os.environ["UPLOADED_FILES"] = "/fake/path"
        self.assertEqual(services.get_files(), ["file1.png", "file2.jpg"])

    @mock.patch("os.listdir", side_effect=FileNotFoundError())
    def test_get_files_failure(self, mock_listdir):
        self.assertEqual(services.get_files(), ["empty"])

    @mock.patch("django.core.files.storage.FileSystemStorage.save")
    @mock.patch("django.core.files.storage.FileSystemStorage.path")
    def test_prepare_file_hierarchy(self, mock_path, mock_save):
        os.environ["UPLOADED_FILES"] = tempfile.gettempdir()
        file_mock = SimpleUploadedFile("test.png", b"file_content")
        mock_save.return_value = "test.png"
        mock_path.return_value = "/tmp/test.png"
        result = services.prepare_file_hierarchy(file_mock)
        self.assertEqual(result, "/tmp/test.png")

    def test_convert_result_to_json(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (10, 10), color="white")
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            uploaded_file = SimpleUploadedFile("test.png", f.read())

        result = {"confidence_scores": [0.99], "text_predictions": ["Hello"]}
        json_output = services.convert_result_to_json(uploaded_file, result)
        data = json.loads(json_output)
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["confidence"], [0.99])
        self.assertEqual(data["content"], ["Hello"])
        os.remove(tmp_path)

    @mock.patch("application.utils.validate_image_brightness", return_value=True)
    @mock.patch(
        "api.services.paddle_model.perform_ocr",
        return_value={"text_predictions": ["Hello"], "confidence_scores": [0.99]},
    )
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"binarydata")
    @mock.patch("firebase_admin._apps", new={})
    @mock.patch("firebase_admin.credentials.Certificate")
    @mock.patch("firebase_admin.initialize_app")
    @mock.patch("firebase_admin.firestore.client")
    @mock.patch("api.services.prepare_file_hierarchy", return_value="/tmp/test.png")
    def test_handle_uploaded_file_paddle(
        self,
        mock_prepare,
        mock_db,
        mock_init,
        mock_cert,
        mock_apps,
        mock_open,
        mock_perform_ocr,
        mock_brightness,
    ):
        file = SimpleUploadedFile("test.png", b"data")
        mock_doc = mock.MagicMock()
        mock_db.return_value.collection.return_value.document.return_value = mock_doc
        result_json = services.handle_uploaded_file(file, user_uid="123")
        self.assertIn("Hello", result_json)
        mock_doc.set.assert_called_once()

    @mock.patch("application.utils.validate_image_brightness", return_value=True)
    @mock.patch(
        "api.services.easy_model.perform_ocr",
        return_value={"text_predictions": ["Cześć"], "confidence_scores": [0.98]},
    )
    @mock.patch("api.services.prepare_file_hierarchy", return_value="/tmp/test.png")
    @mock.patch("firebase_admin._apps", new={})
    @mock.patch("firebase_admin.credentials.Certificate")
    @mock.patch("firebase_admin.initialize_app")
    @mock.patch("firebase_admin.firestore.client")
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"binarydata")
    def test_handle_uploaded_file_polish_mode(
        self,
        mock_open,
        mock_db,
        mock_init,
        mock_cert,
        mock_apps,
        mock_prepare,
        mock_easyocr,
        mock_brightness,
    ):
        services.POLISH_MODE = True
        file = SimpleUploadedFile("test.png", b"abc")
        mock_doc = mock.MagicMock()
        mock_db.return_value.collection.return_value.document.return_value = mock_doc
        result = services.handle_uploaded_file(file)
        self.assertIn("Cześć", result)
        mock_easyocr.assert_called_once()

    @mock.patch("application.utils.validate_image_brightness", return_value=True)
    @mock.patch(
        "api.services.paddle_model.perform_ocr",
        return_value={"text_predictions": ["Debug"], "confidence_scores": [0.95]},
    )
    @mock.patch("api.services.prepare_file_hierarchy", return_value="/tmp/test.png")
    @mock.patch("firebase_admin._apps", new={})
    @mock.patch("firebase_admin.credentials.Certificate")
    @mock.patch("firebase_admin.initialize_app")
    @mock.patch("firebase_admin.firestore.client")
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"binarydata")
    @mock.patch("builtins.print")
    def test_handle_uploaded_file_debug_mode(
        self,
        mock_print,
        mock_open,
        mock_db,
        mock_init,
        mock_cert,
        mock_apps,
        mock_prepare,
        mock_perform_ocr,
        mock_brightness,
    ):
        services.DEBUG_MODE = True
        services.POLISH_MODE = False
        file = SimpleUploadedFile("debug.png", b"abc")
        mock_doc = mock.MagicMock()
        mock_db.return_value.collection.return_value.document.return_value = mock_doc
        result = services.handle_uploaded_file(file)
        self.assertIn("Debug", result)
        mock_print.assert_any_call("PaddleOCR results:")

    @mock.patch("application.utils.validate_image_brightness", return_value=False)
    @mock.patch("api.services.prepare_file_hierarchy")
    def test_handle_uploaded_file_dark_image(self, mock_prepare, mock_brightness):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (10, 10), color="black")
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        mock_prepare.return_value = tmp_path
        file = SimpleUploadedFile("test.png", b"abc")
        result = services.handle_uploaded_file(file)
        self.assertIsNone(result)

        os.remove(tmp_path)

    @mock.patch("os.path.exists", return_value=False)
    @mock.patch("application.utils.validate_image_brightness", return_value=True)
    @mock.patch("firebase_admin._apps", new={})
    @mock.patch("api.services.prepare_file_hierarchy")
    def test_handle_uploaded_file_firebase_key_missing(
        self, mock_prepare, mock_apps, mock_brightness, mock_exists
    ):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img = Image.new("RGB", (10, 10), color="white")
            img.save(tmp, format="PNG")
            tmp_path = tmp.name

        mock_prepare.return_value = tmp_path
        os.environ["FIREBASE_KEY"] = "/nonexistent/key.json"
        file = SimpleUploadedFile("test.png", b"abc")

        with self.assertRaises(Exception) as context:
            services.handle_uploaded_file(file)
        self.assertIn("Firebase credentials not found", str(context.exception))

        os.remove(tmp_path)

    @mock.patch("application.utils.validate_image_brightness", return_value=True)
    @mock.patch(
        "api.services.paddle_model.perform_ocr",
        return_value={"text_predictions": ["Brak UID"], "confidence_scores": [1.0]},
    )
    @mock.patch("firebase_admin.credentials.Certificate")
    @mock.patch("firebase_admin.initialize_app")
    @mock.patch("firebase_admin.firestore.client")
    @mock.patch("firebase_admin._apps", new={})
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"imgdata")
    @mock.patch("api.services.prepare_file_hierarchy", return_value="/tmp/test.png")
    def test_handle_uploaded_file_without_user_uid(
        self,
        mock_prepare,
        mock_open,
        mock_apps,
        mock_db,
        mock_init,
        mock_cert,
        mock_ocr,
        mock_brightness,
    ):
        file = SimpleUploadedFile("test.png", b"abc")
        mock_doc = mock.MagicMock()
        mock_db.return_value.collection.return_value.document.return_value = mock_doc
        result = services.handle_uploaded_file(file)
        self.assertIn("Brak UID", result)

    @mock.patch("application.utils.validate_image_brightness", return_value=True)
    @mock.patch(
        "api.services.easy_model.perform_ocr",
        return_value={"text_predictions": ["Cześć"], "confidence_scores": [0.88]},
    )
    @mock.patch("firebase_admin._apps", new={})
    @mock.patch("firebase_admin.credentials.Certificate")
    @mock.patch("firebase_admin.initialize_app")
    @mock.patch("firebase_admin.firestore.client")
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=b"data")
    @mock.patch("api.services.prepare_file_hierarchy", return_value="/tmp/test.png")
    @mock.patch("builtins.print")
    def test_handle_uploaded_file_polish_debug(
        self,
        mock_print,
        mock_prepare,
        mock_open,
        mock_db,
        mock_init,
        mock_cert,
        mock_apps,
        mock_easyocr,
        mock_brightness,
    ):
        services.POLISH_MODE = True
        services.DEBUG_MODE = True
        file = SimpleUploadedFile("test.png", b"abc")
        mock_doc = mock.MagicMock()
        mock_db.return_value.collection.return_value.document.return_value = mock_doc
        result = services.handle_uploaded_file(file)
        self.assertIn("Cześć", result)
        mock_print.assert_any_call("EasyOCR results:")

    def test_output_processed_as_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            output_path = tmp.name
        services.output_processed_as_txt(["one", "two", "three"], output_path)
        with open(output_path, "r") as f:
            content = f.read()
            self.assertIn("one", content)
        os.remove(output_path)

    def test_output_processed_as_docx(self):
        output_path = tempfile.NamedTemporaryFile(suffix=".docx", delete=False).name
        services.output_processed_as_docx(["Hello", "world"], output_path)
        self.assertTrue(os.path.exists(output_path))
        os.remove(output_path)

    @mock.patch("builtins.print")
    def test_debug_mode_prints(self, mock_print):
        services.DEBUG_MODE = True
        os.environ["UPLOADED_FILES"] = tempfile.gettempdir()
        file = SimpleUploadedFile("test.png", b"abc")

        with mock.patch(
            "django.core.files.storage.FileSystemStorage.save", return_value="test.png"
        ), mock.patch(
            "django.core.files.storage.FileSystemStorage.path",
            return_value="/tmp/test.png",
        ):
            services.prepare_file_hierarchy(file)
        mock_print.assert_called()
