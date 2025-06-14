from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from application.forms import UploadFileForm, SubmitTicketForm


class UploadFileFormTests(TestCase):
    def test_valid_png_file(self):
        file = SimpleUploadedFile(
            "test.png", b"\x89PNG\r\n\x1a\n", content_type="image/png"
        )
        form = UploadFileForm(files={"file": file})
        self.assertTrue(form.is_valid())

    def test_valid_jpg_file(self):
        file = SimpleUploadedFile(
            "test.jpg", b"\xff\xd8\xff\xe0", content_type="image/jpeg"
        )
        form = UploadFileForm(files={"file": file})
        self.assertTrue(form.is_valid())

    def test_valid_jpeg_file(self):
        file = SimpleUploadedFile(
            "test.jpeg", b"\xff\xd8\xff\xe1", content_type="image/jpeg"
        )
        form = UploadFileForm(files={"file": file})
        self.assertTrue(form.is_valid())

    def test_valid_bmp_file(self):
        file = SimpleUploadedFile("test.bmp", b"BM\xf8", content_type="image/bmp")
        form = UploadFileForm(files={"file": file})
        self.assertTrue(form.is_valid())

    def test_valid_gif_file(self):
        file = SimpleUploadedFile("test.gif", b"GIF89a", content_type="image/gif")
        form = UploadFileForm(files={"file": file})
        self.assertTrue(form.is_valid())

    def test_valid_tiff_file(self):
        file = SimpleUploadedFile("test.tiff", b"II*\x00", content_type="image/tiff")
        form = UploadFileForm(files={"file": file})
        self.assertTrue(form.is_valid())

    def test_valid_webp_file(self):
        file = SimpleUploadedFile(
            "test.webp", b"RIFF....WEBP", content_type="image/webp"
        )
        form = UploadFileForm(files={"file": file})
        self.assertTrue(form.is_valid())

    def test_invalid_file_extension(self):
        bad_file = SimpleUploadedFile(
            "test.txt", b"not an image", content_type="text/plain"
        )
        form = UploadFileForm(files={"file": bad_file})
        self.assertFalse(form.is_valid())
        self.assertIn("Only image files are allowed.", form.errors["file"][0])


class SubmitTicketFormTests(TestCase):
    def test_all_fields_valid(self):
        form_data = {
            "name": "test",
            "email": "test@example.com",
            "message": "Testowa wiadomość",
        }
        form = SubmitTicketForm(data=form_data)
        self.assertTrue(form.is_valid())

    def test_all_fields_missing(self):
        form = SubmitTicketForm(data={})
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)
        self.assertIn("email", form.errors)
        self.assertIn("message", form.errors)

    def test_missing_name_only(self):
        form_data = {"email": "test@example.com", "message": "Wiadomość"}
        form = SubmitTicketForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)

    def test_missing_email_only(self):
        form_data = {"name": "test", "message": "Wiadomość"}
        form = SubmitTicketForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("email", form.errors)

    def test_missing_message_only(self):
        form_data = {"name": "test", "email": "test@example.com"}
        form = SubmitTicketForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("message", form.errors)

    def test_missing_name_and_email(self):
        form_data = {"message": "Wiadomość"}
        form = SubmitTicketForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)
        self.assertIn("email", form.errors)

    def test_missing_name_and_message(self):
        form_data = {"email": "test@example.com"}
        form = SubmitTicketForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("name", form.errors)
        self.assertIn("message", form.errors)

    def test_missing_email_and_message(self):
        form_data = {"name": "test"}
        form = SubmitTicketForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("email", form.errors)
        self.assertIn("message", form.errors)

    def test_invalid_email(self):
        form_data = {
            "name": "test",
            "email": "test-example.com",
            "message": "Testowa wiadomosc",
        }
        form = SubmitTicketForm(data=form_data)
        self.assertFalse(form.is_valid())
        self.assertIn("email", form.errors)
