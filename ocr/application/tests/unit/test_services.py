import os
import tempfile
from unittest import mock, TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from application.services import (
    prepare_file_hierarchy,
    output_processed_as_txt,
    output_processed_as_docx
)

class ServicesTests(TestCase):

    @mock.patch.dict(os.environ, {'UPLOADED_FILES': tempfile.gettempdir()})
    def test_prepare_file_hierarchy_saves_file(self):
        content = b"test image data"
        test_file = SimpleUploadedFile("test.png", content, content_type="image/png")
        saved_path = prepare_file_hierarchy(test_file)
        self.assertTrue(os.path.exists(saved_path))
        with open(saved_path, 'rb') as f:
            self.assertEqual(f.read(), content)

        os.remove(saved_path)

    def test_output_processed_as_txt(self):
        words = ["This", "is", "a", "test", "file", "for", "OCR", "output."]
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            output_processed_as_txt(words, tmp.name, line_width=20)
            tmp.seek(0)
            content = tmp.read().decode()

        self.assertIn("This is a test", content)
        self.assertIn("OCR output.", content)
        os.remove(tmp.name)

    def test_output_processed_as_docx(self):
        words = ["This", "is", "a", "test", "file", "for", "OCR", "output."]
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            output_processed_as_docx(words, tmp.name, font_size=12)

        self.assertTrue(os.path.exists(tmp.name))
        os.remove(tmp.name)
