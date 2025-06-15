import os
import tempfile
import shutil
import sys
import builtins
import importlib.util
from unittest import TestCase, mock


class TestManage(TestCase):

    def setUp(self):
        # Utwórz tymczasowy folder i ustaw zmienną środowiskową
        self.temp_dir = tempfile.mkdtemp()
        os.environ["UPLOADED_FILES"] = self.temp_dir

    def tearDown(self):
        # Usuń katalog po teście, jeśli jeszcze istnieje
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_cleanup_uploaded_files_removes_directory(self):
        # Umieść plik, by sprawdzić, czy zostanie usunięty
        temp_file = os.path.join(self.temp_dir, "test.txt")
        with open(temp_file, "w") as f:
            f.write("test")

        # Import manage.py i wywołaj cleanup
        spec = importlib.util.spec_from_file_location("manage", "manage.py")
        manage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manage)

        manage.cleanup_uploaded_files()

        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertFalse(os.path.exists(temp_file))  # plik powinien zniknąć

    def test_cleanup_uploaded_files_handles_missing_directory(self):
        shutil.rmtree(self.temp_dir)  # usuń katalog wcześniej

        spec = importlib.util.spec_from_file_location("manage", "manage.py")
        manage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manage)

        # Funkcja powinna odtworzyć katalog
        manage.cleanup_uploaded_files()
        self.assertFalse(os.path.exists(self.temp_dir))

    @mock.patch("django.core.management.execute_from_command_line")
    @mock.patch("os.environ.setdefault")
    def test_main_runs_execute_from_command_line(self, mock_setdefault, mock_execute):
        spec = importlib.util.spec_from_file_location("manage", "manage.py")
        manage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manage)

        with mock.patch("sys.argv", ["manage.py", "runserver"]):
            manage.main()

        mock_setdefault.assert_called_once_with(
            "DJANGO_SETTINGS_MODULE", "ocr.settings"
        )
        mock_execute.assert_called_once_with(["manage.py", "runserver"])
