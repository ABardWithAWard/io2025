from django.contrib.auth.models import User
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from .utils import create_user

path = "D:\\pycharm\\PycharmProjects\\io2025-upload\\ocr\\application\\tests\\"

class LoginFunctionalTest(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        options = webdriver.EdgeOptions()
        # options.add_argument("--headless")  # bez GUI
        cls.browser = webdriver.Edge(options=options)
        cls.browser.maximize_window()

    @classmethod
    def tearDownClass(cls):
        cls.browser.quit()
        super().tearDownClass()

    def setUp(self):
        self.browser.delete_all_cookies()
        create_user("admin@example.com", "admin", True)
        create_user("user@example.com", "user", False)

    def tearDown(self):
        User.objects.all().delete()

    def addFile(self, filename, accept):
        file_input = self.browser.find_element(By.NAME, "file")
        file_input.send_keys(path + filename)

        print(path + filename)

        self.browser.find_element(By.ID, "uploadButton").click()

        time.sleep(100)

        if accept:
            self.browser.find_element(By.ID, "continueButton").click()
        else:
            self.browser.find_element(By.ID, "cancelButton").click()

    def test_file_upload(self):
        self.browser.get(f'{self.live_server_url}/application/')
        try:
            self.addFile("abc.jpg", True)
        except Exception as e:
            print(f"Błąd podczas przesyłania pliku: {e}")
        time.sleep(10)

    # def test_uploaded_files(self):
    #     try:
    #         # Czekaj na widoczność kontenera plików
    #         file_list_container = WebDriverWait(self.browser, 10).until(
    #             EC.visibility_of_element_located((By.ID, "file-list"))
    #         )
    #         print("Kontener plików jest widoczny.")
    #     except Exception as e:
    #         print(f"Nie udało się znaleźć widocznego kontenera plików: {e}")
    #         return  # Zakończ test, jeśli kontener nie jest widoczny
    #
    #     links = file_list_container.find_elements(By.XPATH, ".//li/a")
    #     for link in links:
    #         try:
    #             link.click()  # Kliknij na każdy element 'a'
    #             time.sleep(1)
    #         except Exception as e:
    #             print(f"Nie udało się kliknąć linku: {e}")




