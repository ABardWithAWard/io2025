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

    def tearDown(self):
        User.objects.all().delete()

    def add_name_and_email(self, name, email):
        self.browser.find_element(By.ID, 'id-name').send_keys(name)
        self.browser.find_element(By.ID, 'id-email').send_keys(email)

    def add_content(self, content):
        self.browser.find_element(By.ID, "textarea").send_keys(content)

    def upload(self):
        self.browser.find_element(By.ID, "uploadButton").click()

    def test_contact(self):
        self.browser.get(f'{self.live_server_url}/application/contact/')
        time.sleep(1)
        self.add_name_and_email("Maciek", "Maciek@example.com")
        time.sleep(1)
        self.add_content("Pozdrawiam i życze smacznej kawusi")
        time.sleep(1)
        self.upload()
        time.sleep(1)




