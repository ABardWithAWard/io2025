import tempfile
from PIL import Image
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import time
import os

class FileUploadFrontendTest(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Set up Chrome options for headless testing
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        cls.driver = webdriver.Chrome(options=chrome_options)
        cls.driver.implicitly_wait(10)

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()
        super().tearDownClass()

    def create_test_image_file(self):
        img = Image.new('RGB', (100, 100), color='white')
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file, 'JPEG')
        temp_file.close()
        return temp_file.name

    def wait_for_react_to_load(self, timeout=15):
        try:
            WebDriverWait(self.driver, timeout).until(
                lambda driver: driver.execute_script(
                    "return document.readyState === 'complete' && "
                    "document.querySelector('#root') && "
                    "document.querySelector('#root').children.length > 0"
                )
            )
        except TimeoutException:
            print("Warning: React app might not have loaded completely")

    def check_upload_form_presence(self):
        upload_form = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "uploadForm"))
        )
        self.assertTrue(upload_form.is_displayed(), "Upload form should be present")


        choose_file_input = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "file"))
        )
        self.assertTrue(choose_file_input.is_displayed())

        upload_button = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, "//button[contains(text(), 'Upload')]"))
        )
        self.assertTrue(upload_button.is_displayed())
        self.assertTrue(upload_button.is_enabled())

    def upload_continue(self):
        button = WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]"))
        )
        button.click()

    def upload_cancel(self):
        button = WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Cancel')]"))
        )
        button.click()

    def test_page_loads_successfully(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()
        body = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        self.assertTrue(body.is_displayed(), "Body should be visible on page load")

    def test_upload_form_presence(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()

        self.check_upload_form_presence()

    def test_upload_form_cancel(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()

        self.check_upload_form_presence()

        input_file = self.driver.find_element(By.ID, "file")
        test_file_path = self.create_test_image_file()
        input_file.send_keys(test_file_path)

        upload_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Upload')]")
        upload_button.click()

        self.upload_cancel()

        try:
            os.unlink(test_file_path)
        except OSError:
            pass

    def test_upload_form_continue(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()
        self.check_upload_form_presence()

        input_file = self.driver.find_element(By.ID, "file")
        test_file_path = self.create_test_image_file()
        input_file.send_keys(test_file_path)

        upload_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Upload')]")
        upload_button.click()

        self.upload_continue()

        try:
            os.unlink(test_file_path)
        except OSError:
            pass




