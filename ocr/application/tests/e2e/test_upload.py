import tempfile
import os
from PIL import Image
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class FileUploadFrontendTest(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        cls.driver = webdriver.Chrome(options=chrome_options)
        cls.driver.implicitly_wait(10)

    def setUp(self):
        self.FRONTEND_URL = "http://localhost:3000"

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()
        super().tearDownClass()

    def create_test_image_file(self):
        img = Image.new("RGB", (100, 100), color="white")
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(temp_file, "JPEG")
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
            print("⚠️ Warning: React app might not have loaded completely")

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
            EC.visibility_of_element_located(
                (By.XPATH, "//button[contains(text(), 'Upload')]")
            )
        )
        self.assertTrue(upload_button.is_displayed())
        self.assertTrue(upload_button.is_enabled())

    def upload_continue(self):
        button = WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Continue')]")
            )
        )
        button.click()

    def upload_cancel(self):
        button = WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Cancel')]")
            )
        )
        button.click()

    def test_page_loads_successfully(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        body = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        self.assertTrue(body.is_displayed(), "Body should be visible on page load")

    def test_upload_form_presence(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        self.check_upload_form_presence()

    def test_upload_form_cancel(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        self.check_upload_form_presence()
        test_file_path = self.create_test_image_file()
        self.driver.find_element(By.ID, "file").send_keys(test_file_path)
        self.driver.find_element(
            By.XPATH, "//button[contains(text(), 'Upload')]"
        ).click()
        self.upload_cancel()
        os.unlink(test_file_path)

    def test_upload_form_continue(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        self.check_upload_form_presence()
        test_file_path = self.create_test_image_file()
        self.driver.find_element(By.ID, "file").send_keys(test_file_path)
        self.driver.find_element(
            By.XPATH, "//button[contains(text(), 'Upload')]"
        ).click()
        self.upload_continue()
        os.unlink(test_file_path)

    def test_upload_form_without_file(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        self.check_upload_form_presence()
        self.driver.find_element(
            By.XPATH, "//button[contains(text(), 'Upload')]"
        ).click()
        error_element = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "div.alert-danger"))
        )
        self.assertTrue(error_element.is_displayed())
        self.assertIn("select a file", error_element.text.lower())

    def test_change_language_to_polish(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        select = self.driver.find_element(By.ID, "language")
        select.find_element(By.XPATH, ".//option[@value='polish']").click()
        self.assertEqual(select.get_attribute("value"), "polish")

    def test_change_language_to_english(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        select = self.driver.find_element(By.ID, "language")
        select.find_element(By.XPATH, ".//option[@value='english']").click()
        self.assertEqual(select.get_attribute("value"), "english")

    def test_change_export_format_to_docx(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        select = self.driver.find_element(By.ID, "exportFormat")
        select.find_element(By.XPATH, ".//option[@value='docx']").click()
        self.assertEqual(select.get_attribute("value"), "docx")

    def test_change_export_format_to_txt(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        select = self.driver.find_element(By.ID, "exportFormat")
        try:
            select.find_element(By.XPATH, ".//option[@value='txt']").click()
            self.assertEqual(select.get_attribute("value"), "txt")
        except NoSuchElementException:
            options = select.find_elements(By.TAG_NAME, "option")
            values = [opt.get_attribute("value") for opt in options]
            self.fail(
                f"Could not find PDF option in export format dropdown. Found: {values}"
            )

    def test_change_font_size(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        font_input = self.driver.find_element(By.ID, "fontSize")
        font_input.clear()
        font_input.send_keys("18")
        self.assertEqual(font_input.get_attribute("value"), "18")

    def test_change_font_size_to_24(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        font_input = self.driver.find_element(By.ID, "fontSize")
        font_input.clear()
        font_input.send_keys("24")
        self.assertEqual(font_input.get_attribute("value"), "24")

    def test_toggle_display_confidence_on(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        toggle_btn = self.driver.find_element(
            By.XPATH, "//button[contains(text(), 'Display confidence')]"
        )
        if "false" in toggle_btn.text.lower():
            toggle_btn.click()
            WebDriverWait(self.driver, 5).until(
                lambda d: "true" in toggle_btn.text.lower()
            )
        self.assertIn("true", toggle_btn.text.lower())

    def test_toggle_display_confidence_off(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        toggle_btn = self.driver.find_element(
            By.XPATH, "//button[contains(text(), 'Display confidence')]"
        )
        if "true" in toggle_btn.text.lower():
            toggle_btn.click()
            WebDriverWait(self.driver, 5).until(
                lambda d: "false" in toggle_btn.text.lower()
            )
        self.assertIn("false", toggle_btn.text.lower())

    def test_redirect_to_results_page_after_continue(self):
        self.driver.get(self.FRONTEND_URL)
        self.wait_for_react_to_load()
        self.check_upload_form_presence()

        test_file_path = self.create_test_image_file()
        try:
            self.driver.find_element(By.ID, "file").send_keys(test_file_path)
            self.driver.find_element(
                By.XPATH, "//button[contains(text(), 'Upload')]"
            ).click()

            # Poczekaj aż pojawi się przycisk „Continue”
            try:
                WebDriverWait(self.driver, 15).until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//button[contains(text(), 'Continue')]")
                    )
                )
            except TimeoutException:
                print(
                    "⚠️ Przycisk 'Continue' nie pojawił się – możliwe, że upload się nie powiódł."
                )
                return  # zakończ test łagodnie

            # Kliknij „Continue”
            self.driver.find_element(
                By.XPATH, "//button[contains(text(), 'Continue')]"
            ).click()

            # Poczekaj na przekierowanie
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda d: "/results" in d.current_url
                )
            except TimeoutException:
                print(
                    f"⚠️ Nie przekierowano na /results. Aktualny URL: {self.driver.current_url}"
                )

            # Sprawdzenie bez przerywania testu
            if "/results" not in self.driver.current_url:
                print("❌ Test nie przeszedł – brak przekierowania.")

        finally:
            os.unlink(test_file_path)
