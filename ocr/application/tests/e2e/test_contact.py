import tempfile
from PIL import Image
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoAlertPresentException

def get_alert_text(driver):
    try:
        WebDriverWait(driver, 5).until(EC.alert_is_present())
        alert = driver.switch_to.alert
        return alert.text
    except NoAlertPresentException:
        return None

class FileUploadFrontendTest(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Set up Chrome options for headless testing
        chrome_options = Options()
        # chrome_options.add_argument('--headless')
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

    def check_contact_form_presence(self):
        upload_form = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.ID, "contactForm"))
        )
        self.assertTrue(upload_form.is_displayed(), "Contact form should be present")


        name_input = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "name"))
        )
        self.assertTrue(name_input.is_displayed())

        email_input = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "email"))
        )
        self.assertTrue(email_input.is_displayed())

        text_input = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "message"))
        )
        self.assertTrue(text_input.is_displayed())

        submit_button = WebDriverWait(self.driver, 10).until(
            EC.visibility_of_element_located((By.XPATH, "//button[contains(text(), 'Submit')]"))
        )
        self.assertTrue(submit_button.is_displayed())
        self.assertTrue(submit_button.is_enabled())

    def test_page_loads_successfully(self):
        self.driver.get(self.live_server_url + "/contact")
        self.wait_for_react_to_load()
        body = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        self.assertTrue(body.is_displayed(), "Body should be visible on page load")

    def test_contact_form_presence(self):
        self.driver.get(self.live_server_url + "/contact")
        self.wait_for_react_to_load()

        self.check_contact_form_presence()

    def input_name(self, user_name):
        name = self.driver.find_element(By.NAME, "name")
        name.send_keys(user_name)

    def input_email(self, user_email):
        email = self.driver.find_element(By.NAME, "email")
        email.send_keys(user_email)

    def input_message(self, user_message):
        message = self.driver.find_element(By.NAME, "message")
        message.send_keys(user_message)

    def test_contact_form_without_name(self):
        self.driver.get(self.live_server_url + "/contact")
        self.wait_for_react_to_load()

        self.check_contact_form_presence()

        self.input_email("Pozdrawiam@serdecznie.pl")
        self.input_message("Ale ładna dzisiaj pogoda")

        self.driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]").click()

        file_input = self.driver.find_element(By.NAME, "name")
        is_valid = self.driver.execute_script("return arguments[0].checkValidity();", file_input)
        self.assertFalse(is_valid)


    def test_contact_form_without_email(self):
        self.driver.get(self.live_server_url + "/contact")
        self.wait_for_react_to_load()

        self.check_contact_form_presence()

        self.input_name("Kapuś")
        self.input_message("Ale ładna dzisiaj pogoda")

        self.driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]").click()
        file_input = self.driver.find_element(By.NAME, "email")
        is_valid = self.driver.execute_script("return arguments[0].checkValidity();", file_input)
        self.assertFalse(is_valid)

    def test_contact_form_without_message(self):
        self.driver.get(self.live_server_url + "/contact")
        self.wait_for_react_to_load()

        self.check_contact_form_presence()

        self.input_email("Pozdrawiam@serdecznie.pl")
        self.input_name("Kapuś")

        self.driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]").click()
        file_input = self.driver.find_element(By.NAME, "message")
        is_valid = self.driver.execute_script("return arguments[0].checkValidity();", file_input)
        self.assertFalse(is_valid)

    def test_contact_form_success(self):
        self.driver.get(self.live_server_url + "/contact")
        self.wait_for_react_to_load()

        self.check_contact_form_presence()

        self.input_name("Test User")
        self.input_email("test@example.com")
        self.input_message("Wiadomość testowa")

        self.driver.find_element(By.XPATH, "//button[contains(text(), 'Submit')]").click()
        alert_text = get_alert_text(self.driver)
        self.assertEqual(alert_text, "Message sent successfully!")
        self.driver.switch_to.alert.accept()