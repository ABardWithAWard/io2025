from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from urllib.parse import urlparse
import time

class FileUploadFrontendTest(StaticLiveServerTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--allow-running-insecure-content')
        cls.driver = webdriver.Chrome(options=chrome_options)
        cls.driver.implicitly_wait(10)

        # TODO
        # LOGIN HERE TO USE LOGGED ADMIN

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

    def handle_mobile_menu(self):
        try:
            toggle_button = WebDriverWait(self.driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'button[aria-label="toggle navigation"]'))
            )

            if toggle_button.is_displayed() and toggle_button.is_enabled():
                # Check if menu items are currently visible
                try:
                    self.driver.find_element(By.XPATH, "//a[contains(text(), 'Kontakt')]")
                    return
                except:
                    toggle_button.click()
                    time.sleep(0.5)

        except TimeoutException:
            # No toggle button found, probably desktop view
            pass

    def check_navbar_presence(self):
        navbar = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "nav"))
        )
        self.assertTrue(navbar.is_displayed(), "Navbar should be present")

        self.handle_mobile_menu()

        try:
            ocr = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Aplikacja OCR')]"))
            )
            self.assertTrue(ocr.is_displayed(), "'Aplikacja OCR' should be present")
            self.assertTrue(ocr.is_enabled(), "'Aplikacja OCR' should be clickable")
        except TimeoutException:
            self.fail("'Aplikacja OCR' link not found in navbar")

        try:
            contact = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Kontakt')]"))
            )
            self.assertTrue(contact.is_displayed(), "'Kontakt' should be present")
            self.assertTrue(contact.is_enabled(), "'Kontakt' should be clickable")
        except TimeoutException:
            self.fail("'Kontakt' link not found in navbar")

        try:
            admin = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Panel administracji')]"))
            )
            self.assertTrue(admin.is_displayed(), "'Panel administracji' should be present")
        except TimeoutException:
            self.fail("'Panel administracji' not found in navbar")

        try:
            login = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//button[contains(text(), 'Login')] | //button[contains(text(), 'login')]"))
            )
            self.assertTrue(login.is_displayed(), "'Login' should be present")
            self.assertTrue(login.is_enabled(), "'Login' should be clickable")
        except TimeoutException:
            # Login might not be present if user is already logged in
            pass

    def test_page_loads_successfully(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()
        body = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        self.assertTrue(body.is_displayed(), "Body should be visible on page load")

    def navigate_to_contact(self):
        self.handle_mobile_menu()
        try:
            kontakt = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Kontakt')]"))
            )
            kontakt.click()
            time.sleep(1)
        except TimeoutException:
            self.fail("Could not find or click 'Kontakt' link")

    def navigate_to_admin(self):
        self.handle_mobile_menu()
        try:
            kontakt = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Panel administracji')]"))
            )
            kontakt.click()
            time.sleep(1)
        except TimeoutException:
            self.fail("Could not find or click 'Panel administracji' link")

    def navigate_to_ocr(self):
        self.handle_mobile_menu()
        try:
            ocr = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Aplikacja OCR')]"))
            )
            ocr.click()
            time.sleep(1)
        except TimeoutException:
            self.fail("Could not find or click 'Aplikacja OCR' link")


    def test_navbar_navigation_from_ocr_to_contact(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()

        self.check_navbar_presence()

        self.navigate_to_contact()

        parsed_url = urlparse(self.driver.current_url)
        self.assertEqual(parsed_url.path, '/contact')

    def test_navbar_navigation_from_contact_to_ocr(self):
        self.driver.get(self.live_server_url + '/contact')
        self.wait_for_react_to_load()

        self.check_navbar_presence()

        self.navigate_to_ocr()

        parsed_url = urlparse(self.driver.current_url)
        self.assertEqual(parsed_url.path, '/')

    def test_navbar_navigation_from_ocr_to_admin(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()

        self.check_navbar_presence()

        self.navigate_to_admin()

        parsed_url = urlparse(self.driver.current_url)
        self.assertEqual(parsed_url.path, '/admin')

    def test_navbar_navigation_from_admin_to_ocr(self):
        self.driver.get(self.live_server_url)
        self.wait_for_react_to_load()

        self.check_navbar_presence()

        self.navigate_to_ocr()

        parsed_url = urlparse(self.driver.current_url)
        self.assertEqual(parsed_url.path, '/')

    def test_navbar_navigation_from_admin_to_contact(self):
        self.driver.get(self.live_server_url + '/admin')
        self.wait_for_react_to_load()

        self.check_navbar_presence()

        self.navigate_to_contact()

        parsed_url = urlparse(self.driver.current_url)
        self.assertEqual(parsed_url.path, '/contact')

    def test_navbar_navigation_from_contact_to_admin(self):
        self.driver.get(self.live_server_url + '/contact')
        self.wait_for_react_to_load()

        self.check_navbar_presence()

        self.navigate_to_admin()

        parsed_url = urlparse(self.driver.current_url)
        self.assertEqual(parsed_url.path, '/admin')









