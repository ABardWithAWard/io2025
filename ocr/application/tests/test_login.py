from django.contrib.auth.models import User
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from .utils import create_user


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

    def login(self, email, password):
        login_button = self.browser.find_element(By.NAME, "loginButton").click()

        email_input = self.browser.find_element(By.NAME, "email")
        password_input = self.browser.find_element(By.NAME, "password")
        email_input.send_keys(email)
        password_input.send_keys(password)

        login_button = self.browser.find_element(By.XPATH, "//button[@type='submit' and contains(@class, 'submit-btn')]").click()
        time.sleep(10)

    def logout(self):
        logout_button = self.browser.find_element(By.NAME, "logoutButton").click()

    def test_login_admin(self):
        self.browser.get(f'{self.live_server_url}/application/')
        self.login("admin@example.com", "admin")
        time.sleep(1)
        self.logout()

    def test_login_user(self):
        self.browser.get(f'{self.live_server_url}/application/')
        self.login("user@example.com", "user")
        time.sleep(1)
        self.logout()
        time.sleep(1)





