from django.test import LiveServerTestCase
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.webdriver import WebDriver

# class would usually be an entire suite, so this is overkill (but temporary)
class VisitWebsiteTest(LiveServerTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.selenium = WebDriver()
        cls.selenium.implicitly_wait(10)

    @classmethod
    def tearDownClass(cls):
        cls.selenium.quit()
        super().tearDownClass()

    def test_visit(self):
        self.selenium.get(f"{self.live_server_url}")
        assert self.selenium.title == "File Upload"