from django.test import TestCase

class SimpleCSRFViewTest(TestCase):
    def test_csrf_endpoint_exists(self):
        # Checking if csrf token exists
        response = self.client.get('/api/csrf-token/')
        self.assertEqual(response.status_code, 200)

    def test_csrf_returns_token(self):
        # checking if csrf token is not None or empty
        response = self.client.get('/api/csrf-token/')
        data = response.json()
        self.assertIn('csrf_token', data)
        self.assertIsNotNone(data['csrf_token'])

    def test_csrf_cookie_is_set(self):
        response = self.client.get('/api/csrf-token/')
        self.assertIn('csrftoken', response.cookies)
