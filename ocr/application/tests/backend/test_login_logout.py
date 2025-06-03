from django.test import TestCase, Client
from unittest.mock import patch
import json


class LoginLogoutAPITestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.login = '/api/login/'
        self.logout = '/api/logout/'

    def test_logout_success(self):
        # Mock logged in session
        session = self.client.session
        session['firebase_uid'] = 'test_firebase_uid'
        session['user_email'] = 'test@example.com'
        session.save()

        response = self.client.post(self.logout)
        # Check if everything went right, check if message is correct
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['message'], 'Logout successful')
        # Check if session is cleared
        self.assertNotIn('firebase_uid', self.client.session)
        self.assertNotIn('user_email', self.client.session)

    def test_logout_handles_errors(self):
        # Mock session
        session = self.client.session
        session['firebase_uid'] = 'test_firebase_uid'
        session['user_email'] = 'test@example.com'
        session.save()

        # Try patching where logout is actually imported in your views
        with patch('api.views.logout') as mock_logout:  # Adjust path to your views
            mock_logout.side_effect = Exception("Logout error")

            response = self.client.post(self.logout)
            self.assertEqual(response.status_code, 500)
            response_data = json.loads(response.content)
            self.assertIn('error', response_data)


    # To be fixed!!!
    @patch('firebase_admin.auth.verify_id_token')
    @patch('api.views.LoginAPIView._generate_unique_username')
    def test_login_success(self, mock_generate_username, mock_verify_token):
        mock_generate_username.return_value = 'test_user'
        # Mock Firebase token verification
        mock_verify_token.return_value = {
            'email': 'test@example.com',
            'uid': 'firebase_uid_123'
        }

        # Fake firebase id token
        login_data = {
            'idToken': 'fake_firebase_token'
        }

        # Request with json data to login endpoint
        response = self.client.post(self.login,
                                    data=json.dumps(login_data),
                                    content_type='application/json')

        # DEBUG: Print actual error if not 200
        if response.status_code != 200:
            print(f"Status Code: {response.status_code}")
            print(f"Response Content: {response.content.decode()}")
            # Try to parse as JSON to see error details
            try:
                error_data = json.loads(response.content)
                print(f"Error Data: {error_data}")
            except:
                print("Could not parse response as JSON")

        # Check if response code is positive
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['message'], 'Login successful')
        self.assertIn('user', response_data)
        self.assertEqual(response_data['user']['email'], 'test@example.com')

        # Fix session checking - refresh session after POST request
        # Method 1: Reload session
        session = self.client.session
        session.load()  # Refresh session data

        # Check if session is created
        self.assertEqual(session['firebase_uid'], 'firebase_uid_123')
        self.assertEqual(session['user_email'], 'test@example.com')

    def test_login_missing_token(self):
        # Mock login without id token
        response = self.client.post(self.login,
                                    data=json.dumps({}),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'ID token is required')

    @patch('firebase_admin.auth.verify_id_token')
    def test_login_invalid_token(self, mock_verify_token):
        # Mock login with invalid id token
        mock_verify_token.side_effect = Exception("Invalid token")

        login_data = {
            'idToken': 'invalid_firebase_token'
        }

        response = self.client.post(self.login,
                                    data=json.dumps(login_data),
                                    content_type='application/json')

        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.content)
        self.assertIn('error', response_data)

    @patch('firebase_admin.auth.verify_id_token')
    def test_login_missing_email_in_token(self, mock_verify_token):
        # Mock login without email
        mock_verify_token.return_value = {
            'uid': 'firebase_uid_123'
        }

        login_data = {
            'idToken': 'fake_firebase_token'
        }

        response = self.client.post(self.login,
                                    data=json.dumps(login_data),
                                    content_type='application/json')

        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertEqual(response_data['error'], 'Email not found in token')