from unittest.mock import patch, Mock, MagicMock
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APITestCase, APIClient


class FirebaseRegistrationTestCase(APITestCase):

    def setUp(self):
        # api links
        self.client = APIClient()
        self.register_url = "/api/register/"

    def tearDown(self):
        User.objects.all().delete()

    # Replaces real firebase create_user function with a mock so we can simulate a response
    @patch("firebase_admin.auth.create_user")
    def test_register_user_success(self, mock_create_user):
        mock_user_record = Mock()
        mock_user_record.uid = "test_firebase_uid_123"
        mock_create_user.return_value = mock_user_record

        registration_data = {"email": "test@example.com", "password": "password123"}

        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], "User registered")
        self.assertEqual(response.data["uid"], "test_firebase_uid_123")

        # Verify Firebase auth.create_user was called with correct parameters
        mock_create_user.assert_called_once_with(
            email="test@example.com", password="password123"
        )

    # Replaces real firebase create_user function with a mock so we can simulate a response
    @patch("firebase_admin.auth.create_user")
    def test_register_user_duplicate_email(self, mock_create_user):
        # Tells mock version to call exception when the mock version of auth.create_user used
        mock_create_user.side_effect = Exception("The email address is already in use")

        registration_data = {"email": "existing@example.com", "password": "password123"}

        # Response using registration api and above data
        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("The email address is already in use", response.data["error"])

    def test_register_without_email(self):
        registration_data = {"password": "password123"}
        # Response using registration api and above data
        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Email and password are required", response.data["error"])

    def test_register_without_password(self):
        registration_data = {
            "email": "test@example.com",
        }
        # Response using registration api and above data
        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Email and password are required", response.data["error"])

    def test_register_without_data(self):
        registration_data = {}
        # Response using registration api and above data
        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Email and password are required", response.data["error"])

    # Replaces real firebase create_user function with a mock so we can simulate a response
    @patch("firebase_admin.auth.create_user")
    def test_register_user_firebase_error(self, mock_create_user):
        # Tells mock version to call exception when the mock version of auth.create_user used
        mock_create_user.side_effect = Exception("Firebase authentication failed")

        registration_data = {"email": "test@example.com", "password": "password123"}
        # Response using registration api and above data
        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("Firebase authentication failed", response.data["error"])

    # Replaces real firebase create_user function with a mock so we can simulate a response
    @patch("firebase_admin.auth.create_user")
    def test_register_short_password(self, mock_create_user):
        # Tells mock version to call exception when the mock version of auth.create_user used
        mock_create_user.side_effect = Exception(
            "Password should be at least 6 characters"
        )

        registration_data = {"email": "test@example.com", "password": "123"}
        # Response using registration api and above data
        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn(
            "Password should be at least 6 characters", response.data["error"]
        )

    # Replaces real firebase create_user function with a mock so we can simulate a response
    @patch("firebase_admin.auth.create_user")
    def test_register_invalid_email(self, mock_create_user):
        # Tells mock version to call exception when the mock version of auth.create_user used
        mock_create_user.side_effect = Exception("The email address is badly formatted")

        registration_data = {
            "email": "invalid-email-format",
            "password": "ValidPass123!",
        }
        # Response using registration api and above data
        response = self.client.post(self.register_url, registration_data)
        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("The email address is badly formatted", response.data["error"])
