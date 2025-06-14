import json
import os
from unittest.mock import patch, MagicMock
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from firebase_admin import auth


class FirebaseAuth(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.mock_firebase_user = {
            "uid": "test-firebase-uid-123",
            "email": "test@example.com",
            "display_name": "Test User",
            "email_verified": True,
        }
        # Mock what would id_token.verify_oauth2_token return
        self.mock_decoded_token = {
            "uid": "test-firebase-uid-123",
            "email": "test@example.com",
            "name": "Test User",
            "email_verified": True,
        }


class GoogleAuthTestCase(FirebaseAuth):
    def setUp(self):
        super().setUp()
        self.url = "/api/google-auth/"
        self.valid_token = "valid_google_token"

    @patch.dict(os.environ, {"FIREBASE_KEY": "firebaseSecretKey.json"})
    @patch("google.oauth2.id_token.verify_oauth2_token")
    @patch("firebase_admin.auth.get_user_by_email")
    @patch("firebase_admin.initialize_app")
    @patch("firebase_admin._apps", new_callable=list)
    def test_google_auth_existing_user_success(
        self, mock_apps, mock_init_app, mock_get_user, mock_verify_token
    ):
        # Mock empty Firebase apps list
        mock_apps.clear()
        # Mock decoding the Google ID token
        mock_verify_token.return_value = self.mock_decoded_token
        # Mock Firebase user object
        mock_firebase_user = MagicMock()
        mock_firebase_user.uid = self.mock_firebase_user["uid"]
        # Mock fetching existing Firebase user by email
        mock_get_user.return_value = mock_firebase_user

        data = {"idToken": self.valid_token}
        response = self.client.post(self.url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], "Google login successful")
        self.assertEqual(response.data["user"]["email"], "test@example.com")
        self.assertEqual(response.data["user"]["firebase_uid"], "test-firebase-uid-123")

        # Verify Firebase methods were called
        mock_verify_token.assert_called_once()
        mock_get_user.assert_called_once_with("test@example.com")

    @patch.dict(os.environ, {"FIREBASE_KEY": "firebaseSecretKey.json"})
    @patch("google.oauth2.id_token.verify_oauth2_token")
    @patch("firebase_admin.auth.create_user")
    @patch("firebase_admin.auth.get_user_by_email")
    @patch("firebase_admin.initialize_app")
    @patch("firebase_admin._apps", new_callable=list)
    def test_google_auth_new_user_creation(
        self,
        mock_apps,
        mock_init_app,
        mock_get_user,
        mock_create_user,
        mock_verify_token,
    ):
        # Setup mocks
        mock_apps.clear()
        mock_verify_token.return_value = self.mock_decoded_token
        # Simulate user not found in Firebase
        mock_get_user.side_effect = auth.UserNotFoundError("User not found")

        # Mocks new Firebase user creation
        mock_new_user = MagicMock()
        mock_new_user.uid = self.mock_firebase_user["uid"]
        mock_create_user.return_value = mock_new_user

        data = {"idToken": self.valid_token}
        response = self.client.post(self.url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["message"], "Google login successful")

        # Verify new user creation
        mock_create_user.assert_called_once_with(
            email="test@example.com", display_name="Test User", email_verified=True
        )

    def test_google_auth_missing_token(self):
        # Send empty data to /api/google-auth/
        data = {}
        response = self.client.post(self.url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response.data["error"], "ID token is required")

    @patch("google.oauth2.id_token.verify_oauth2_token")
    def test_google_auth_invalid_token(self, mock_verify_token):
        # Mock invalid token
        mock_verify_token.side_effect = ValueError("Invalid token")

        data = {"idToken": "invalid-token"}
        response = self.client.post(self.url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("error", response.data)

    @patch.dict(os.environ, {"FIREBASE_KEY": "firebaseSecretKey.json"})
    @patch("google.oauth2.id_token.verify_oauth2_token")
    @patch("firebase_admin.initialize_app")
    @patch("firebase_admin._apps", new_callable=list)
    def test_google_auth_missing_email_in_token(
        self, mock_apps, mock_init_app, mock_verify_token
    ):
        mock_apps.clear()
        # Mock token without email
        mock_verify_token.return_value = {"uid": "123", "name": "Test"}  # brak email

        data = {"idToken": self.valid_token}
        response = self.client.post(self.url, data, format="json")

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertIn("error", response.data)
