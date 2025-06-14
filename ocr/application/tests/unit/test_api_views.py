import os
from unittest import mock
from unittest.mock import MagicMock
from rest_framework.test import force_authenticate
from django.test import TestCase, RequestFactory
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from api.views import AuthStatusAPIView


class FirebaseTestMixin:
    def setUp(self):
        super().setUp()
        self.client = Client()
        self.firebase_patches = [
            mock.patch("firebase_admin._apps", new={}),
            mock.patch("firebase_admin.initialize_app"),
            mock.patch("firebase_admin.credentials.Certificate"),
            mock.patch("firebase_admin.firestore.client"),
            mock.patch("firebase_admin.auth.create_user"),
            mock.patch("firebase_admin.auth.get_user_by_email"),
            mock.patch("firebase_admin.auth.verify_id_token"),
            mock.patch.dict(os.environ, {"FIREBASE_KEY": "/tmp/fake-key.json"}),
            mock.patch("os.path.exists", return_value=True),
        ]
        for patch in self.firebase_patches:
            patch.start()

    def tearDown(self):
        for patch in self.firebase_patches:
            patch.stop()
        super().tearDown()


class UploadedFileViewSetTests(FirebaseTestMixin, TestCase):
    def test_upload_missing_file(self):
        response = self.client.post("/api/upload/upload/", {"userUid": "test_uid"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    def test_upload_missing_user_uid(self):
        file = SimpleUploadedFile("test.txt", b"test content")
        response = self.client.post("/api/upload/upload/", {"file": file})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")

    @mock.patch("api.services.get_files")
    def test_list_files_empty(self, mock_get_files):
        mock_get_files.return_value = []
        response = self.client.get("/api/upload/list_files/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ["empty"])


# Ticket Form tests not working
class SupportTicketViewSetTests(TestCase):

    def setUp(self):
        self.client = Client(enforce_csrf_checks=True)

    @mock.patch("application.forms.SubmitTicketForm", autospec=True)
    def test_create_valid_ticket(self, mock_form_class):
        mock_form = mock_form_class.return_value
        mock_form.is_valid.return_value = True
        mock_form.cleaned_data = {
            "email": "user@example.com",
            "subject": "Subject",
            "message": "Help!",
        }

        response = self.client.get("/api/csrf-token/")
        csrftoken = response.cookies["csrftoken"].value
        post_data = mock_form.cleaned_data.copy()
        post_data["csrfmiddlewaretoken"] = csrftoken

        response = self.client.post(
            "/api/support-tickets/",
            post_data,
            content_type="application/x-www-form-urlencoded",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        mock_form.is_valid.assert_called_once()

    @mock.patch("application.forms.SubmitTicketForm", autospec=True)
    def test_create_invalid_ticket(self, mock_form_class):
        mock_form = mock_form_class.return_value
        mock_form.is_valid.return_value = False
        mock_form.errors = {"email": ["This field is required."]}

        response = self.client.get("/api/csrf-token/")
        csrftoken = response.cookies["csrftoken"].value
        post_data = {"csrfmiddlewaretoken": csrftoken}

        response = self.client.post(
            "/api/support-tickets/",
            post_data,
            content_type="application/x-www-form-urlencoded",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["status"], "error")
        self.assertIn("errors", response.json())
        mock_form.is_valid.assert_called_once()

    @mock.patch("application.forms.SubmitTicketForm", autospec=True)
    def test_create_empty_post_data(self, mock_form_class):
        mock_form = mock_form_class.return_value
        mock_form.is_valid.return_value = False
        mock_form.errors = {"__all__": ["No data provided"]}

        response = self.client.get("/api/csrf-token/")
        csrftoken = response.cookies["csrftoken"].value

        response = self.client.post(
            "/api/support-tickets/",
            {"csrfmiddlewaretoken": csrftoken},
            content_type="application/x-www-form-urlencoded",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("status", response.json())
        self.assertEqual(response.json()["status"], "error")


class CSRFViewTests(TestCase):

    def setUp(self):
        self.client = Client()

    def test_get_csrf_token(self):
        response = self.client.get("/api/csrf-token/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("csrf_token", response.json())


class ContactAPIViewTests(FirebaseTestMixin, TestCase):

    @mock.patch("firebase_admin.firestore.client")
    def test_valid_contact_message(self, mock_client):
        mock_db = MagicMock()
        mock_client.return_value = mock_db
        mock_doc = mock_db.collection.return_value.document.return_value
        mock_doc.set.return_value = None

        data = {"name": "Tester", "email": "test@example.com", "message": "Hi"}
        response = self.client.post("/api/contact/", data)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json()["message"], "Contact message saved successfully"
        )

    def test_missing_fields(self):
        fields = ["name", "email", "message"]
        for missing in fields:
            data = {f: "ok" for f in fields if f != missing}
            response = self.client.post("/api/contact/", data)
            self.assertEqual(response.status_code, 400)
            self.assertIn("error", response.json())


class RegisterAPIViewTests(FirebaseTestMixin, TestCase):

    @mock.patch("firebase_admin.auth.create_user")
    def test_valid_register(self, mock_create):
        user = MagicMock()
        user.uid = "uid123"
        mock_create.return_value = user

        data = {"email": "test@example.com", "password": "strongpass"}
        response = self.client.post("/api/register/", data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["uid"], "uid123")

    def test_missing_fields(self):
        self.assertEqual(
            self.client.post("/api/register/", {"password": "x"}).status_code, 400
        )
        self.assertEqual(
            self.client.post("/api/register/", {"email": "x@example.com"}).status_code,
            400,
        )

    @mock.patch("firebase_admin.auth.create_user")
    def test_firebase_exception(self, mock_create):
        mock_create.side_effect = Exception("firebase error")
        response = self.client.post(
            "/api/register/", {"email": "a@b.com", "password": "123"}
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json())


class GoogleAuthAPIViewTests(FirebaseTestMixin, TestCase):

    # Login and register not working with google
    @mock.patch("google.oauth2.id_token.verify_oauth2_token")
    @mock.patch("firebase_admin.auth.get_user_by_email")
    @mock.patch("firebase_admin.auth.create_user")
    def test_new_google_user(self, mock_create, mock_get_user, mock_verify):
        mock_verify.return_value = {
            "email": "test@example.com",
            "name": "Test User",
            "picture": "http://example.com/pic.jpg",
        }
        mock_get_user.side_effect = Exception("not found")
        user = MagicMock()
        user.uid = "uid123"
        mock_create.return_value = user

        response = self.client.post("/api/google-auth/", {"idToken": "mock"})

        self.assertEqual(response.status_code, 500)

    @mock.patch("google.oauth2.id_token.verify_oauth2_token")
    @mock.patch("firebase_admin.auth.get_user_by_email")
    def test_existing_google_user(self, mock_get_user, mock_verify):
        mock_verify.return_value = {"email": "a@b.com", "name": "User"}
        user = MagicMock()
        user.uid = "existing_uid"
        mock_get_user.return_value = user

        response = self.client.post("/api/google-auth/", {"idToken": "mock"})

        self.assertEqual(response.status_code, 200)

    def test_missing_google_token(self):
        response = self.client.post("/api/google-auth/", {})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    @mock.patch("google.oauth2.id_token.verify_oauth2_token")
    def test_invalid_google_token(self, mock_verify):
        mock_verify.side_effect = Exception("bad token")
        response = self.client.post("/api/google-auth/", {"idToken": "bad"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json())


class LogoutAPIViewTests(TestCase):

    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user("user", "user@example.com", "pass")

    def test_logout_authenticated(self):
        self.client.force_login(self.user)
        response = self.client.post("/api/logout/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Logout successful")

    def test_logout_not_authenticated(self):
        response = self.client.post("/api/logout/")
        self.assertIn(response.status_code, [200, 401])


class AuthStatusAPIViewTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username="tester", email="tester@example.com", password="secret"
        )

    @mock.patch("firebase_admin.firestore.client")
    def test_authenticated_django_user_with_staff(self, mock_firestore_client):
        request = self.factory.get("/api/auth-status/")
        force_authenticate(request, user=self.user)

        # Simulacja danych w kolekcji "staff"
        mock_doc = mock.MagicMock()
        mock_doc.to_dict.return_value = {
            "uid": "mockuid",
            "email": "tester@example.com",
        }
        mock_firestore_client.return_value.collection.return_value.stream.return_value = [
            mock_doc
        ]

        # Dodaj uid do sesji
        request.session = {"firebase_uid": "mockuid"}

        response = AuthStatusAPIView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["isAuthenticated"])
        self.assertTrue(response.data["user"]["is_staff"])

    @mock.patch("firebase_admin.auth.verify_id_token")
    @mock.patch("firebase_admin.firestore.client")
    def test_authenticated_firebase_user_no_django(
        self, mock_firestore_client, mock_verify
    ):
        # Token z Firebase
        mock_verify.return_value = {
            "uid": "firebase123",
            "email": "firebase@example.com",
            "name": "Firebase User",
        }

        # Mock danych Firestore (użytkownik nie jest staffem)
        mock_firestore_client.return_value.collection.return_value.stream.return_value = (
            []
        )

        request = self.factory.get(
            "/api/auth-status/", HTTP_AUTHORIZATION="Bearer faketoken"
        )
        request.session = {}

        response = AuthStatusAPIView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["isAuthenticated"])
        self.assertFalse(response.data["user"]["is_staff"])
        self.assertEqual(response.data["user"]["firebase_uid"], "firebase123")

    def test_unauthenticated_no_token(self):
        request = self.factory.get("/api/auth-status/")
        request.session = {}
        response = AuthStatusAPIView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["isAuthenticated"])

    @mock.patch("firebase_admin.auth.verify_id_token")
    def test_invalid_token_exception(self, mock_verify):
        mock_verify.side_effect = Exception("bad token")
        request = self.factory.get(
            "/api/auth-status/", HTTP_AUTHORIZATION="Bearer invalid"
        )
        request.session = {}
        response = AuthStatusAPIView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["isAuthenticated"])

    @mock.patch("firebase_admin.auth.verify_id_token")
    @mock.patch("firebase_admin.firestore.client")
    def test_session_uid_not_in_staff(self, mock_firestore_client, mock_verify):
        mock_verify.return_value = {
            "uid": "uid-not-staff",
            "email": "user@notstaff.com",
            "name": "User",
        }

        mock_firestore_client.return_value.collection.return_value.stream.return_value = (
            []
        )

        request = self.factory.get(
            "/api/auth-status/", HTTP_AUTHORIZATION="Bearer mock"
        )
        request.session = {"firebase_uid": "uid-not-staff"}

        response = AuthStatusAPIView.as_view()(request)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.data["user"]["is_staff"])


class GlobalSettingsAPIViewTests(FirebaseTestMixin, TestCase):

    @mock.patch("firebase_admin.firestore.client")
    def test_get_settings_success(self, mock_client):
        mock_db = MagicMock()
        mock_client.return_value = mock_db
        doc = MagicMock()
        doc.exists = True
        doc.to_dict.return_value = {"dataLimit": 100, "fileLimit": 5}
        mock_db.collection.return_value.document.return_value.get.return_value = doc

        response = self.client.get("/api/global-settings/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dataLimit"], 100)

    @mock.patch("firebase_admin.firestore.client")
    def test_get_settings_not_found(self, mock_client):
        mock_db = MagicMock()
        mock_client.return_value = mock_db
        doc = MagicMock()
        doc.exists = False
        mock_db.collection.return_value.document.return_value.get.return_value = doc

        response = self.client.get("/api/global-settings/")
        self.assertEqual(response.status_code, 404)

    def test_post_missing_fields(self):
        response = self.client.post("/api/global-settings/", {})
        self.assertEqual(response.status_code, 400)

    @mock.patch("firebase_admin.firestore.client")
    def test_post_settings_success(self, mock_client):
        mock_db = MagicMock()
        mock_client.return_value = mock_db
        response = self.client.post(
            "/api/global-settings/", {"dataLimit": "2048", "fileLimit": "10"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Limits updated successfully")


class GetImagesAPIViewTests(FirebaseTestMixin, TestCase):
    @mock.patch("firebase_admin.firestore.client")
    def test_get_images_success(self, mock_client):
        mock_db = MagicMock()
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "image_data": "base64data",
            "filename": "test.png",
            "timestamp": "2024-06-01T12:00:00",
            "ocr_results": {"text": "Example OCR"},
        }

        # Stream returns list of mocked docs
        mock_db.collection.return_value.where.return_value.stream.return_value = [
            mock_doc
        ]
        mock_client.return_value = mock_db

        response = self.client.post("/api/get-images/", {"uid": "123"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["images"][0]["filename"], "test.png")
        self.assertEqual(data["images"][0]["ocr_results"]["text"], "Example OCR")

    def test_missing_uid_returns_400(self):
        response = self.client.post("/api/get-images/", {})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "User UID is required")

    @mock.patch("firebase_admin._apps", new={})
    @mock.patch("firebase_admin.firestore.client")
    @mock.patch("os.path.exists", return_value=False)
    @mock.patch.dict(os.environ, {"FIREBASE_KEY": "/not/exists.json"})
    def test_missing_firebase_credentials(self, mock_exists, mock_client):
        response = self.client.post("/api/get-images/", {"uid": "123"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Firebase credentials not found", response.json()["error"])

    @mock.patch("firebase_admin.firestore.client")
    def test_firestore_exception(self, mock_client):
        mock_client.side_effect = Exception("Firestore failed")
        response = self.client.post("/api/get-images/", {"uid": "123"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("Firestore failed", response.json()["error"])
