from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock
import json
import os


class ContactAPITestCase(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.contact = '/api/contact/'
        self.valid_contact_data = {
            'name': 'Test',
            'email': 'Test@example.com',
            'message': 'This is a test message for contact form.'
        }

    @patch('firebase_admin.firestore.client')
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_success(self, mock_exists, mock_init_app, mock_firestore_client):
        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_collection = MagicMock()
        mock_document = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document

        response = self.client.post(
            self.contact,
            data=json.dumps(self.valid_contact_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        response_data = response.json()
        self.assertEqual(response_data['message'], 'Contact message saved successfully')

        mock_db.collection.assert_called_once_with('contacts')
        mock_collection.document.assert_called_once()
        mock_document.set.assert_called_once()

        call_args = mock_document.set.call_args[0][0]
        self.assertEqual(call_args['name'], 'Test')
        self.assertEqual(call_args['email'], 'Test@example.com')
        self.assertEqual(call_args['message'], 'This is a test message for contact form.')
        self.assertIn('timestamp', call_args)

    def test_contact_without_name(self):
        invalid_data = {
            'email': 'Test@example.com',
            'message': 'This is a test message for contact form.'
        }
        response = self.client.post(
            self.contact,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        response_data = response.json()
        self.assertEqual(response_data['error'], 'All fields are required')

    def test_contact_without_email(self):
        invalid_data = {
            'name': 'Test',
            'message': 'This is a test message for contact form.'
        }
        response = self.client.post(
            self.contact,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        response_data = response.json()
        self.assertEqual(response_data['error'], 'All fields are required')

    def test_contact_without_message(self):
        invalid_data = {
            'name': 'Test',
            'email': 'Test@example.com',
        }
        response = self.client.post(
            self.contact,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        response_data = response.json()
        self.assertEqual(response_data['error'], 'All fields are required')

    def test_contact_empty_fields(self):
        invalid_data = {
            'name': '',
            'email': 'Test@example.com',
            'message': 'This is a test message.'
        }

        response = self.client.post(
            self.contact,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        response_data = response.json()
        self.assertEqual(response_data['error'], 'All fields are required')

    @patch('firebase_admin.firestore.client')
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_whitespace_only_fields(self, mock_exists, mock_init_app, mock_firestore_client):
        invalid_data = {
            'name': '   ',
            'email': 'Test@example.com',
            'message': 'This is a test message.'
        }

        mock_exists.return_value = True
        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_collection = MagicMock()
        mock_document = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document

        response = self.client.post(
            self.contact,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_firebase_credentials_not_found(self, mock_exists):
        mock_exists.return_value = False

        response = self.client.post(
            self.contact,
            data=json.dumps(self.valid_contact_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        response_data = response.json()
        self.assertEqual(response_data['error'], 'Firebase credentials not found')

    @patch('firebase_admin.firestore.client')
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_firestore_operation_error(self, mock_exists, mock_init_app, mock_firestore_client):
        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_collection = MagicMock()
        mock_document = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document
        mock_document.set.side_effect = Exception("Firestore write failed")

        response = self.client.post(
            self.contact,
            data=json.dumps(self.valid_contact_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        response_data = response.json()
        self.assertEqual(response_data['error'], 'Firestore write failed')

    @patch('firebase_admin.firestore.client')
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_with_long_inputs(self, mock_exists, mock_init_app, mock_firestore_client):
        long_data = {
            'name': 'ABC',
            'email': 'test@example.com',
            'message': 'B' * 10000
        }

        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_collection = MagicMock()
        mock_document = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document

        response = self.client.post(
            self.contact,
            data=json.dumps(long_data),
            content_type='application/json'
        )

        if response.status_code != status.HTTP_201_CREATED:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.content}")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    @patch('firebase_admin.firestore.client')
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_special_characters(self, mock_exists, mock_init_app, mock_firestore_client):
        special_data = {
            'name': 'José María O\'Connor',
            'email': 'josé@example.com',
            'message': 'Test message with émojis 🚀 and special chars: @#$%^&*()'
        }

        mock_exists.return_value = True

        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_collection = MagicMock()
        mock_document = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document

        response = self.client.post(
            self.contact,
            data=json.dumps(special_data, ensure_ascii=False),
            content_type='application/json; charset=utf-8'
        )

        if response.status_code != status.HTTP_201_CREATED:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.content}")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

        call_args = mock_document.set.call_args[0][0]
        self.assertEqual(call_args['name'], 'José María O\'Connor')
        self.assertIn('🚀', call_args['message'])

    def test_contact_invalid_json(self):
        response = self.client.post(
            self.contact,
            data='invalid json',
            content_type='application/json'
        )

        self.assertIn(response.status_code, [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR])

    def test_contact_wrong_content_type(self):
        response = self.client.post(
            self.contact,
            data=self.valid_contact_data,
            content_type='application/x-www-form-urlencoded'
        )

        self.assertIn(response.status_code, [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR])

    @patch('firebase_admin.firestore.client')
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_extremely_large_payload(self, mock_exists, mock_init_app, mock_firestore_client):
        large_data = {
            'name': 'Test' * 1000,
            'email': 'test@example.com',
            'message': 'X' * 100000
        }

        mock_exists.return_value = True
        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_collection = MagicMock()
        mock_document = MagicMock()
        mock_db.collection.return_value = mock_collection
        mock_collection.document.return_value = mock_document

        response = self.client.post(
            self.contact,
            data=json.dumps(large_data),
            content_type='application/json'
        )

        if response.status_code != status.HTTP_201_CREATED:
            print(f"Large payload response status: {response.status_code}")
            print(f"Large payload response content: {response.content}")

        self.assertIn(response.status_code, [
            status.HTTP_201_CREATED,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_400_BAD_REQUEST
        ])