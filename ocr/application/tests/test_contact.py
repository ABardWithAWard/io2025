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
        # Mock file exists
        mock_exists.return_value = True

        # Mock Firestore client and document operations
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

        # Check response
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        response_data = response.json()
        self.assertEqual(response_data['message'], 'Contact message saved successfully')

        # Verify Firebase operations were called
        mock_db.collection.assert_called_once_with('contacts')
        mock_collection.document.assert_called_once()
        mock_document.set.assert_called_once()

        # Verify the data passed to Firestore
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

    # To fix, user can put whitespaces and it works
    def test_contact_whitespace_only_fields(self):
        invalid_data = {
            'name': '   ',
            'email': 'Test@example.com',
            'message': 'This is a test message.'
        }

        response = self.client.post(
            self.contact,
            data=json.dumps(invalid_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('firebase_admin._apps', [])
    @patch.dict(os.environ, {'FIREBASE_KEY': 'firebaseSecretKey.json'})
    @patch('os.path.exists')
    def test_contact_firebase_credentials_not_found(self, mock_exists):
        # Mock when firebase credentials were unable to found
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
        # Mock fail to add to firebase
        mock_exists.return_value = True

        # Mock Firestore client but make document.set() fail
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

    # Test fails when big inputs
    def test_contact_with_long_inputs(self):
        # Long input values
        long_data = {
            'name': 'A' * 1000,  # Very long name
            'email': 'test@example.com',
            'message': 'B' * 10000  # Very long message
        }

        with patch('firebase_admin.firestore.client') as mock_firestore_client, \
                patch('firebase_admin.initialize_app'), \
                patch('firebase_admin._apps', []), \
                patch.dict(os.environ, {'FIREBASE_KEY': '/path/to/firebase/key.json'}), \
                patch('os.path.exists', return_value=True):
            # Mock Firestore operations
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

            self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    # Test fails when weird inputs
    def test_contact_special_characters(self):
        # Test weird characters in message
        special_data = {
            'name': 'José María O\'Connor',
            'email': 'josé@example.com',
            'message': 'Test message with émojis 🚀 and special chars: @#$%^&*()'
        }

        with patch('firebase_admin.firestore.client') as mock_firestore_client, \
                patch('firebase_admin.initialize_app'), \
                patch('firebase_admin._apps', []), \
                patch.dict(os.environ, {'FIREBASE_KEY': '/path/to/firebase/key.json'}), \
                patch('os.path.exists', return_value=True):
            # Mock Firestore operations
            mock_db = MagicMock()
            mock_firestore_client.return_value = mock_db
            mock_collection = MagicMock()
            mock_document = MagicMock()
            mock_db.collection.return_value = mock_collection
            mock_collection.document.return_value = mock_document

            response = self.client.post(
                self.contact,
                data=json.dumps(special_data),
                content_type='application/json'
            )

            self.assertEqual(response.status_code, status.HTTP_201_CREATED)

            # Check if special characters are preserved
            call_args = mock_document.set.call_args[0][0]
            self.assertEqual(call_args['name'], 'José María O\'Connor')
            self.assertIn('🚀', call_args['message'])