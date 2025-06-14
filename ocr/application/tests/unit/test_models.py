from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.utils.timezone import now
from django.core.exceptions import ValidationError
from application.models import UploadedFile, SupportTicket


class UploadedFileModelTests(TestCase):

    def test_file_upload_and_timestamp(self):
        test_file = SimpleUploadedFile(
            "test.png", b"file_content", content_type="image/png"
        )
        uploaded_file = UploadedFile.objects.create(file=test_file)

        # check if object was saved
        self.assertEqual(UploadedFile.objects.count(), 1)

        # check if file name ends with .png
        self.assertTrue(uploaded_file.file.name.endswith(".png"))

        # check if timestamp was added correctly
        self.assertIsNotNone(uploaded_file.uploaded_at)
        self.assertLessEqual(uploaded_file.uploaded_at, now())


class SupportTicketModelTests(TestCase):

    def setUp(self):
        self.valid_data = {
            "name": "test",
            "email": "test@example.com",
            "message": "Testowa wiadomość",
        }

    def test_create_support_ticket(self):
        ticket = SupportTicket.objects.create(**self.valid_data)
        self.assertEqual(SupportTicket.objects.count(), 1)
        self.assertEqual(ticket.name, self.valid_data["name"])
        self.assertEqual(ticket.email, self.valid_data["email"])
        self.assertEqual(ticket.message, self.valid_data["message"])

    def test_string_representation(self):
        ticket = SupportTicket.objects.create(**self.valid_data)
        self.assertIsInstance(str(ticket), str)

    def test_max_length_constraints(self):
        # Use input longer than allowed (201)
        long_name = "a" * 201
        long_email = "b" * 201
        ticket = SupportTicket(name=long_name, email=long_email, message="msg")
        with self.assertRaises(ValidationError):
            ticket.full_clean()

    def test_max_length_invalid_name(self):
        long_name = "a" * 201
        valid_email = "b" * 100
        ticket = SupportTicket(name=long_name, email=valid_email, message="msg")
        with self.assertRaises(ValidationError):
            ticket.full_clean()

    def test_max_length_invalid_email(self):
        valid_name = "a" * 100
        long_email = "b" * 201
        ticket = SupportTicket(name=valid_name, email=long_email, message="msg")
        with self.assertRaises(ValidationError):
            ticket.full_clean()

    def test_max_length_valid(self):
        valid_combinations = [
            ("a" * 1, "b" * 199),
            ("a" * 199, "b" * 1),
            ("a" * 100, "b" * 100),
        ]
        for name, email in valid_combinations:
            ticket = SupportTicket(name=name, email=email, message="msg")
            try:
                ticket.full_clean()
            except ValidationError:
                self.fail(
                    f"ValidationError raised for valid input: name={len(name)}, email={len(email)}"
                )
