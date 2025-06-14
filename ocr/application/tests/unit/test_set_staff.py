import unittest
from unittest.mock import patch, MagicMock
import set_staff


class TestSetStaff(unittest.TestCase):

    @patch("set_staff.firebase_admin._apps", new={})
    @patch("set_staff.credentials.Certificate")
    @patch("set_staff.firebase_admin.initialize_app")
    @patch("set_staff.auth.get_user_by_email")
    @patch("set_staff.auth.set_custom_user_claims")
    @patch("set_staff.firestore.client")
    def test_set_staff_status_success(
        self, mock_firestore, mock_set_claims, mock_get_user, mock_init, mock_cert
    ):
        mock_user = MagicMock(uid="123")
        mock_get_user.return_value = mock_user
        mock_doc = MagicMock()
        mock_firestore.return_value.collection.return_value.document.return_value = (
            mock_doc
        )

        with patch("builtins.print") as mock_print:
            set_staff.set_staff_status("test@example.com")
            mock_print.assert_any_call(
                "Successfully made test@example.com a staff member in Firebase"
            )
            mock_set_claims.assert_called_once_with("123", {"is_staff": True})
            mock_doc.set.assert_called()

    @patch(
        "set_staff.auth.get_user_by_email",
        side_effect=set_staff.auth.UserNotFoundError("Not found"),
    )
    def test_set_staff_status_user_not_found(self, mock_get_user):
        with patch("builtins.print") as mock_print:
            set_staff.set_staff_status("notfound@example.com")
            mock_print.assert_any_call(
                "Firebase user with email notfound@example.com does not exist"
            )

    @patch("set_staff.auth.get_user_by_email", side_effect=Exception("Some error"))
    def test_set_staff_status_general_error(self, mock_get_user):
        with patch("builtins.print") as mock_print:
            set_staff.set_staff_status("error@example.com")
            mock_print.assert_any_call("Error: Some error")

    @patch("set_staff.firebase_admin._apps", new={})
    @patch("set_staff.credentials.Certificate")
    @patch("set_staff.firebase_admin.initialize_app")
    @patch("set_staff.auth.get_user_by_email")
    @patch("set_staff.auth.set_custom_user_claims")
    @patch("set_staff.firestore.client")
    def test_remove_staff_status_success(
        self, mock_firestore, mock_set_claims, mock_get_user, mock_init, mock_cert
    ):
        mock_user = MagicMock(uid="456")
        mock_get_user.return_value = mock_user
        mock_doc = MagicMock()
        mock_firestore.return_value.collection.return_value.document.return_value = (
            mock_doc
        )

        with patch("builtins.print") as mock_print:
            set_staff.remove_staff_status("remove@example.com")
            mock_print.assert_any_call(
                "Successfully removed staff status from remove@example.com"
            )
            mock_set_claims.assert_called_once_with("456", {"is_staff": False})
            mock_doc.delete.assert_called_once()

    @patch("set_staff.firestore.client")
    def test_list_staff_success(self, mock_firestore):
        mock_doc1 = MagicMock()
        mock_doc1.to_dict.return_value = {
            "email": "staff1@example.com",
            "uid": "uid1",
            "added_at": "2024-01-01T00:00:00Z",
        }
        mock_doc2 = MagicMock()
        mock_doc2.to_dict.return_value = {
            "email": "staff2@example.com",
            "uid": "uid2",
            "added_at": "2024-01-02T00:00:00Z",
        }

        mock_firestore.return_value.collection.return_value.stream.return_value = [
            mock_doc1,
            mock_doc2,
        ]

        with patch("builtins.print") as mock_print:
            set_staff.list_staff()
            mock_print.assert_any_call("Email: staff1@example.com")
            mock_print.assert_any_call("Email: staff2@example.com")

    @patch("set_staff.firestore.client", side_effect=Exception("Firestore error"))
    def test_list_staff_error(self, mock_firestore):
        with patch("builtins.print") as mock_print:
            set_staff.list_staff()
            mock_print.assert_any_call("Error: Firestore error")


if __name__ == "__main__":
    unittest.main()
