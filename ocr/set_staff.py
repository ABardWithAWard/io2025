import firebase_admin
from firebase_admin import credentials, auth, firestore
import os


def set_staff_status(email):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebaseSecretKey.json")
            firebase_admin.initialize_app(cred)

        # Get Firebase user by email
        firebase_user = auth.get_user_by_email(email)

        # Set custom claims to make user staff
        auth.set_custom_user_claims(firebase_user.uid, {"is_staff": True})

        # Add UID to staff collection in Firestore
        db = firestore.client()
        staff_ref = db.collection("staff").document(firebase_user.uid)
        staff_ref.set(
            {
                "uid": firebase_user.uid,
                "email": email,
                "added_at": firestore.SERVER_TIMESTAMP,
            }
        )

        print(f"Successfully made {email} a staff member in Firebase")
    except auth.UserNotFoundError:
        print(f"Firebase user with email {email} does not exist")
    except Exception as e:
        print(f"Error: {str(e)}")


def remove_staff_status(email):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebaseSecretKey.json")
            firebase_admin.initialize_app(cred)

        firebase_user = auth.get_user_by_email(email)

        auth.set_custom_user_claims(firebase_user.uid, {"is_staff": False})

        # Remove from staff collection in Firestore
        db = firestore.client()
        staff_ref = db.collection("staff").document(firebase_user.uid)
        staff_ref.delete()

        print(f"Successfully removed staff status from {email}")
    except auth.UserNotFoundError:
        print(f"Firebase user with email {email} does not exist")
    except Exception as e:
        print(f"Error: {str(e)}")


def list_staff():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate("firebaseSecretKey.json")
            firebase_admin.initialize_app(cred)

        # Get all staff members from Firestore
        db = firestore.client()
        staff_docs = db.collection("staff").stream()

        print("\nCurrent staff members:")
        print("-" * 50)
        for doc in staff_docs:
            data = doc.to_dict()
            print(f"Email: {data.get('email')}")
            print(f"UID: {data.get('uid')}")
            print(f"Added at: {data.get('added_at')}")
            print("-" * 50)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python set_staff.py add <email>    - Add staff member")
        print("  python set_staff.py remove <email> - Remove staff member")
        print("  python set_staff.py list          - List all staff members")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "add" and len(sys.argv) == 3:
        set_staff_status(sys.argv[2])
    elif command == "remove" and len(sys.argv) == 3:
        remove_staff_status(sys.argv[2])
    elif command == "list":
        list_staff()
    else:
        print("Invalid command or missing arguments")
        print("Usage:")
        print("  python set_staff.py add <email>    - Add staff member")
        print("  python set_staff.py remove <email> - Remove staff member")
        print("  python set_staff.py list          - List all staff members")
