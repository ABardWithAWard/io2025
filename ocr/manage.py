#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import atexit
import os
import shutil
import sys


def cleanup_uploaded_files():
    try:
        upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])
        print(f"Cleaning up directory: {upload_dir}")

        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            os.makedirs(upload_dir)
            print("Cleanup completed successfully")
        else:
            print(f"Directory {upload_dir} does not exist")

    except Exception as e:
        print(f"Error during cleanup: {e}")


# Register cleanup function to run on Django server termination
atexit.register(cleanup_uploaded_files)

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
