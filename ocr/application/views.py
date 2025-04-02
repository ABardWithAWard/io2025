import os
import subprocess
import time
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from .forms import UploadFileForm

# Imaginary function to handle an uploaded file.
# HAS TO BE CHANGED!!!
def handle_uploaded_file(file):
    # Get the absolute path from environment variable
    upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])
    print(f"Upload directory: {upload_dir}")
    
    # Save the uploaded file first
    storage = FileSystemStorage(location=upload_dir)
    file_path = storage.save(file.name, file)
    full_path = storage.path(file_path)
    print(f"Saved file to: {full_path}")
    
    # Wait for file to be fully written
    while not os.path.exists(full_path):
        time.sleep(0.1)
    
    # Additional small delay to ensure file is completely written
    time.sleep(0.5)
    
    # Create reversed_images directory if it doesn't exist
    reversed_dir = os.path.join(upload_dir, 'reversed_images')
    os.makedirs(reversed_dir, exist_ok=True)
    print(f"Created reversed images directory: {reversed_dir}")
    
    # Run the color_reverse.py script on the uploaded file
    try:
        # Use the absolute path to the script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'color_reverse.py'))
        print(f"Running script from: {script_path}")
        print(f"Processing directory: {upload_dir}")
        print(f"Output directory: {reversed_dir}")
        
        # Run with shell=True to handle Windows paths better and pass both directories
        subprocess.run(f'python "{script_path}" "{upload_dir}" "{reversed_dir}"', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running color_reverse.py: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES["file"])
            return render(request, "application/upload.html", {"form": form})
    else:
        form = UploadFileForm()
    return render(request, "application/upload.html", {"form": form})


def get_files(request):
    directory = os.environ['UPLOADED_FILES']  # Default to MEDIA_ROOT if not set

    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        files = []

    return JsonResponse(files, safe=False)