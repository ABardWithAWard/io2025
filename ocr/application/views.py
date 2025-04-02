import os
import subprocess
import time
import shutil
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from .forms import UploadFileForm

# TODO: Connect this to ocr model
def handle_uploaded_file(file):
    # Get the absolute path from environment variable
    upload_dir = os.path.abspath(os.environ['UPLOADED_FILES'])
    print(f"Upload directory: {upload_dir}")
    
    # Save the uploaded file first
    # This code is just to merge file name and .env variable
    storage = FileSystemStorage(location=upload_dir)
    file_path = storage.save(file.name, file)
    full_path = storage.path(file_path)
    
    # Wait for file to be fully written
    while not os.path.exists(full_path):
        print("Not yet!")
        time.sleep(0.1)

    time.sleep(0.5)
    
    # Create reversed_images directory if it doesn't exist
    # Files with names like abc_reverse.png might cause problems later as we have no good way
    # to distinguish them from already reversed files, so I separate differently colored files
    reversed_dir = os.path.join(upload_dir, 'reversed_images')
    os.makedirs(reversed_dir, exist_ok=True)
    print(f"Created reversed images directory: {reversed_dir}")
    
    # Run the color_reverse.py script on the uploaded file
    try:
        # Use the absolute path to the script
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'color_reverse.py'))
        print(f"Running script from: {script_path}")
        print(f"Processing file: {full_path}")

        temp_dir = os.path.join(upload_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        temp_file = os.path.join(temp_dir, file.name)
        shutil.copy2(full_path, temp_file)
        print(f"Copied file to temp directory: {temp_file}")
        
        # Run with shell=True to handle Windows paths better and pass both directories
        subprocess.run(f'python "{script_path}" "{temp_dir}" "{reversed_dir}"', shell=True, check=True)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running color_reverse.py: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    # Haven't tested this functionality on unix

def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES["file"])
            # Redirect to the same page with a GET request to reset the form
            return HttpResponseRedirect(request.path)
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