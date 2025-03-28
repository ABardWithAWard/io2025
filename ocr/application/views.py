import os

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from .forms import UploadFileForm

# Imaginary function to handle an uploaded file.
# HAS TO BE CHANGED!!!
def handle_uploaded_file(file):
    with open(file.temporary_file_path(), "wb+") as destination:
        print(file.temporary_file_path())
        for chunk in file.chunks():
            destination.write(chunk).save()

def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            in_memory_file_obj = request.FILES["file"]
            #before starting to upload files make sure you set up .env variable correctly
            FileSystemStorage(location=os.environ['UPLOADED_FILES']).save(in_memory_file_obj.name, in_memory_file_obj)
            return render(request, "index.html", {"form": form})
    else:
        form = UploadFileForm()
    return render(request, "index.html", {"form": form})


def get_files(request):
    directory = os.environ['UPLOADED_FILES']  # Default to MEDIA_ROOT if not set

    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        files = []

    return JsonResponse(files, safe=False)