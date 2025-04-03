import os
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from .forms import UploadFileForm
from .services import handle_uploaded_file


def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES["file"])
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
