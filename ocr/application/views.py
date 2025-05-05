import os
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.generic import View, TemplateView
from django.conf import settings
from .forms import UploadFileForm, SubmitTicketForm
from .models import SupportTicket
from .services import handle_uploaded_file
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(ensure_csrf_cookie, name='dispatch')
class ReactAppView(TemplateView):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        try:
            with open(os.path.join(settings.REACT_APP_BUILD_DIR, 'index.html')) as f:
                return HttpResponse(f.read())
        except Exception as e:
            return HttpResponse(
                """
                <div style="text-align: center; margin-top: 50px;">
                    <h1>Error loading React app</h1>
                    <p>Please make sure the React app is built and the build directory is properly configured.</p>
                    <p>Error details: {}</p>
                </div>
                """.format(str(e))
            )

@ensure_csrf_cookie
def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES["file"])
            return JsonResponse({'status': 'success'})
        return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@ensure_csrf_cookie
def get_files(request):
    if request.method == "GET":
        directory = os.environ.get('UPLOADED_FILES', settings.MEDIA_ROOT)
        try:
            files = os.listdir(directory)
            return JsonResponse(files, safe=False)
        except FileNotFoundError:
            return JsonResponse([], safe=False)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@ensure_csrf_cookie
def enter_contact_ticket(request):
    if request.method == "POST":
        form = SubmitTicketForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data["name"]
            email = form.cleaned_data["email"]
            message = form.cleaned_data["message"]
            ticket = SupportTicket(name=name, email=email, message=message)
            ticket.save()
            return JsonResponse({'status': 'success'})
        return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)

@ensure_csrf_cookie
def get_csrf_token(request):
    return JsonResponse({'csrf_token': get_token(request)})