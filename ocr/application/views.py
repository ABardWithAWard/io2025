import os
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.conf import settings
from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from django.template.loader import render_to_string
from django.urls import resolve
from django.views import View
import json
from django.shortcuts import render

@method_decorator(ensure_csrf_cookie, name='dispatch')
class ReactAppView(TemplateView):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        # Check if the request is for an API endpoint
        path = request.path_info
        if path.startswith('/api/'):
            # Let the API views handle the response
            return JsonResponse({'error': 'Not found'}, status=404)

        # Check if the request wants JSON response
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({'error': 'Not found'}, status=404)

        try:
            # Get the CSRF token
            csrf_token = request.COOKIES.get('csrftoken', '')
            
            # Read the index.html file
            with open(os.path.join(settings.REACT_APP_BUILD_DIR, 'index.html'), 'r') as f:
                html = f.read()
            
            # Replace absolute paths with relative ones
            html = html.replace('src="/static/', f'src="{settings.STATIC_URL}static/')
            html = html.replace('href="/static/', f'href="{settings.STATIC_URL}static/')
            html = html.replace('href="/manifest.json"', f'href="{settings.STATIC_URL}manifest.json"')
            html = html.replace('href="/favicon.ico"', f'href="{settings.STATIC_URL}favicon.ico"')
            html = html.replace('href="/logo192.png"', f'href="{settings.STATIC_URL}logo192.png"')
            
            response = StreamingHttpResponse(
                streaming_content=[html],
                content_type='text/html'
            )
            response['X-Content-Type-Options'] = 'nosniff'
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
            return response
            
        except Exception as e:
            error_html = """
                <div style="text-align: center; margin-top: 50px;">
                    <h1>Error loading React app</h1>
                    <p>Please make sure the React app is built and the build directory is properly configured.</p>
                    <p>Error details: {}</p>
                </div>
            """.format(str(e))
            
            return StreamingHttpResponse(
                streaming_content=[error_html],
                content_type='text/html'
            )

@method_decorator(csrf_exempt, name='dispatch')
class ContactView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            name = data.get('name')
            email = data.get('email')
            message = data.get('message')
            
            print(data)
            
            return JsonResponse({'message': 'Message received successfully'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

def index(request):
    context = {
        'static_url': settings.STATIC_URL,
        'google_client_id': settings.GOOGLE_OAUTH2_CLIENT_ID
    }
    return render(request, 'index.html', context)