import os
from django.http import HttpResponse
from django.conf import settings
from django.views.generic import TemplateView
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.template.loader import render_to_string

@method_decorator(ensure_csrf_cookie, name='dispatch')
class ReactAppView(TemplateView):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        try:
            # Get the CSRF token
            csrf_token = request.COOKIES.get('csrftoken', '')
            
            # Render the template with the CSRF token
            html = render_to_string('index.html', {
                'csrf_token': csrf_token,
                'static_url': settings.STATIC_URL,
            })
            
            return HttpResponse(html)
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