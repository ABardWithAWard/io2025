from django.urls import path, include, re_path
from . import views
from django.contrib import admin
from . import login
from api.urls import urlpatterns as api_urls

app_name = 'application'

urlpatterns = [
    # API endpoints
    path('api/', include(api_urls)),

    path('admin/', admin.site.urls),

    # This must come last to prevent overriding all other paths
    re_path(r'^(?!api/|admin/|media/).*$', views.ReactAppView.as_view(), name='react_app'),
]
