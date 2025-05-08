from django.urls import path, include
from . import views
from . import login
from api.urls import urlpatterns as api_urls

app_name = 'application'

urlpatterns = [
    # API endpoints
    path('api/', include(api_urls)),
    
    # Auth endpoints
    path('auth/login/', login.handle_login, name='handle_login'),
    path('auth/register/', login.handle_register, name='handle_register'),
    path('auth/logout/', login.handle_logout, name='handle_logout'),
    path('auth/google/', login.google_auth, name='handle_google_auth'),
    path('auth/google/callback/', login.google_auth_callback, name='google_auth_callback'),

    # This must come last to prevent overriding all other paths
    path('', views.ReactAppView.as_view(), name='react_app'),
]
