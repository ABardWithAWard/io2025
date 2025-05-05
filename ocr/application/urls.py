from django.urls import path
from . import views
from . import login

app_name = 'application'

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('files/', views.get_files, name='get_files'),
    path('contact/', views.enter_contact_ticket, name='enter_contact_ticket'),
    path('csrf-token/', views.get_csrf_token, name='get_csrf_token'),
    path('auth/login/', login.handle_login, name='handle_login'),
    path('auth/register/', login.handle_register, name='handle_register'),
    path('auth/logout/', login.handle_logout, name='handle_logout'),
    path('auth/google/', login.google_auth, name='handle_google_auth'),
    path('auth/google/callback/', login.google_auth_callback, name='google_auth_callback'),

    # This must come last to prevent overriding all other paths
    path('', views.ReactAppView.as_view(), name='react_app'),
]
