from django.urls import path
from . import views
from . import login

app_name = 'application'

urlpatterns = [
    path('', views.upload_file, name='index'),
    path("api/files", views.get_files, name="get_files"),
    path('contact/', views.contact, name="contact"),
    # Login-related URLs
    path('auth/login/', login.handle_login, name='handle_login'),
    path('auth/register/', login.handle_register, name='handle_register'),
    path('auth/logout/', login.handle_logout, name='logout'),
    path('auth/google/', login.google_auth, name='handle_google_auth'),
    path('auth/google/callback/', login.google_auth_callback, name='google_auth_callback'),
]

