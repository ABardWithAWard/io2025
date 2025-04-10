from django.urls import path
from . import views, login

app_name = 'application'

urlpatterns = [
    path('', views.upload_file, name='index'),
    path("api/files", views.get_files, name="get_files"),
    path('contact/', views.contact, name="contact"),
    path('handle-login/', login.handle_login, name='handle_login'),
    path('handle-register/', login.handle_register, name='handle_register'),
    path('handle-google-auth/', login.handle_google_auth, name='handle_google_auth'),
]

