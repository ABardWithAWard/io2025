from django.urls import path
from . import views

app_name = 'application'

urlpatterns = [
    path('', views.upload_file, name='index'),
    path("api/files", views.get_files, name="get_files"),
    path('contact/', views.contact, name="contact"),
]

