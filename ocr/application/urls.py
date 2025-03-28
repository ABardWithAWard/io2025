from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='index'),
    path("api/files", views.get_files, name="get_files"),
]

