from django.shortcuts import render

# Create your views here.

def index(request):
    """View function for home page of site."""

    # Render the HTML template index.html
    return render(request, 'application/index.html')