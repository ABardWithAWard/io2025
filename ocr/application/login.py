import os
from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.contrib.auth import login as auth_login, authenticate, logout
from django.contrib.auth.models import User
from django.contrib import messages
import requests
from urllib.parse import urlencode
import json

def handle_login(request):
    """Handle regular email/password login"""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        # Authenticate using email as username
        user = authenticate(username=email, password=password)
        
        if user is not None:
            auth_login(request, user)
            return HttpResponseRedirect('/application/')
        else:
            messages.error(request, 'Invalid email or password')
            return HttpResponseRedirect('/application/')
    
    return HttpResponseRedirect('/application/')

def handle_register(request):
    """Handle user registration"""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return HttpResponseRedirect('/application/')
        
        if User.objects.filter(username=email).exists():
            messages.error(request, 'Email already registered')
            return HttpResponseRedirect('/application/')
        
        # Create new user
        user = User.objects.create_user(username=email, email=email, password=password)
        # Specify the backend when logging in
        auth_login(request, user, backend='django.contrib.auth.backends.ModelBackend')
        return HttpResponseRedirect('/application/')
    
    return HttpResponseRedirect('/application/')

def handle_logout(request):
    """Handle user logout"""
    logout(request)
    return HttpResponseRedirect('/application/')

def google_auth(request):
    """Initiate Google OAuth2 flow with direct redirect"""
    # Generate state parameter for CSRF protection
    state = os.urandom(16).hex()
    request.session['oauth_state'] = state
    
    # Build the authorization URL
    auth_url = 'https://accounts.google.com/o/oauth2/v2/auth'
    params = {
        'client_id': settings.GOOGLE_OAUTH2_CLIENT_ID,
        'redirect_uri': settings.GOOGLE_OAUTH2_REDIRECT_URI,
        'response_type': 'code',
        'scope': 'email profile',
        'access_type': 'online',
        'state': state,
        'prompt': 'select_account'
    }
    
    # Redirect directly to Google's OAuth page
    return HttpResponseRedirect(f'{auth_url}?{urlencode(params)}')

def google_auth_callback(request):
    """Handle Google OAuth2 callback"""
    # Verify state parameter to prevent CSRF
    state = request.GET.get('state')
    stored_state = request.session.get('oauth_state')
    
    if not state or state != stored_state:
        messages.error(request, 'Invalid state parameter. Possible CSRF attack.')
        return HttpResponseRedirect('/application/')
    
    # Clear the state from session
    if 'oauth_state' in request.session:
        del request.session['oauth_state']
    
    code = request.GET.get('code')
    if not code:
        messages.error(request, 'Authorization code not received')
        return HttpResponseRedirect('/application/')
    
    try:
        # Exchange code for tokens
        token_url = 'https://oauth2.googleapis.com/token'
        token_data = {
            'client_id': settings.GOOGLE_OAUTH2_CLIENT_ID,
            'client_secret': settings.GOOGLE_OAUTH2_CLIENT_SECRET,
            'code': code,
            'redirect_uri': settings.GOOGLE_OAUTH2_REDIRECT_URI,
            'grant_type': 'authorization_code'
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        tokens = token_response.json()
        
        if 'access_token' not in tokens:
            messages.error(request, 'Failed to obtain access token')
            return HttpResponseRedirect('/application/')
        
        # Get user info
        userinfo_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
        headers = {'Authorization': f'Bearer {tokens["access_token"]}'}
        userinfo_response = requests.get(userinfo_url, headers=headers)
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()
        
        email = userinfo.get('email')
        if not email:
            messages.error(request, 'Email not provided by Google')
            return HttpResponseRedirect('/application/')
        
        # Get or create user
        user, created = User.objects.get_or_create(
            username=email,
            defaults={'email': email}
        )
        
        # Log the user in
        auth_login(request, user)
        return HttpResponseRedirect('/application/')
        
    except requests.exceptions.RequestException as e:
        messages.error(request, f'Authentication failed: {str(e)}')
        return HttpResponseRedirect('/application/')
    except Exception as e:
        messages.error(request, f'An unexpected error occurred: {str(e)}')
        return HttpResponseRedirect('/application/')
