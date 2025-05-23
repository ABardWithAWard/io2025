import os
from google.oauth2 import id_token
from google.auth.transport import requests
from django.db.migrations import serializer
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings
from django.utils.decorators import method_decorator
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from application.models import dataLimit, fileLimit, blockList, UploadedFile, SupportTicket
from application.forms import UploadFileForm, SubmitTicketForm
from .services import handle_uploaded_file, get_files
from .serializers import (
    UploadedFileSerializer, SupportTicketSerializer
)
from django.http import JsonResponse
import firebase_admin
from firebase_admin import credentials, firestore, auth
import json
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User

class UploadedFileViewSet(viewsets.ModelViewSet):
    queryset = UploadedFile.objects.all()
    serializer_class = UploadedFileSerializer
    permission_classes = [AllowAny]

    @action(detail=False, methods=['post'])
    def upload(self, request):
        print("FILES:", request.FILES)
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Get userUid from form data, it will be null if not provided
            user_uid = request.POST.get('userUid')
            file_instance = handle_uploaded_file(request.FILES["file"], user_uid)
            return Response({'status': 'success'})
        return Response({'status': 'error', 'errors': form.errors}, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def list_files(self, request):
        files = get_files()
        return Response(files)

class SupportTicketViewSet(viewsets.ModelViewSet):
    queryset = SupportTicket.objects.all()
    serializer_class = SupportTicketSerializer

    def create(self, request):
        form = SubmitTicketForm(request.data)
        if form.is_valid():
            ticket = form.save()
            return Response({'status': 'success'})
        return Response({'status': 'error', 'errors': form.errors}, status=status.HTTP_400_BAD_REQUEST)

class CSRFView(APIView):
    permission_classes = [AllowAny]
    @method_decorator(ensure_csrf_cookie)
    def get(self, request):
        return JsonResponse({'csrf_token': get_token(request)})

class ContactAPIView(APIView):
    permission_classes = [AllowAny]  # Allow public access to this endpoint
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            name = data.get('name')
            email = data.get('email')
            message = data.get('message')
            
            if not all([name, email, message]):
                return Response(
                    {'error': 'All fields are required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Initialize Firebase if not already initialized
            if not firebase_admin._apps:
                cred_path = os.environ['FIREBASE_KEY']
                if not os.path.exists(cred_path):
                    return Response(
                        {'error': 'Firebase credentials not found'}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)

            # Get Firestore client
            db = firestore.client()
            
            # Add contact message to Firestore
            contact_ref = db.collection('contacts').document()
            contact_ref.set({
                'name': name,
                'email': email,
                'message': message,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            
            return Response(
                {'message': 'Contact message saved successfully'}, 
                status=status.HTTP_201_CREATED
            )
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@method_decorator(csrf_exempt, name='dispatch')
class RegisterAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:

            email = request.data.get("email")
            password = request.data.get("password")

            if not email or not password:
                return Response({"error": "Email and password are required"}, status=status.HTTP_400_BAD_REQUEST)

            user_record = auth.create_user(email=email, password=password)
            return Response({"message": "User registered", "uid": user_record.uid})

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class GoogleAuthAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            id_token_str = request.data.get("idToken")
            print(id_token_str)

            if not id_token_str:
                return Response(
                    {"error": "ID token is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Firebase initialization
            if not firebase_admin._apps:
                cred_path = os.environ["FIREBASE_KEY"]
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)

            # Verify the ID token
            decoded_token = id_token.verify_oauth2_token(id_token_str, requests.Request())
            email = decoded_token.get("email")
            name = decoded_token.get("name", email.split('@')[0])  # Use name from token or email prefix

            if not email:
                return Response(
                    {"error": "Email not found in token"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                # Try to get existing Firebase user
                firebase_user = auth.get_user_by_email(email)
            except auth.UserNotFoundError:
                # Create new Firebase user if doesn't exist
                firebase_user = auth.create_user(
                    email=email,
                    display_name=name,
                    email_verified=True  # Google accounts are pre-verified
                )

            # Get or create Django user
            user, _ = User.objects.get_or_create(
                email=email,
                defaults={
                    "username": self._generate_unique_username(email),
                    "first_name": name
                }
            )

            # Set session data
            request.session['firebase_uid'] = firebase_user.uid
            request.session['user_email'] = email
            
            # Login the user
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')

            return Response({
                "message": "Google login successful",
                "user": {
                    "email": user.email,
                    "username": user.username,
                    "is_staff": user.is_staff,
                    "firebase_uid": firebase_user.uid
                }
            })

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _generate_unique_username(self, email):
        base_username = email.split("@")[0]
        username = base_username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1
        return username

class LogoutAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            logout(request)
            request.session.flush()  # Clear all session data
            return Response({'message': 'Logout successful'})
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class AuthStatusAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            # Check Django authentication
            is_django_authenticated = request.user.is_authenticated
            
            # Get Firebase auth token from request headers
            auth_header = request.headers.get('Authorization')
            is_firebase_authenticated = False
            firebase_uid = None
            
            if auth_header and auth_header.startswith('Bearer '):
                id_token = auth_header.split('Bearer ')[1]
                try:
                    # Verify Firebase token
                    decoded_token = auth.verify_id_token(id_token)
                    is_firebase_authenticated = True
                    firebase_uid = decoded_token.get('uid')
                    
                    # If Firebase is authenticated but Django isn't, sync the session
                    if is_firebase_authenticated and not is_django_authenticated:
                        email = decoded_token.get('email')
                        if email:
                            user, _ = User.objects.get_or_create(
                                email=email,
                                defaults={
                                    "username": email.split('@')[0],
                                    "first_name": decoded_token.get('name', email.split('@')[0])
                                }
                            )
                            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
                            is_django_authenticated = True
                except Exception as e:
                    print(f"Firebase token verification failed: {str(e)}")
            
            # User is considered authenticated if either Django or Firebase auth is valid
            is_authenticated = is_django_authenticated or is_firebase_authenticated
            
            if is_authenticated:
                return Response({
                    'isAuthenticated': True,
                    'user': {
                        'email': request.user.email if is_django_authenticated else decoded_token.get('email'),
                        'username': request.user.username if is_django_authenticated else decoded_token.get('name', ''),
                        'is_staff': request.user.is_staff if is_django_authenticated else False,
                        'firebase_uid': firebase_uid or request.session.get('firebase_uid')
                    }
                })
            return Response({'isAuthenticated': False})
        except Exception as e:
            print(f"Auth status check error: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@method_decorator(csrf_exempt, name='dispatch')
class LoginAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            id_token_str = request.data.get("idToken")
            if not id_token_str:
                return Response({"error": "ID token is required"}, status=status.HTTP_400_BAD_REQUEST)

            decoded_token = auth.verify_id_token(id_token_str)
            email = decoded_token.get("email")

            if not email:
                return Response({"error": "Email not found in token"}, status=status.HTTP_400_BAD_REQUEST)

            user, _ = User.objects.get_or_create(
                email=email,
                defaults={"username": self._generate_unique_username(email)}
            )

            login(request, user)

            return Response({
                "message": "Login successful",
                "user": {
                    "email": user.email,
                    "username": user.username,
                    "is_staff": user.is_staff
                }
            })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)