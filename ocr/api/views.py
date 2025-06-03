import os
from google.oauth2 import id_token
from google.auth.transport import requests
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from django.utils.decorators import method_decorator
from django.middleware.csrf import get_token
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from application.models import dataLimit, fileLimit, UploadedFile, SupportTicket
from application.forms import UploadFileForm, SubmitTicketForm
from .services import handle_uploaded_file, get_files
from .serializers import (
    UploadedFileSerializer, SupportTicketSerializer
)
import firebase_admin
from firebase_admin import credentials, firestore, auth
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings
from django.views.generic import TemplateView
from django.contrib.auth import login, logout
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
            name = request.data.get('name')
            email = request.data.get('email')
            message = request.data.get('message')

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
            print(f"Django auth status: {is_django_authenticated}")

            # Get Firebase auth token from request headers
            auth_header = request.headers.get('Authorization')
            print(f"Auth header: {auth_header}")
            
            is_firebase_authenticated = False
            firebase_uid = request.session.get('firebase_uid')
            is_staff = False

            print(f"Session firebase_uid: {firebase_uid}")

            # Initialize Firebase if not already initialized
            if not firebase_admin._apps:
                cred_path = os.environ.get('FIREBASE_KEY', 'firebaseSecretKey.json')
                print(f"Using Firebase credentials from: {cred_path}")
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("Firebase initialized successfully")

            # Check staff status if we have a UID (either from session or token)
            if firebase_uid:
                try:
                    db = firestore.client()
                    print(f"Checking staff collection for UID: {firebase_uid}")
                    
                    # Get all staff members
                    staff_docs = db.collection('staff').stream()
                    print("\nCurrent staff members:")
                    print("-" * 50)
                    
                    # Check if UID exists in staff collection
                    for doc in staff_docs:
                        data = doc.to_dict()
                        print(f"Email: {data.get('email')}")
                        print(f"UID: {data.get('uid')}")
                        print(f"Added at: {data.get('added_at')}")
                        print("-" * 50)
                        
                        # Check if this document's UID matches our user's UID
                        if data.get('uid') == firebase_uid:
                            is_staff = True
                            print(f"Found matching staff UID: {firebase_uid}")
                            break
                    
                    print(f"Final staff status for {firebase_uid}: {is_staff}")
                    
                except Exception as firebase_error:
                    print(f"Error checking staff collection: {str(firebase_error)}")
                    raise

            # If we have an auth header, verify the token and update session
            if auth_header and auth_header.startswith('Bearer '):
                id_token = auth_header.split('Bearer ')[1]
                try:
                    # Verify Firebase token
                    decoded_token = auth.verify_id_token(id_token)
                    is_firebase_authenticated = True
                    token_uid = decoded_token.get('uid')
                    email = decoded_token.get('email')
                    print(f"Email from token: {email}")
                    print(f"UID from token: {token_uid}")
                    print(f"Session UID matches token UID: {firebase_uid == token_uid}")

                    # Update session UID if needed
                    if not firebase_uid or firebase_uid != token_uid:
                        firebase_uid = token_uid
                        request.session['firebase_uid'] = firebase_uid
                        print(f"Set session firebase_uid to: {firebase_uid}")
                        
                        # Recheck staff status with new UID
                        try:
                            db = firestore.client()
                            print(f"Rechecking staff collection for new UID: {firebase_uid}")
                            
                            # Get all staff members
                            staff_docs = db.collection('staff').stream()
                            print("\nCurrent staff members:")
                            print("-" * 50)
                            
                            # Check if UID exists in staff collection
                            for doc in staff_docs:
                                data = doc.to_dict()
                                print(f"Email: {data.get('email')}")
                                print(f"UID: {data.get('uid')}")
                                print(f"Added at: {data.get('added_at')}")
                                print("-" * 50)
                                
                                # Check if this document's UID matches our user's UID
                                if data.get('uid') == firebase_uid:
                                    is_staff = True
                                    print(f"Found matching staff UID: {firebase_uid}")
                                    break
                            
                            print(f"Final staff status for {firebase_uid}: {is_staff}")
                            
                        except Exception as firebase_error:
                            print(f"Error checking staff collection: {str(firebase_error)}")
                            raise

                    # If Firebase is authenticated but Django isn't, sync the session
                    if is_firebase_authenticated and not is_django_authenticated:
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
            print(f"Final auth status - Authenticated: {is_authenticated}, Staff: {is_staff}")

            if is_authenticated:
                return Response({
                    'isAuthenticated': True,
                    'user': {
                        'email': request.user.email if is_django_authenticated else decoded_token.get('email'),
                        'username': request.user.username if is_django_authenticated else decoded_token.get('name', ''),
                        'is_staff': is_staff,
                        'firebase_uid': firebase_uid
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


@method_decorator(ensure_csrf_cookie, name='dispatch')
class ReactAppView(TemplateView):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        # Check if the request is for an API endpoint
        path = request.path_info
        if path.startswith('/api/'):
            # Let the API views handle the response
            return JsonResponse({'error': 'Not found'}, status=404)

        # Check if the request wants JSON response
        if request.headers.get('Accept') == 'application/json':
            return JsonResponse({'error': 'Not found'}, status=404)

        try:
            # Get the CSRF token
            csrf_token = request.COOKIES.get('csrftoken', '')

            # Read the index.html file
            with open(os.path.join(settings.REACT_APP_BUILD_DIR, 'index.html'), 'r') as f:
                html = f.read()

            # Replace absolute paths with relative ones
            html = html.replace('src="/static/', f'src="{settings.STATIC_URL}static/')
            html = html.replace('href="/static/', f'href="{settings.STATIC_URL}static/')
            html = html.replace('href="/manifest.json"', f'href="{settings.STATIC_URL}manifest.json"')
            html = html.replace('href="/favicon.ico"', f'href="{settings.STATIC_URL}favicon.ico"')
            html = html.replace('href="/logo192.png"', f'href="{settings.STATIC_URL}logo192.png"')

            response = StreamingHttpResponse(
                streaming_content=[html],
                content_type='text/html'
            )
            response['X-Content-Type-Options'] = 'nosniff'
            response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
            return response

        except Exception as e:
            error_html = """
                <div style="text-align: center; margin-top: 50px;">
                    <h1>Error loading React app</h1>
                    <p>Please make sure the React app is built and the build directory is properly configured.</p>
                    <p>Error details: {}</p>
                </div>
            """.format(str(e))

            return StreamingHttpResponse(
                streaming_content=[error_html],
                content_type='text/html'
            )