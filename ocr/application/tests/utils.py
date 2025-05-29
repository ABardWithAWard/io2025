from django.contrib.auth.models import User
def create_user(email='admin@example.com', password='admin123', is_admin=False):
    if is_admin:
        User.objects.create_superuser(username=email, password=password, email=email)
    else:
        User.objects.create_user(username=email, password=password, email=email)
    return User