from django.utils.deprecation import MiddlewareMixin
from django.contrib import messages
from django.shortcuts import redirect


#
class AdminSessionTimeoutMiddleware(MiddlewareMixin):
    """
    Django middleware to handle session timeout for admin users.
    Sets admin session expiry to 5 minutes and redirects expired sessions to login.
    """

    def process_request(self, request):
        if request.path.startswith("/admin/"):
            if request.user.is_authenticated:
                # Session expire time for admin set at 5 minutes
                request.session.set_expiry(300)

            elif "_auth_user_id" in request.session:
                # Session expired -> Move to login page
                return redirect("/application/")
