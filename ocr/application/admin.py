from django.contrib import admin
from .models import fileLimit, dataLimit, blockList

class MyAdminSite(admin.AdminSite):
    site_header = "Login do panelu"
    # Ścieżka pod application/templates
    # Jedyne czym się różni od django/contrib/admin/templates/admin/login.html to dodaniem include naszego navbaru
    # między znacznikiem blokowym nav-global
    login_template = "admin/login.html"

class limitAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

admin_site = MyAdminSite(name='myadmin')
admin_site.register(dataLimit, limitAdmin)
admin_site.register(fileLimit, limitAdmin)
admin_site.register(blockList)