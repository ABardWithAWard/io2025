from django.contrib import admin
from .models import fileLimit, dataLimit

class MyAdminSite(admin.AdminSite):
    site_header = "Login do panelu"
    login_template = "admin/login.html"

class limitAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

admin_site = MyAdminSite(name='myadmin')
admin_site.register(dataLimit, limitAdmin)
admin_site.register(fileLimit, limitAdmin)