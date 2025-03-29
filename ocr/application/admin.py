from django.contrib import admin

from .models import fileLimit, dataLimit, blockList

class limitAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

admin.site.register(dataLimit, limitAdmin)
admin.site.register(fileLimit, limitAdmin)
admin.site.register(blockList)