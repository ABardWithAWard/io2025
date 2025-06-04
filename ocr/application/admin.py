from django.contrib import admin
from django import forms
from django.template.response import TemplateResponse
from django.conf import settings
from django.shortcuts import redirect
from django.urls import path
import os
import firebase_admin
from firebase_admin import credentials, firestore
from .models import FirebaseDataLimit, FirebaseFileLimit


def get_firestore_client():
    if not firebase_admin._apps:
        cred_path = os.path.join(settings.BASE_DIR, "firebaseSecretKey.json")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

class DataLimitForm(forms.Form):
    dataLimit = forms.IntegerField(label="Limit danych")

class FileLimitForm(forms.Form):
    fileLimit = forms.IntegerField(label="Limit plików")

class FirebaseDataLimitAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def get_actions(self, request):
        actions = super().get_actions(request)
        actions.pop('delete_selected', None)
        return actions

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                'change/',
                self.admin_site.admin_view(self.change_view),
                name='application_firedatalimit_change',
            ),
        ]
        return custom_urls + urls

    def changelist_view(self, request, extra_context=None):
        return redirect('admin:application_firedatalimit_change')

    def change_view(self, request):
        db = get_firestore_client()
        doc_ref = db.collection("global_settings").document("limits")
        doc = doc_ref.get()
        limits = doc.to_dict() if doc.exists else {}

        if request.method == "POST":
            form = DataLimitForm(request.POST)
            if form.is_valid():
                limits['dataLimit'] = form.cleaned_data['dataLimit']
                doc_ref.set(limits)
                self.message_user(request, "Limit danych zapisany.")
        else:
            form = DataLimitForm(initial={'dataLimit': limits.get('dataLimit', 0)})

        admin_form = admin.helpers.AdminForm(
            form,
            fieldsets=[(None, {'fields': form.fields})],
            prepopulated_fields={},
        )

        opts = self.model._meta
        context = dict(
            self.admin_site.each_context(request),
            title="Zmień limit danych",
            adminform=admin_form,
            object_id="datalimit",
            original="Limit danych",
            media=form.media,
            errors=form.errors,
            opts=opts,
            app_label=opts.app_label,
            add=False,
            change=True,
            has_add_permission=self.has_add_permission(request),
            has_change_permission=self.has_change_permission(request),
            has_delete_permission=self.has_delete_permission(request),
            has_view_permission=self.has_view_permission(request),
            save_as=False,
            save_on_top=self.save_on_top,
            show_delete=False,
            has_editable_inline_admin_formsets=False,
        )
        return TemplateResponse(request, "admin/change_form.html", context)

class FirebaseFileLimitAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def get_actions(self, request):
        actions = super().get_actions(request)
        actions.pop('delete_selected', None)
        return actions

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                'change/',
                self.admin_site.admin_view(self.change_view),
                name='application_firefilelimit_change',
            ),
        ]
        return custom_urls + urls

    def changelist_view(self, request, extra_context=None):
        return redirect('admin:application_firefilelimit_change')

    def change_view(self, request):
        db = get_firestore_client()
        doc_ref = db.collection("global_settings").document("limits")
        doc = doc_ref.get()
        limits = doc.to_dict() if doc.exists else {}

        if request.method == "POST":
            form = FileLimitForm(request.POST)
            if form.is_valid():
                limits['fileLimit'] = form.cleaned_data['fileLimit']
                doc_ref.set(limits)
                self.message_user(request, "Limit plików zapisany.")
        else:
            form = FileLimitForm(initial={'fileLimit': limits.get('fileLimit', 0)})

        admin_form = admin.helpers.AdminForm(
            form,
            fieldsets=[(None, {'fields': form.fields})],
            prepopulated_fields={},
        )

        opts = self.model._meta
        context = dict(
            self.admin_site.each_context(request),
            title="Zmień limit plików",
            adminform=admin_form,
            object_id="filelimit",
            original="Limit plików",
            media=form.media,
            errors=form.errors,
            opts=opts,
            app_label=opts.app_label,
            add=False,
            change=True,
            has_add_permission=self.has_add_permission(request),
            has_change_permission=self.has_change_permission(request),
            has_delete_permission=self.has_delete_permission(request),
            has_view_permission=self.has_view_permission(request),
            save_as=False,
            save_on_top=self.save_on_top,
            show_delete=False,
            has_editable_inline_admin_formsets=False,
        )
        return TemplateResponse(request, "admin/change_form.html", context)


admin.site.register(FirebaseDataLimit, FirebaseDataLimitAdmin)
admin.site.register(FirebaseFileLimit, FirebaseFileLimitAdmin)
