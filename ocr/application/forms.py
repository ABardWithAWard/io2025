from django import forms
from application.utils import validate_image_file

class UploadFileForm(forms.Form):
    file = forms.FileField(
        validators=[validate_image_file],
        widget=forms.FileInput(attrs={
            'accept': '.png,.jpg,.jpeg,.bmp,.gif,.tiff,.webp'
        })
    )

class SubmitTicketForm(forms.Form):
    name = forms.CharField(required=True)
    email = forms.EmailField(required=True)
    message = forms.CharField(widget=forms.Textarea)