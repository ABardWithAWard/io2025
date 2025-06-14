from django import forms
from application.utils import validate_image_file


class UploadFileForm(forms.Form):
    """
    Django form for uploading image files with validation for allowed file types.
    """

    file = forms.FileField(
        validators=[validate_image_file],
        widget=forms.FileInput(
            attrs={"accept": ".png,.jpg,.jpeg,.bmp,.gif,.tiff,.webp"}
        ),
    )


class SubmitTicketForm(forms.Form):
    """
    Django form for submitting support tickets with user contact information and message.
    """

    name = forms.CharField(required=True)
    email = forms.EmailField(required=True)
    message = forms.CharField(widget=forms.Textarea)
