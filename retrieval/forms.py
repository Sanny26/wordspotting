from django import forms

from .models import Collections

class SelectForm(forms.Form):

	name = forms.ModelChoiceField(label='Choose a collection',
								 queryset=Collections.objects.all())


class UploadForm(forms.Form):

	query = forms.ImageField(label='Upload a query image')