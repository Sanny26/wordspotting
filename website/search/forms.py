from django import forms

class ImSearchForm(forms.Form):
	imquery = forms.ImageField(label='Upload an image ', required=False, widget=forms.FileInput(
            attrs={
                'style': 'border-color: blue;',
                'placeholder': 'Choose an image file'
            }
        ))
	txtquery = forms.CharField(label='Search for something', required=False, widget=forms.TextInput(
            attrs={
                'class': 'form-control',
                'placeholder': 'Search for something...'
            }
        ))

class TxtSearchForm(forms.Form):
	query = forms.CharField(label='Search for something')

	