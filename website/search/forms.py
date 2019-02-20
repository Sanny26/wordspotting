from django import forms

class SearchForm(forms.Form):
	imquery = forms.ImageField(label='Upload an image ', required=False, widget=forms.FileInput(
            attrs={
                'style': 'border-color: blue;',
                'placeholder': 'Choose an image file'
            }
        ))
	txtquery = forms.CharField(label='Search for something', required=False, widget=forms.TextInput(
            attrs={
                'class': 'form-control keyboardInput',
                'placeholder': 'Search for something...'
            }
        ))

	