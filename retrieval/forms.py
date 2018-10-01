from django import forms

from .models import Collections


class ImSearchForm(forms.Form):

	name = forms.ModelChoiceField(label='Choose a collection',
								 queryset=Collections.objects.all(),
								 initial=Collections.objects.filter(id = 0))
	query = forms.ImageField(label='Upload a query image')


class TxtSearchForm(forms.Form):

	name = forms.ModelChoiceField(label='Choose a collection',
								 queryset=Collections.objects.all(),
								 initial=Collections.objects.filter(id = 0))
	query = forms.CharField(label='Enter your query')

	