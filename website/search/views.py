from django.shortcuts import render

from .forms import ImSearchForm

import cv2
import numpy as np

# Create your views here.
def index(request):
	page_template =  'search/home.html'
	context = {}
	return render(request, page_template, context)

def query(request):
	page_template =  'search/query.html'
	context = {}
	if request.method == 'POST':
		form1 = ImSearchForm(request.POST, request.FILES)
		if form1.is_valid():
			if 'imquery' in request.FILES:
				fobj = request.FILES['imquery']
				jpeg_array = bytearray(fobj.read())
				img = cv2.imdecode(np.asarray(jpeg_array), 1)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			print(form1.errors)

	form1 = ImSearchForm()
	context['form1'] = form1
	return render(request, page_template, context)
