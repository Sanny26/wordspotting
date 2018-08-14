from django.shortcuts import render, redirect

import numpy as np
import cv2

from .models import Collections
from .forms import SelectForm, UploadForm
from .word_index import query_word
from .extractFeature import feature

def index(request):
	page_template =  'retrieval/home.html'
	context = {}
	
	if request.method == 'POST':
		form = SelectForm(request.POST)
		if form.is_valid():
			name = form.cleaned_data['name']
			request.session['chosen_id'] = Collections.objects.filter(collection_name = name)[0].id
			return redirect('upload_query')
	else:
		form = SelectForm()

	context['form'] = form

	return render(request, page_template, context)


def upload_query(request):
	page_template = 'retrieval/query.html'
	context = {}
	chosen_id = request.session['chosen_id']
	if request.method == 'POST':
		form = UploadForm(request.POST, request.FILES)
		if form.is_valid():
			fobj = request.FILES['query']
			jpeg_array = bytearray(fobj.read())
			img = cv2.imdecode(np.asarray(jpeg_array), 1)
			request.session['img_shape'] = img.shape
			request.session['qbs'] = img.tolist()
			return redirect('results')
	else:
		form = UploadForm()

	context['form'] = form

	return render(request, page_template, context)


def results(request):
	page_template = 'retrieval/results.html'
	model_path = 'saved_models/new-iam.t7'
	kdtree_path = 'saved_models/mohanlal_kdtree.p'
	page2word_path = 'saved_models/page_to_word.p'
	query_image = request.session['qbs']
	img_shape = request.session['img_shape']
	collection_id = request.session['chosen_id']
	context = {}

	img_feat = feature(np.array(query_image), model_path)
	#API_URL = "http://localhost:5000/search"
	#payload = {'image': query_image}

	kdtree = open(kdtree_path, 'rb')
	page2word = open(page2word_path, 'rb')

	results = query_word(img_feat, kdtree, page2word) 

	context['results'] = results[0]
	print(results)
	return render(request, page_template, context) 