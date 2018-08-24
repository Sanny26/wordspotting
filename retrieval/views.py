from django.shortcuts import render, redirect, HttpResponse, reverse
from django.core.paginator import Paginator
# from django.utils.httpwrappers import 

import numpy as np
import cv2
import base64
from io import BytesIO, StringIO
from PIL import Image

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
	model_path = 'saved_models/new-iam.t7'
	kdtree_path = 'saved_models/mohanlal_kdtree.p'
	page2word_path = 'saved_models/page_to_word.p'
	
	context = {}
	chosen_id = request.session['chosen_id']
	if request.method == 'POST':
		form = UploadForm(request.POST, request.FILES)
		if form.is_valid():
			fobj = request.FILES['query']
			jpeg_array = bytearray(fobj.read())
			img = cv2.imdecode(np.asarray(jpeg_array), 1)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img_feat = feature(img, model_path)

			kdtree = open(kdtree_path, 'rb')
			page2word = open(page2word_path, 'rb')

			results = query_word(img_feat, kdtree, page2word) 
			request.session['results'] = results[0]

			request.session['qimg'] = img.tolist()
			return redirect('results')
	else:
		form = UploadForm()

	context['form'] = form

	return render(request, page_template, context)


def show_image(request):
	qimg = np.array(request.session['qimg'])
	
	print(qimg.dtype, qimg.shape)
	qimg = Image.fromarray(np.uint8(qimg))
	response = HttpResponse(content_type="image/png")
	qimg.save(response, "PNG")
	return response


def results(request):
	page_template = 'retrieval/results.html'
	results = request.session['results']
	collection_id = request.session['chosen_id']

	context = {}
	pages = []
	for each in results:
		pages.append(each.split('/')[0]+'.jpg')
	paginator = Paginator(pages, 1)

	display_page = request.GET.get('page')
	pages = paginator.get_page(display_page)

	context['pages'] = pages
	context['qimg'] = reverse('show_image')

	return render(request, page_template, context) 