from django.shortcuts import render, redirect, HttpResponse, reverse
from django.core.paginator import Paginator

import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt

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
	wrd_pos_fpath = 'saved_models/positions.pkl'
	
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

			request.session['qimg'] = img.tolist()

			with open(wrd_pos_fpath, 'rb') as fobj:
				wrd_pos = pickle.load(fobj)

			positions = []
			for each in results[0]:
				pos = [int(pos) for pos in wrd_pos[each]]
				positions.append(pos)

			request.session['results'] = results[0]
			request.session['pos'] = positions
			#return redirect('results')
			return redirect(reverse('view_results', args=[0]))
	else:
		form = UploadForm()

	context['form'] = form

	return render(request, page_template, context)


def show_image(request):
	qimg = np.array(request.session['qimg'])
	qimg = Image.fromarray(np.uint8(qimg))
	response = HttpResponse(content_type="image/png")
	qimg.save(response, "PNG")
	return response


def show_page(request):
	qimg = np.array(request.session['nimg'])
	qimg = Image.fromarray(np.uint8(qimg))
	response = HttpResponse(content_type="image/png")
	qimg.save(response, "PNG")
	return response


def results(request):
	page_template = 'retrieval/results.html'
	
	results = request.session['results']
	collection_id = request.session['chosen_id']
	pos = request.session['pos']

	context = {}
	pages = []
	page_ims = []
	for each in results:
		nimg_path = "media/cleaned/"+each.split('/')[0]+'.jpg'
		nimg = cv2.imread(nimg_path)
		y1, y2, x1, x2 = pos
		print(each, pos)
		nimg = cv2.rectangle(nimg, (x1, y1), (x2, y2), (0, 255, 0), 3)
		#cv2.imwrite('test.jpg', nimg)
		#pages.append(each.split('/')[0]+'.jpg')
		page_ims.append(nimg.tolist())
		pages.append(each.split('/')[0]+'.jpg')
	
	paginator = Paginator(pages, 1)

	display_page = request.GET.get('page')
	pages = paginator.get_page(display_page)

	#request.session['page_ims'] = pages_ims
	context['pages'] = pages
	context['qimg'] = reverse('show_image')
	return render(request, page_template, context) 


def view_results(request, pid):
	page_template = 'retrieval/view_results.html'
	
	pid = int(pid)
	results = request.session['results']
	collection_id = request.session['chosen_id']
	positions = request.session['pos']

	context = {}
	path = results[pid]
	pos = positions[pid]
	nimg_path = "media/cleaned/"+path.split('/')[0]+'.jpg'
	nimg = cv2.imread(nimg_path)
	y1, y2, x1, x2 = pos
	nimg = cv2.rectangle(nimg, (x1, y1), (x2, y2), (0, 255, 0), 3)
	context['qimg'] = reverse('show_image')
	request.session['nimg'] = nimg.tolist()
	context['nimg'] = reverse('show_page')

	context['pid'] = pid
	if pid!=(len(pos)-1):
		context['next_pid'] = pid+1
	else:
		context['next_pid'] = -1
	if pid>0:
		context['prev_pid'] = int(pid)-1
	else:
		context['prev_pid'] = -1
	return render(request, page_template, context)
