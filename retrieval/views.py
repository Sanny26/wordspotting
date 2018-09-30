from django.shortcuts import render, redirect, HttpResponse, reverse
from django.core.paginator import Paginator

import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import time
import os
from shutil import copyfile

from .models import Collections
from .forms import ImSearchForm, TxtSearchForm
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
	demo_path = 'media/demo/imgs/'
	
	context = {}
	chosen_id = request.session['chosen_id']
	if request.method == 'POST':
		form1 = ImSearchForm(request.POST, request.FILES)
		form2 = TxtSearchForm(request.POST)
		if form1.is_valid():
			name = form1.cleaned_data['name']
			request.session['chosen_id'] = Collections.objects.filter(collection_name = name)[0].id
			begin = time.time()
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
			print(results[0])
			for each in results[0]:
				pos = [int(pos) for pos in wrd_pos[each]]
				positions.append(pos)
			print('Total time to process query', time.time()-begin)
			request.session['results'] = results[0]
			request.session['positions'] = positions
			print(results[0])
			return redirect('results')
		if form2.is_valid():
			name = form2.cleaned_data['name']
			request.session['chosen_id'] = Collections.objects.filter(collection_name = name)[0].id
			query = form2.cleaned_data['query'].encode('utf-8')
			print('!!!!!!!!!!!!!!!!!', query)

			return redirect('upload_query')
	else:
		form1 = ImSearchForm()
		form2 = TxtSearchForm()

	context['form1'] = form1
	context['form2'] = form2
	paths = [[demo_path+file,file.split('.')[0]] for file in os.listdir(demo_path)]
	context['dpaths'] = paths

	return render(request, page_template, context)


def show_image(request):
	qimg = np.array(request.session['qimg'])
	qimg = Image.fromarray(np.uint8(qimg))
	response = HttpResponse(content_type="image/png")
	qimg.save(response, "PNG")
	return response


def show_page(request):
	begin = time.time()
	path = request.session['path']
	pos = request.session['pos']
	nimg = cv2.imread(path)
	y1, y2, x1, x2 = pos
	nimg = cv2.rectangle(nimg, (x1, y1), (x2, y2), (0, 255, 0), 3)
	nimg = Image.fromarray(nimg)
	response = HttpResponse(content_type="image/png")
	nimg.save(response, "PNG")
	print('Time to render total page', time.time()-begin)
	return response


def demo_results(request, img):
	page_template = 'retrieval/demo_results.html'
	context = {}

	files = os.listdir('media/demo/output/'+img)
	files.sort()
	files_out =[[each[1:], '/media/demo/output/'+img+'/'+each] for each in files]
	
	paginator = Paginator(files_out, 6)

	display_page = request.GET.get('page')
	pages = paginator.get_page(display_page)

	context['pages'] = pages
	context['qimg'] = "/media/demo/imgs/"+img+".jpg"
	return render(request, page_template, context) 


def results(request):
	page_template = 'retrieval/results.html'
	
	results = request.session['results']
	collection_id = request.session['chosen_id']
	context = {}
	pages = []
	for i, each in enumerate(results):
		## makign demo output
		#copyfile("media/small_set/"+each, "media/demo/output/54/"+str(i)+each.split('/')[0]+'.jpg')
		pages.append([each.split('/')[0]+'.jpg', each])
	
	paginator = Paginator(pages, 6)

	display_page = request.GET.get('page')
	pages = paginator.get_page(display_page)

	context['pages'] = pages
	context['qimg'] = reverse('show_image')
	return render(request, page_template, context) 


def view_results(request, pid):
	page_template = 'retrieval/view_results.html'
	
	pid = int(pid)
	results = request.session['results']
	collection_id = request.session['chosen_id']
	positions = request.session['positions']

	context = {}
	path = results[pid]
	pos = positions[pid]
	nimg_path = "media/cleaned/"+path.split('/')[0]+'.jpg'
	context['qimg'] = reverse('show_image')
	#request.session['nimg'] = nimg.tolist()
	request.session['path'] = nimg_path
	request.session['pos'] = pos
	context['nimg'] = reverse('show_page')

	context['pid'] = pid
	if pid!=(len(results)-1):
		context['next_pid'] = pid+1
	else:
		context['next_pid'] = -1
	if pid>0:
		context['prev_pid'] = int(pid)-1
	else:
		context['prev_pid'] = -1
	return render(request, page_template, context)
