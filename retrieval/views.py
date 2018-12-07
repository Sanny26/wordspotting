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
import requests

from .models import Collections
from .forms import ImSearchForm, TxtSearchForm
from .word_index import query_word
from .extractFeature import feature
from .E2Efeat import imgFeat, txtFeat

def text_query(request):
	page_template = 'retrieval/layouts2.html'
	demo_path = 'media/demo/imgs/'
	context = {}
	cname = request.session['cname']
	Cname = cname.replace('_', ' ')
	context['Cname'] = Cname
	context['cname'] = cname
	collection = Collections.objects.filter(collection_name = Cname)[0]
	model_path = collection.weights_path
	kdtree_path = collection.kdtree_path
	page2word_path = collection.page2word_path
	wrd_pos_fpath = collection.wrd_pos_fpath
	# kdtree_path = 'saved_models/e2e_model/kdtree_synthfeat.p'
	if request.method == 'POST':
		query = request.POST['text_query']
		feat, img = txtFeat(query, model_path)

		kdtree = open(kdtree_path, 'rb')
		page2word = open(page2word_path, 'rb')
		with open(wrd_pos_fpath, 'rb') as fobj:
			wrd_pos = pickle.load(fobj)
		
		results = query_word(feat, kdtree, page2word) 
		request.session['qimg'] = query

		positions = []
		for each in results[0]:
			pos = [int(pos) for pos in wrd_pos[each]]
			positions.append(pos)
		request.session['results'] = results[0]
		request.session['positions'] = positions
		request.session['ftype'] = 'txt'
		return redirect('results')
	
	return render(request, page_template, context)


def index(request):
	page_template =  'retrieval/home.html'
	context = {}
	collections = Collections.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)

def collection_index(request, cname):
	page_template = 'retrieval/chome.html'
	context = {}
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	collection = Collections.objects.filter(collection_name = Cname)[0]
	context['desc'] = collection.desc
	context['Cname'] = Cname
	context['cname'] = cname
	if collection.collection_link != '':
		context['url'] = collection.collection_link

	return render(request, page_template, context)



def image_query(request):
	page_template = 'retrieval/query.html'
	demo_path = 'media/demo/imgs/'
	# KERAS_REST_API_URL = "http://localhost:5000/predict"
	
	cname = request.session['cname']
	Cname = cname.replace('_', ' ')
	collection = Collections.objects.filter(collection_name = Cname)[0]
	model_path = collection.weights_path
	kdtree_path = collection.kdtree_path
	page2word_path = collection.page2word_path
	wrd_pos_fpath = collection.wrd_pos_fpath
	context = {}
	if request.method == 'POST':
		form1 = ImSearchForm(request.POST, request.FILES)
		if form1.is_valid():
			begin = time.time()
			fobj = request.FILES['query']
			jpeg_array = bytearray(fobj.read())
			img = cv2.imdecode(np.asarray(jpeg_array), 1)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			'''
			payload = {"image": jpeg_array}
			begin = time.time()
			r = requests.post(KERAS_REST_API_URL, files=payload).json()
			if r["success"]:
				request.session['results'] = r["results"][0]
				request.session['positions'] = r["positions"]
			print('Query processing time', time.time()-begin)
			request.session['qimg'] = img.tolist()'''
			
			# img_feat = feature(img, model_path)
			img_feat = imgFeat(img, model_path)
			print('Total time to extract feature vector', time.time()-begin)

			begin = time.time()
			kdtree = open(kdtree_path, 'rb')
			page2word = open(page2word_path, 'rb')
			with open(wrd_pos_fpath, 'rb') as fobj:
				wrd_pos = pickle.load(fobj)
			print('Total time to load pickle files', time.time()-begin)

			begin = time.time()
			results = query_word(img_feat, kdtree, page2word) 
			print('Total time to search in KD Tree', time.time()-begin)

			request.session['qimg'] = img.tolist()

			positions = []
			for each in results[0]:
				pos = [int(pos) for pos in wrd_pos[each]]
				positions.append(pos)
			request.session['results'] = results[0]
			request.session['positions'] = positions
			request.session['ftype'] = 'img'
			return redirect('results')
	else:
		form1 = ImSearchForm()

	context['form1'] = form1
	paths = [[demo_path+file,file.split('.')[0]] for file in os.listdir(demo_path)]
	context['dpaths'] = paths
	context['Cname'] = Cname
	context['cname'] = cname

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
	cname = request.session['cname']
	positions = request.session['positions']
	context = {}
	pages = []
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	context['Cname'] = Cname
	collection = Collections.objects.filter(collection_name = Cname)[0]
	context['word_path'] = collection.words_path
	

	## makign demo output
	'''with open('media/demo/positions.txt') as file:
		for i, each in enumerate(results):
			copyfile("media/small_set/"+each, "media/demo/output/54/"+str(i)+each.split('/')[0]+'.jpg')
			file.write(each.split('/')[0]+'.jpg', ", ".format(positions[i]))
	file.close()'''
		

	for i, each in enumerate(results):
		pages.append([each.split('/')[0]+'.jpg', each])

	paginator = Paginator(pages, 6)

	display_page = request.GET.get('page')
	pages = paginator.get_page(display_page)

	context['pages'] = pages
	if request.session['ftype'] == 'img':
		context['qimg'] = reverse('show_image')
	else:
		context['qimg'] = request.session['qimg']
	context['cname'] = cname
	context['ftype'] = request.session['ftype']
	return render(request, page_template, context) 


def view_results(request, page, pid):
	page_template = 'retrieval/view_results.html'
	
	pid = int(pid)
	page = int(page)
	results = request.session['results']
	cname = request.session['cname']
	positions = request.session['positions']

	context = {}

	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	context['Cname'] = Cname
	

	if page!=0:
		pid = (page-1)*6 + pid
	path = results[pid]
	pos = positions[pid]
	nimg_path = "media/uploads/"+path.split('/')[0]+'.jpg'

	if request.session['ftype'] == 'img':
		context['qimg'] = reverse('show_image')
	else:
		context['qimg'] = request.session['qimg']
	request.session['path'] = nimg_path
	request.session['pos'] = pos
	context['nimg'] = reverse('show_page')
	context['ftype'] = request.session['ftype']
	context['pid'] = pid
	if pid!=(len(results)-1):
		context['next_pid'] = pid+1
		context['page'] = 0
	else:
		context['next_pid'] = -1
		context['page'] = 0
	if pid>0:
		context['prev_pid'] = int(pid)-1
		context['page'] = 0
	else:
		context['prev_pid'] = -1
		context['page'] = 0
	context['cname'] = cname
	return render(request, page_template, context)
