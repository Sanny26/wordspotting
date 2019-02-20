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
import json

from .models import Collection
from .forms import SearchForm


def about_project(request):
	page_template = 'search/about.html'
	context = {}
	collections = Collection.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)

def redirect_query(request, cname, choice):
	request.session['cname'] = cname
	if choice == '0':
		return redirect('upload_query')
	elif choice == '1':
		return redirect('text_query')
	else:
		return redirect('home')

def index(request):
	page_template =  'search/home.html'
	context = {}
	collections = Collection.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)

def collection_index(request, cname):
	page_template = 'search/chome.html'
	context = {}
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	collection = Collection.objects.filter(collection_name = Cname)[0]
	context['desc'] = collection.desc
	context['Cname'] = Cname
	context['cname'] = cname
	if collection.collection_link != '':
		context['url'] = collection.collection_link
	collections = Collection.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)

def query(request, cname):
	page_template =  'search/search.html'
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	collection = Collection.objects.filter(collection_name = Cname)[0]
	context = {}
	if request.method == 'POST':
		form1 = SearchForm(request.POST, request.FILES)
		if form1.is_valid():
			if 'imquery' in request.FILES:
				fobj = request.FILES['imquery']
				jpeg_array = bytearray(fobj.read())
				img = cv2.imdecode(np.asarray(jpeg_array), 1)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				results, positions = search_img(jpeg_array)
				request.session['results'] = results[0]
				request.session['positions'] = positions
				request.session['cname'] = cname
				return redirect('results')
			elif form1.cleaned_data['txtquery']:
				results, positions = search_txt(form1.cleaned_data['txtquery'])
				request.session['results'] = results[0]
				request.session['positions'] = positions
				request.session['cname'] = cname
				return redirect('results')

		else:
			print(form1.errors)

	form1 = SearchForm()
	context['form1'] = form1
	collections = Collection.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)


def search_txt(query):
	API_URL = 'http://localhost:5000/predict'
	payload = {'text': query}
	r = requests.post(API_URL, data=json.dumps(payload)).json()
	if r['success']:
		results = r['results']
		positions = r['positions']
	else:
		return False

	return results, positions

def search_img(query):
	API_URL = 'http://localhost:5000/predict'
	payload = {'image': query}
	print('sent query')
	r = requests.post(API_URL, files=payload).json()
	print('recevied query')
	if r['success']:
		results = r['results']
		positions = r['positions']
	else:
		return False

	return results, positions

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
	if request.session['mflag'] == 0:
		y1, y2, x1, x2 = pos
		nimg = cv2.rectangle(nimg, (x1, y1), (x2, y2), (0, 255, 0), 3)
	else:
		for each in pos:
			y1, y2, x1, x2 = each
			nimg = cv2.rectangle(nimg, (x1, y1), (x2, y2), (0, 255, 0), 3)	

	nimg = Image.fromarray(nimg)
	response = HttpResponse(content_type="image/png")
	nimg.save(response, "PNG")
	print('Time to render total page', time.time()-begin)
	return response

def demo_results(request, im_id):
	cname = request.session['cname']
	Cname = cname.replace('_', ' ')
	collection = Collection.objects.filter(collection_name = Cname)[0]
	demo_path = collection.demo_path
	model_path = collection.weights_path
	kdtree_path = collection.kdtree_path
	page2word_path = collection.page2word_path
	wrd_pos_fpath = collection.wrd_pos_fpath

	paths = [demo_path+'/'+file for file in os.listdir(demo_path)]
	paths.sort()
	img_path = paths[int(im_id)-1]
	print(img_path)
	img = cv2.imread(img_path, 0)
	img_feat = imgFeat(img, model_path)
	kdtree = open(kdtree_path, 'rb')
	page2word = open(page2word_path, 'rb')
	with open(wrd_pos_fpath, 'rb') as fobj:
		wrd_pos = pickle.load(fobj)
	
	begin = time.time()
	results = query_word(img_feat, kdtree, page2word) 
	
	request.session['qimg'] = img.tolist()

	positions = []
	for each in results[0]:
		pos = [int(pos) for pos in wrd_pos[each]]
		positions.append(pos)
	request.session['results'] = results[0]
	request.session['positions'] = positions
	request.session['ftype'] = 'img'
	return redirect('results')

def results(request):
	page_template = 'search/results.html'
	
	results = request.session['results']
	cname = request.session['cname']
	positions = request.session['positions']
	context = {}
	pages = []
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	context['Cname'] = Cname
	collection = Collection.objects.filter(collection_name = Cname)[0]
	context['word_path'] = collection.words_path
	

	## makign demo output
	'''with open('media/demo/positions.txt') as file:
		for i, each in enumerate(results):
			copyfile("media/small_set/"+each, "media/demo/output/54/"+str(i)+each.split('/')[0]+'.jpg')
			file.write(each.split('/')[0]+'.jpg', ", ".format(positions[i]))
	file.close()'''
		
	print(results)
	for i, each in enumerate(results):
		pages.append([each.split('/')[0]+'.jpg', each, i])

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
	collections = Collection.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context) 

def mresults(request, page):
	page_template = 'search/mresults.html'
	context = {}
	word_results = request.session['results']
	positions = request.session['positions']
	cname = request.session['cname']
	Cname = cname.replace('_', ' ')
	context['Cname'] = Cname

	page = int(page)
	path = word_results[page][-1]
	pos = positions[page]
	nimg_path = "media/uploads/"+path+'.jpg'
	request.session['mflag'] = 1
	request.session['path'] = nimg_path
	request.session['pos'] = pos
	context['nimg'] = reverse('show_page')
	context['ftype'] = request.session['ftype']
	context['page'] = page
	context['mlen'] = min(10, len(word_results))
	context['nid'] = page+1
	context['pid'] = page-1
	if request.session['ftype'] == 'img':
		context['qimg'] = reverse('show_image')
	else:
		context['qimg'] = request.session['qimg']
	collections = Collection.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)

def view_results(request, page, pid):
	page_template = 'search/view_results.html'
	
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
	request.session['mflag'] = 0
	context['nimg'] = reverse('show_page')
	context['ftype'] = request.session['ftype']
	context['pid'] = pid + 1
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
	collections = Collection.objects.all().values_list('collection_name', flat=True) 
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	
	return render(request, page_template, context)


