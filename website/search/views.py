from django.shortcuts import render, redirect, HttpResponse, reverse
from django.core.paginator import Paginator
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.conf import settings

import numpy as np
import cv2
from PIL import Image
import pickle

import time
import os
from shutil import copyfile
import requests
import json
import base64

from .models import Collection
from .forms import SearchForm
from .nms import non_max_suppression_fast

# global pos_index
# with open('/media/data1/simple-keras-rest-api/gw/positions.pkl', 'rb') as fobj:
#         pos_index = pickle.load(fobj)

def detail(request):
	page_template = 'search/detail.html'
	context = {}
	collections = Collection.objects.all().values_list('collection_name', flat=True)
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)

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

def search_txt(query, api_cname):
	print(api_cname)
	if api_cname=="lal":
		API_URL = 'http://10.4.16.103:9701/predict'
		# API_URL = 'http://10.4.16.53:9710/predict'
	else:
		API_URL = 'http://10.4.16.103:9700/predict'
	payload = {'text': query.lower(), 'cname':api_cname}
	# payload = {'text': query.lower(), 'cname':'coi_i2'}
	print('Payload: ', payload)
	r = requests.post(API_URL, data=json.dumps(payload)).json()
	if r['success']:
		results = []
		positions = []
		for item in r['results']:
			results.append(item[0])
			positions.append(item[1][0])
	else:
		return False
	# print('API function...', positions)
	# print('API function...', results)
	return results, positions

def search_img(query, api_cname):
	print(api_cname)
	if api_cname=="lal":
		API_URL = 'http://10.4.16.103:9701/predict'
		# API_URL = 'http://10.4.16.53:9710/predict'
	else:
		API_URL = 'http://10.4.16.103:9700/predict'
	# print(query)
	data = {'image': query, 'cname':api_cname}
	#
	headers = {'Content-type': 'application/json'}
	r = requests.post(API_URL, data = json.dumps(data), headers=headers).json()
	if r['success']:
		results = []
		positions = []
		for item in r['results']:
			results.append(item[0])
			positions.append(item[1][0])
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
	results = request.session['results']
	pos_index = request.session['positions']
	path = request.session['path']
	nimg = cv2.imread(path)
	nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
	# print('---------------> Page', path)

	for each in pos_index[request.session['rind']]:
		print('--------------> Pos ', each)
		# y1, y2, x1, x2 = list(map(int, pos_index[each]))
		y1, y2, x1, x2 = each
		nimg = cv2.rectangle(nimg, (x1, y1), (x2, y2), (255, 0, 0), 3)
	nimg = Image.fromarray(nimg)
	# print('--------------------------->', nimg.mode)
	response = HttpResponse(content_type="image/png")
	nimg.save(response, "PNG")
	print('Time to render total page', time.time()-begin)
	return response

def show_line(request, pid):
	pid = int(pid)
	results = request.session['results']
	pos_index = request.session['positions']
	cname = request.session['cname']
	path = results[pid]
	path =  settings.STATIC_PATH +"/files/"+cname+"/uploads/"+path+'.jpg'
	print('---------------->', path)
	nimg = cv2.imread(path)
	nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

	nms_input = []
	lines_bboxes = []
	for ptuple in pos_index[pid]:
		rt, rb, ct, cb = ptuple
		nimg = cv2.rectangle(nimg, (ct, rt), (cb, rb), (255, 0, 0), 5)
		lines_bboxes.append([rt, 0, rb, nimg.shape[1]])
		# nms_input.append([rt, 0, rb, nimg.shape[1]])
	# lines_bboxes = non_max_suppression_fast(np.array(nms_input), 0.3)
	# print(pos_index[pid])
	# print('---->   Len of line boxes', pid, len(lines_bboxes))

	phrase_context_imgs = []
	for y1, _, y2, _ in lines_bboxes:
		img = cv2.copyMakeBorder(nimg[max(0, y1-30):y2+30, :],
				                     0, 5, 0, 0, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
		phrase_context_imgs.append(img)
	# print('Len of line boxes', len(phrase_context_imgs))
	# print(f'---------->For Pos: {pid}, found {len(phrase_context_imgs)} phrases')
	nimg = np.concatenate(phrase_context_imgs)
	nimg = Image.fromarray(nimg).convert('RGB')
	response = HttpResponse(content_type="image/png")
	nimg.save(response, "PNG")
	return response


def demo_results(request, img_id):
	cname = request.session['cname']
	Cname = cname.replace('_', ' ')
	collection = Collection.objects.filter(collection_name = Cname)[0]
	demo_path = collection.demo_path
	img_path =  f'{settings.STATIC_PATH}/{demo_path}/{img_id}.jpg'
	fb = open(img_path, 'rb')
	f = fb.read()
	b = bytearray(f)
	img = cv2.imdecode(np.asarray(b), 1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# im_b64 = base64.b64encode(b).decode("utf8")
	# results, positions = search_img(im_b64, request.session['api_cname'])
	# with open('static/files/'+cname+'/results/'+img_id+'.p', 'wb') as f:
	# 	pickle.dump([results, positions], f)
	
	with open('static/files/'+cname+'/results/'+img_id+'.p', 'rb') as f:
	 	results, positions = pickle.load(f)
	request.session['results'] = results
	request.session['positions'] = positions
	request.session['cname'] = cname
	request.session['ftype'] = 'img'
	request.session['qimg'] = img.tolist()
	return redirect('results')


def query(request, cname):
	page_template =  'search/search.html'
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	collection = Collection.objects.filter(collection_name = Cname)[0]
	context = {}
	context['desc'] = collection.desc
	context['url'] = collection.collection_link
	context['lang'] = collection.language
	context['demo_imgs'] = os.listdir(settings.STATIC_PATH +"/files/"+cname+"/imgs/")
	context['cname'] = cname
	context['Cname'] = Cname
	request.session['api_cname'] = collection.api_cname

	if request.method == 'POST':
		form1 = SearchForm(request.POST, request.FILES)
		if form1.is_valid():
			if 'imquery' in request.FILES:
				fobj = request.FILES['imquery'].read()
				jpeg_array = bytearray(fobj)
				img = cv2.imdecode(np.asarray(jpeg_array), 1)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				im_b64 = base64.b64encode(fobj).decode("utf8")
				results, positions = search_img(im_b64, request.session['api_cname'])
				request.session['results'] = results
				request.session['positions'] = positions
				request.session['cname'] = cname
				request.session['ftype'] = 'img'
				request.session['qimg'] = img.tolist()
				return redirect('results')
			elif form1.cleaned_data['txtquery']:
				results, positions = search_txt(form1.cleaned_data['txtquery'], request.session['api_cname'])
				# print('--------->query results: ', results)
				request.session['results'] = results
				request.session['positions'] = positions
				request.session['cname'] = cname
				request.session['ftype'] = 'txt'
				request.session['qimg'] = form1.cleaned_data['txtquery']
				return redirect('results')
		else:
			print(form1.errors)

	form1 = SearchForm()
	context['form1'] = form1
	collections = Collection.objects.all().values_list('collection_name', flat=True)
	context['collections'] = [(each.replace(' ', '_'), each)for each in collections]
	return render(request, page_template, context)


def results(request):
	page_template = 'search/results.html'
	results = request.session['results']
	cname = request.session['cname']
	context = {}
	pages = []
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	context['Cname'] = Cname
	collection = Collection.objects.filter(collection_name = Cname)[0]
	context['word_path'] = collection.words_path
	context['lang'] = collection.language

	if request.method == 'POST':
		form1 = SearchForm(request.POST, request.FILES)
		if form1.is_valid():
			if 'imquery' in request.FILES:
				fobj = request.FILES['imquery'].read()
				jpeg_array = bytearray(fobj)
				img = cv2.imdecode(np.asarray(jpeg_array), 1)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				im_b64 = base64.b64encode(fobj).decode("utf8")
				results, positions = search_img(im_b64, request.session['api_cname'])
				request.session['results'] = results
				request.session['positions'] = positions
				request.session['cname'] = cname
				request.session['ftype'] = 'img'
				request.session['qimg'] = img.tolist()
				return redirect('results')
			elif form1.cleaned_data['txtquery']:
				results, positions = search_txt(form1.cleaned_data['txtquery'], request.session['api_cname'])
				request.session['results'] = results
				request.session['positions'] = positions
				print('----------> Received: ', positions)
				request.session['cname'] = cname
				request.session['ftype'] = 'txt'
				request.session['qimg'] = form1.cleaned_data['txtquery']
				return redirect('results')
		else:
			print(form1.errors)

	form1 = SearchForm()
	context['form1'] = form1

	if len(results)==0:
		context['nflag'] = 1
	else:
		context['nflag'] = 0
	for i, each in enumerate(results):
		pages.append([each+'.jpg', results[each], i])

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

def line_results(request):
	page_template = 'search/line_results.html'
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
	context['lang'] = collection.language


	if request.method == 'POST':
		form1 = SearchForm(request.POST, request.FILES)
		if form1.is_valid():
			if 'imquery' in request.FILES:
				fobj = request.FILES['imquery'].read()
				jpeg_array = bytearray(fobj)
				img = cv2.imdecode(np.asarray(jpeg_array), 1)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				im_b64 = base64.b64encode(fobj).decode("utf8")
				results, positions = search_img(im_b64, request.session['api_cname'])
				request.session['results'] = results
				request.session['positions'] = positions
				request.session['cname'] = cname
				request.session['ftype'] = 'img'
				request.session['qimg'] = img.tolist()
				return redirect('results')
			elif form1.cleaned_data['txtquery']:
				results, positions = search_txt(form1.cleaned_data['txtquery'], request.session['api_cname'])
				request.session['results'] = results
				request.session['positions'] = positions
				# print('---------->Line result Post Received: ', results)
				# print('---------->Line result Post Received: ', positions)
				request.session['cname'] = cname
				request.session['ftype'] = 'txt'
				request.session['qimg'] = form1.cleaned_data['txtquery']
				return redirect('results')

		else:
			print(form1.errors)

	form1 = SearchForm()
	context['form1'] = form1

	if len(results)==0:
		context['nflag'] = 1
	else:
		context['nflag'] = 0

	print('-----> Line results: ', len(results))
	for i, each in enumerate(results):
		# pages.append([each[0].split('/')[0]+'.jpg', each[1][1], i])
		pages.append([each, 'unk', i])

	paginator = Paginator(pages, 3)

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

def view_results(request, page, pid):
	page_template = 'search/view_results.html'

	pid = int(pid)
	page = int(page)
	results = request.session['results']
	cname = request.session['cname']
	context = {}
	request.session['cname'] = cname
	Cname = cname.replace('_', ' ')
	context['Cname'] = Cname
	collection = Collection.objects.filter(collection_name = Cname)[0]
	context['lang'] = collection.language
	if request.method == 'POST':
		form1 = SearchForm(request.POST, request.FILES)
		if form1.is_valid():
			if 'imquery' in request.FILES:
				fobj = request.FILES['imquery'].read()
				jpeg_array = bytearray(fobj)
				img = cv2.imdecode(np.asarray(jpeg_array), 1)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				im_b64 = base64.b64encode(fobj).decode("utf8")
				results, positions = search_img(im_b64, request.session['api_cname'])
				request.session['results'] = results
				request.session['positions'] = positions
				request.session['cname'] = cname
				request.session['ftype'] = 'img'
				request.session['qimg'] = img.tolist()
				return redirect('results')
			elif form1.cleaned_data['txtquery']:
				results, positions = search_txt(form1.cleaned_data['txtquery'], request.session['api_cname'])
				request.session['results'] = results
				request.session['positions'] = positions
				request.session['cname'] = cname
				request.session['ftype'] = 'txt'
				request.session['qimg'] = form1.cleaned_data['txtquery']
				return redirect('results')
		else:
			print(form1.errors)

	form1 = SearchForm()
	context['form1'] = form1

	if page!=0:
		pid = (page-1)*3 + pid
	path = results[pid]
	request.session['rind'] = pid
	# pos = results[path][1]
	# print('--------->Path from APi', path)
	nimg_path =  settings.STATIC_PATH +"/files/"+cname+"/uploads/"+path+'.jpg'
	# print('--------Updated path for loading', nimg_path)
	if request.session['ftype'] == 'img':
		context['qimg'] = reverse('show_image')
	else:
		context['qimg'] = request.session['qimg']
	request.session['path'] = nimg_path
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
