from django.urls import path, re_path

from . import views


# use re_path for using it along with regular expressions
urlpatterns = [
	path('', views.index, name='index'),
	path('query', views.upload_query, name='upload_query'),
	path('results', views.results, name='results'),
	path('show_image', views.show_image, name='show_image'),
	#re_path(r'^show_page/(?P<pid>\d)', views.show_page, name='show_page'),
	path('show_page', views.show_page, name='show_page'),
	re_path(r'^view_results/(?P<pid>\d)', views.view_results, name='view_results'),
	
	]