from django.urls import path

from . import views


# use re_path for using it along with regular expressions
urlpatterns = [
	path('', views.index, name='index'),
	path('query', views.upload_query, name='upload_query'),
	path('results', views.results, name='results')
	]