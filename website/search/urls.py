from django.urls import path, re_path

from . import views

urlpatterns = [
	path('', views.index, name='home'),
	path('results', views.line_results, name='results'),
	path('show_image', views.show_image, name='show_image'),
	path('show_page', views.show_page, name='show_page'),
	path('project', views.about_project, name='about_project'),
	path('detail', views.detail, name='detail'),
	re_path(r'^show_line/(?P<pid>\d)/$', views.show_line, name='show_line'),
	re_path(r'^search/(?P<cname>\w+)/$', views.query, name='search'),
	re_path(r'^chome/(?P<cname>\w+)/$', views.collection_index, name='chome'),
	re_path(r'^mresults/(?P<page>\d)/$', views.mresults, name='mresults'),
	re_path(r'^view_results/(?P<page>\d)/(?P<pid>\d)/$', views.view_results, name='view_results'),
	re_path(r'^dresults/(?P<img_id>\w+)/$', views.demo_results, name='dresults'),
	re_path(r'^redirect_/(?P<cname>\w+)/(?P<choice>\d)/$', views.redirect_query, name='redirect_'),
	]