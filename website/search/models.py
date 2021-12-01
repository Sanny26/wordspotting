from django.db import models

# Create your models here.

class Collection(models.Model):
	collection_name = models.CharField(max_length=200)
	collection_link = models.URLField(blank=True)
	desc = models.TextField(blank=True)
	language = models.CharField(max_length=200, blank='True')
	api_cname = models.CharField(max_length=200, blank='True')
	words_path = models.CharField(max_length=200, blank='True')
	demo_path = models.CharField(max_length=200, blank='True')

	def __str__(self):
		return self.collection_name
