# Generated by Django 2.0.7 on 2018-07-31 10:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('retrieval', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='collections',
            name='collection_link',
            field=models.URLField(blank=True),
        ),
        migrations.AlterField(
            model_name='collections',
            name='desc',
            field=models.TextField(blank=True),
        ),
    ]
