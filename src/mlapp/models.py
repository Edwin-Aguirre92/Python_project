from django.db import models

# Create your models here.
class Mlapp(models.Model):
    title = models.TextField(default="")
    description = models.TextField(default="")

