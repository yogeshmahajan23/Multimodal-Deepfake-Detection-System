from django.db import models
from django.shortcuts import render




class EXEFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='exe_files/')  # ✅ This must be 'file'
    uploaded_at = models.DateTimeField(auto_now_add=True)



    
class Article(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    published_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Contact(models.Model):
    name = models.CharField(max_length=122)
    email = models.CharField(max_length=122)
    phone = models.CharField(max_length=12)
    desc = models.TextField()
    date = models.DateField()
    
    def __str__(self):
        return self.name