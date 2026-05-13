"""
URL configuration for DeepByte project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin  # Type ignored for linting
from django.urls import path  
from DeepByte1.views import (  # ✅ Import all required views correctly
    
 detect_fake_audio, analyzeAudio, analyzeVideo, viewBS, history, help, about, contactus, index, file, upload_exe, list_exe_files)
from DeepByte1.views1 import analyzeDF
from django.conf import settings
from django.conf.urls.static import static




urlpatterns = [
  
    
    path('upload_exe/', upload_exe, name='upload_exe'),
    path('list_exe_files/', list_exe_files, name='list_exe_files'),
    path('detect_fake_audio/', detect_fake_audio, name='detect_fake_audio'),
    path("analyzeAudio/", analyzeAudio, name="analyzeAudio"),
    path("analyzeVideo/", analyzeVideo, name="analyzeVideo"),
    path("", index, name="home"),  
    path("file/", file, name="file"),
    path("analyzeDF/", analyzeDF, name="analyzeDF"),
    path("viewBS/", viewBS, name="viewBS"),
    path("history/", history, name="history"),
    path("help/", help, name="help"),
    path("about/", about, name="about"),
    path("contactus/", contactus, name="contactus"),
]



urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)