from django.urls import path
from . import views

app_name = 'faceblur'

urlpatterns = [
    path('', views.home, name='home'),
    path('process/', views.process_images, name='process_images'),
    path('match/', views.match_images, name='match_images'),
    path('about/', views.about, name='about'),
]
