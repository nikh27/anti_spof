from django.urls import path
from .views import *

urlpatterns = [
    path('',home, name='home'),
    path('/main',main, name='main'),
    path('start/',start, name='start'),
    path('add/', add, name='add'),
]
