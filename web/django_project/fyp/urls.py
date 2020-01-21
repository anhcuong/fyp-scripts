from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="FYP-home"),
    path('alertHistory', views.alertHistory, name="Alert History"),
    path('displayAlert', views.displayAlert, name="Display Alert"),
    path('saveAlert', views.saveAlert, name="Alert"),
    path('displayFrame', views.displayFrame, name="Frame"),
    path('displayCrowdCount', views.displayCrowdCount, name="Crowd Count"),
]