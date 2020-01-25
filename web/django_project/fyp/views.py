from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpRequest
from django.http import HttpResponseRedirect

from .models import Alert

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

import sys
import requests as rq
import json
from django_eventstream import send_event



def home(request):
	return render(request, 'fyp/index.html')


def alertHistory(request):
	context = {
		'alerts':Alert.objects.all().order_by('-id')
	}
	return render(request, 'fyp/history.html', context)

#Save Alert to DB
@api_view(['POST'])
def saveAlert(request):
	alert = Alert(snapshotURL1=request.data["snapshotURL1"],
		snapshotURL2=request.data["snapshotURL2"],
		snapshotURL3=request.data["snapshotURL3"],
		snapshotURL4=request.data["snapshotURL4"],
		snapshotURL5=request.data["snapshotURL5"],
		eventType=request.data["eventType"],
		handled=request.data["handled"])
	try:
		alert.save()
		return Response("Saved",  status=status.HTTP_200_OK)
	except:
		return Response("Error",  status=status.HTTP_400_BAD_REQUEST)

#Pop up when there is event detected
@api_view(['POST'])
def displayAlert(request):
	context = {
		'snapshotURL1':request.data["snapshotURL1"],
		'snapshotURL2':request.data["snapshotURL2"],
		'snapshotURL3':request.data["snapshotURL3"],
		'snapshotURL4':request.data["snapshotURL4"],
		'snapshotURL5':request.data["snapshotURL5"],
		'eventType':request.data["eventType"]
	}

	try:
		print("Inside TRY in displayAlert!!!!!!!!!!!!!!!!!!!")
		send_event('test', 'message', context)

		return Response(context,  status=status.HTTP_200_OK)
	except:
		return Response(context,  status=status.HTTP_400_BAD_REQUEST)

#Display latest RAW/Heatmap image
@api_view(['POST'])
def displayFrame(request):
	context = {
		'snapshotRawURL':request.data["snapshotRawURL"],
		'snapshotHeatURL':request.data["snapshotHeatURL"],
		'fallingAccuracy': request.data["fallingAccuracy"],
		'fightingAccuracy': request.data["fightingAccuracy"]
	}

	try:
		print("Inside TRY in displayFrame!!!!!!!!!!!!!!!!!!!")
		send_event('frame', 'message', context)

		return Response(context,  status=status.HTTP_200_OK)
	except:
		return Response(context,  status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def displayCrowdCount(request):
	context = {
		'snapshotURL':request.data["snapshotURL"]
	}

	try:
		print("Inside TRY in displayCrowdCount!!!!!!!!!!!!!!!!!!!")
		send_event('crowd', 'message', context)

		return Response(context,  status=status.HTTP_200_OK)
	except:
		return Response(context,  status=status.HTTP_400_BAD_REQUEST)



