from django.db import models
from django.utils import timezone

class Alert(models.Model):
	snapshotURL1 = models.CharField(max_length=100)
	snapshotURL2= models.CharField(max_length=100)
	snapshotURL3 = models.CharField(max_length=100)
	snapshotURL4= models.CharField(max_length=100)
	snapshotURL5 = models.CharField(max_length=100)
	eventType = models.CharField(max_length=100)
	timestamp = models.DateTimeField(default=timezone.now)
	handled = models.CharField(max_length=100)