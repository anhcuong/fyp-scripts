# Generated by Django 3.0.2 on 2020-01-16 17:00

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('fyp', '0005_auto_20200117_0011'),
    ]

    operations = [
        migrations.AlterField(
            model_name='alert',
            name='timestamp',
            field=models.DateTimeField(default=datetime.datetime(2020, 1, 16, 17, 0, 39, 172131, tzinfo=utc)),
        ),
    ]