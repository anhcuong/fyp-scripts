# Generated by Django 3.0.2 on 2020-01-16 15:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fyp', '0002_remove_alert_eventid'),
    ]

    operations = [
        migrations.RenameField(
            model_name='alert',
            old_name='snapshotURL',
            new_name='snapshotURL1',
        ),
        migrations.AddField(
            model_name='alert',
            name='snapshotURL2',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='alert',
            name='snapshotURL3',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='alert',
            name='snapshotURL4',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='alert',
            name='snapshotURL5',
            field=models.CharField(default='', max_length=100),
            preserve_default=False,
        ),
    ]