# Generated by Django 3.0.2 on 2020-01-16 15:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fyp', '0003_auto_20200116_2329'),
    ]

    operations = [
        migrations.AlterField(
            model_name='alert',
            name='timestamp',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
