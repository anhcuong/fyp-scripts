Note

# Deployment.
- Run the following command at path djangoRest\django_project
- Default url for the app is 127.0.0.1:8000

```bash
cd django_project/
python manage.py runserver
```

# Integration
Images needs to be in the following folders before it can be displayed in the UI:
- Raw image foler path: djangoRest\django_project\fyp\static\fyp\img\raw
- heatmap folder path: djangoRest\django_project\fyp\static\fyp\img\heatmap
- crowd graph foler path: djangoRest\django_project\fyp\static\fyp\img\crowd_graph

 3 API calls, refer to the 2 screenshots.
- displayAlert (to be called when there is an event detected)
![displayAlert](displayAlert.png "displayAlert")

- displayFrame (to be called to display raw frame and heatmap)
![displayFrame](displayFrame.png "displayFrame")

- displayCrowdCount (to be called to display the crowd count graph)
![displayCrowdCount](displayCrowdCount.png "displayCrowdCount")

- Other configurable
based.html line 181-183
e.g.
    var raw_frame_format = '.jpg';
    var heatmap_frame_format = '.png';
    var crowd_count_format = '.png';

# Limitations
Event is only saved when it's handled, one event can be displayed at one time. If another event happened before current event is handled, current event will be lost.
