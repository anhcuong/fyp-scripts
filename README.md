# Final Year Project Repository

# Install packages

- Install virtualenv

```bash
python3 -m .venv
# Windows
.venv/Scripts/activate.bat
pip install -r requirements.txt
```

# GUI

```bash
cd web/django_project/
python manage.py runserver
```

# Backend

- Download [pretrained model](https://drive.google.com/file/d/1ozef9nkD70hV90-KU2RfGU4LlUtAj9Y4/view?usp=sharing) into models/ folder and config the CNN_MODEL_LOCATION/

- Update configuration path (Important!!!). Please refer to `config.py` for list of configurations.

```bash
export FRONTEND_BASE_DIR="C:\Users\Frank\workspaces\fyp-scripts\web\django_project\fyp"
export OPENPOSE=<Path to OpenPose>
export FFMPEG=<Path to FFMPEG>
export RAW_VIDEO_FOLDER=<Path to Raw Video Folder>
export CNN_MODEL_LOCATION=<Path to CNN MODEL>
```

- Run redis server

```bash
docker run -d -p 6379:6379 redis
```

- Run batch pipeline

```bash
cd pipeline/
# Run in 1st terminal
python cli.py process_images
# Run in 2nd terminal
python cli.py predict_crowding
# Run in 3rd terminal
python cli.py run_batch_pipeline
```

- Run real-time stream

Open Live Reporter App in Android or Ios

Update config.py with rtsp stream showed on the App


```bash
cd pipeline/
# Run in 1st terminal
python cli.py copy_raw_processings
# Run in 2nd terminal
python cli.py generate_heatmap_with_keypoints_realtime
# Run in 3rd terminal: Stack images
python cli.py process_imagess
# Run in 4th terminal: Predict fighting and falling
python cli.py predict_fighting_falling_realtime
# Run in 5th terminal: Predict crowding
python cli.py predict_crowding
# RUN THIS LAST in 6th terminal!!!
python cli.py extract_frames_from_stream
```

- Reset workspace

```bash
cd pipeline/
python cli.py reset
```
