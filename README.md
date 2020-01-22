# Final Year Project Repository

- Install virtualenv

```bash
python3 -m .venv
# Windows
.venv/Scripts/activate.bat
pip install -r requirements.txt
```

- Download [pretrained model](https://drive.google.com/file/d/1ozef9nkD70hV90-KU2RfGU4LlUtAj9Y4/view?usp=sharing) into models/ folder and config the CNN_MODEL_LOCATION/

- Update configuration path (Important!!!). Please refer to `config.py` for list of configurations.

```bash
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
python cli.py process_images
python cli.py run_batch_pipeline
```

- Run real-time stream


```bash
cd pipeline/

python cli.py copy_raw_processings
python cli.py extract_frames_from_stream
python cli.py generate_heatmap_realtime
# Or with keypoints
python cli.py generate_heatmap_with_keypoints_realtime
# Stack images
python cli.py process_imagess
# Predict fighting and falling
python cli.py predict_fighting_falling_realtime
# Predict crowding
python cli.py predict_crowding_realtime
```

- Reset workspace

```bash
cd pipeline/

python cli.py reset
```
