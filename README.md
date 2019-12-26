# Final Year Project Repository

- Install virtualenv

```bash
python3 -m .venv
# Windows
.venv/Scripts/activate.bat
pip install -r requirements.txt
```

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
python cli.py run_batch_pipeline
```

- Run real-time stream


```bash
cd pipeline/

python cli.py copy_raw_processings
python cli.py extract_frames_from_stream
python cli.py generate_heatmap_realtime
python cli.py predict_realtime
```

- Reset workspace


```bash
cd pipeline/

python cli.py reset
```
