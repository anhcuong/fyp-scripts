import os

import redis


cur_dir = os.getcwd()
# FFPMEG
ffmpeg = os.getenv('FFMPEG_LOCATION', 'C:/Users/Frank/workspaces/openpose/ffmpeg/bin/ffmpeg.exe')
fps = int(os.getenv('FPS', 1))
sleep_time = int(os.getenv('SLEEP_TIME', 1))
rtsp_url = os.getenv('RTSP_URL', 'rtsp://192.168.1.10')
# Openpose
openpose = os.getenv('OPENPOSE', 'C:/Users/Frank/workspaces/openpose/bin/OpenPoseDemo.exe')
openpose_model_folder = os.getenv('OPENPOSE_MODEL_FOLDER', 'C:/Users/Frank/workspaces/openpose/models/')

# CNN
cnn_model_location = os.getenv('CNN_MODEL_LOCATION', os.path.join(cur_dir, 'models', 'hmnn_full_best_weights.hdf5'))
cnn_frames_per_prediction = 5

# Crowding
crowding_frames_per_prediction = 1

# Input/Output folder
raw_video_folder = os.getenv('RAW_VIDEO_FOLDER', os.path.join(cur_dir, 'raw_videos'))
subclip_video_folder = os.getenv('SUBCLIP_VIDEO_FOLDER', os.path.join(cur_dir, 'subclip_videos'))
raw_frame_folder = os.getenv('RAW_FRAME_FOLDER', os.path.join(cur_dir, 'raw_frames'))
openpose_processing_folder = os.getenv('OPENPOSE_PROCESSING_FOLDER', os.path.join(cur_dir, 'op_processing'))
heatmap_folder = os.getenv('HEATMAP_FOLDER', os.path.join(cur_dir, 'heatmaps'))
keypoint_folder = os.getenv('KEYPOINT_FOLDER', os.path.join(cur_dir, 'keypoints'))
log_folder = os.getenv('LOG_FOLDER', os.path.join(cur_dir, 'logs'))
frame_prefix = 'raw_'

# redis config
redis_url = os.getenv('REDIS_HOST', '127.0.0.1')
try:
    rd = redis.Redis(host=redis_url, port=6379, db=0)
except:
    print('WARNING!!! Redis has not been setup on local')
    rd = None

# opencv config
video_extension = ['mp4', 'avi']
output_type = '.mp4'
frame_width = 640
frame_height = 480
subclip_duration = 5


for folder in [subclip_video_folder, raw_frame_folder, openpose_processing_folder, heatmap_folder, keypoint_folder, log_folder, cnn_model_location]:
    if not os.path.exists(folder):
        os.mkdir(folder)
