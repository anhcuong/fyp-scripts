import os


cur_dir = os.getcwd()
ffmpeg = os.getenv('FFMPEG_LOCATION', 'C:/Users/Frank/workspaces/openpose/ffmpeg/bin/ffmpeg.exe')
openpose = os.getenv('OPENPOSE', 'C:/Users/Frank/workspaces/openpose/bin/OpenPoseDemo.exe')
rtsp_url = os.getenv('RTSP_URL', 'rtsp://192.168.1.7')
raw_video_folder = os.getenv('RAW_VIDEO_FOLDER', os.path.join(cur_dir, 'raw_videos'))
raw_frame_folder = os.getenv('RAW_FRAME_FOLDER', os.path.join(cur_dir, 'raw_frames'))
openpose_processing_folder = os.getenv('OPENPOSE_PROCESSING_FOLDER', os.path.join(cur_dir, 'op_processing')))
heatmap_folder = os.getenv('HEATMAP_FOLDER', os.path.join(cur_dir, 'heatmaps'))
keypoint_folder = os.getenv('KEYPOINT_FOLDER', os.path.join(cur_dir, 'keypoints'))
log_folder = os.getenv('LOG_FOLDER', os.path.join(cur_dir, 'logs'))
