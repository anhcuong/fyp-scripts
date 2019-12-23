import subprocess

from config import (
    openpose, ffmpeg, raw_frame_folder, rtsp_url, openpose_processing_folder,
    heatmap_folder, keypoint_folder, log_folder)


def run_command(cmd):
    print('Running command: {cmd}'.format(cmd=cmd))
    subprocess.Popen(cmd.split())


def extract_frames_from_stream():
    print('Extract frames from stream')
    cmd = '{ffmpeg} -i {stream_url} -vf fps=1 {raw_folder}/raw_%04d.png -hide_banner'.format(
        ffmpeg=ffmpeg,
        raw_folder=raw_frame_folder,
        stream_url=rtsp_url
    )
    run_command(cmd)


def generate_heatmap_with_openpose():
    print('Generate heatmap with openpose')
    cmd = '{openpose} --image_dir {image_dir} --write_images {heatmap_dir} --disable_blending True --net_resolution "-1x480" --alpha_pose 1 --alpha_heatmap 1 --part_to_show 1 -log_dir {log_location} -display 0'.format(
        openpose=openpose,
        image_dir=openpose_processing_folder,
        log_location=log_location,
        heatmap_dir=heatmap_folder,
    )
    run_command(cmd)


def generate_keypoints_with_openpose():
    print('Generate keypoints with openpose')
    cmd = ''
    run_command(cmd)cmd = '{openpose} --image_dir {image_dir} --write_json {keypoint_dir} --disable_blending True --net_resolution "-1x480" -log_dir {log_location} -display 0'.format(
        openpose=openpose,
        image_dir=openpose_processing_folder,
        log_location=log_location,
        keypoint_dir=keypoint_folder,
    )
    run_command(cmd)
