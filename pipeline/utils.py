import glob
import os
import subprocess
from pathlib import Path
import time

import cv2
from moviepy.editor import VideoFileClip

import requests

from config import (
    openpose, ffmpeg, raw_frame_folder, rtsp_url, openpose_processing_folder,
    heatmap_folder, keypoint_folder, log_folder, fps, frame_prefix,
    video_extension, raw_video_folder, subclip_video_folder,
    output_type, frame_width, frame_height, subclip_duration, openpose_model_folder, stack_frame_folder,
    display_alert_url, display_frame_url, frontend_base_dir, rd, display_crowd_url)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Time taken: {} seconds'.format(te-ts))
        return result
    return timed


def run_command(cmd):
    print('Running command: {cmd}'.format(cmd=cmd))
    subprocess.call(cmd, shell=True)


def extract_frames_from_stream():
    print('Extract frames from stream')
    cmd = '{ffmpeg} -i {stream_url} -vf fps={fps} {raw_folder}/{frame_prefix}%d.png -hide_banner'.format(
        ffmpeg=ffmpeg,
        raw_folder=raw_frame_folder,
        stream_url=rtsp_url,
        fps=fps,
        frame_prefix=frame_prefix,
    )
    run_command(cmd)


@timeit
def generate_heatmap_with_openpose(image_dir, with_keypoints=False):
    print('Generate heatmap with openpose')
    if with_keypoints:
        cmd = '{openpose} --image_dir {image_dir} --model_folder {openpose_model_dir} --write_json {keypoint_dir} --write_images {heatmap_dir} --disable_blending True --net_resolution "-1x480" --alpha_pose 1 --alpha_heatmap 1 --part_to_show 1 -log_dir {log_location} -display 0'.format(
            image_dir=image_dir,
            openpose=openpose,
            log_location=log_folder,
            heatmap_dir=heatmap_folder,
            keypoint_dir=keypoint_folder,
            openpose_model_dir=openpose_model_folder,
        )
    else:
        cmd = '{openpose} --image_dir {image_dir} --model_folder {openpose_model_dir} --write_images {heatmap_dir} --disable_blending True --net_resolution "-1x480" --alpha_pose 1 --alpha_heatmap 1 --part_to_show 1 -log_dir {log_location} -display 0'.format(
            image_dir=image_dir,
            openpose=openpose,
            log_location=log_folder,
            heatmap_dir=heatmap_folder,
            openpose_model_dir=openpose_model_folder,
        )
    run_command(cmd)


@timeit
def generate_keypoints_with_openpose():
    print('Generate keypoints with openpose')
    cmd = '{openpose} --image_dir {image_dir} --model_folder {openpose_model_dir} --write_json {keypoint_dir} --disable_blending True --net_resolution "-1x480" -log_dir {log_location} -display 0'.format(
        openpose=openpose,
        image_dir=openpose_processing_folder,
        log_location=log_folder,
        keypoint_dir=keypoint_folder,
        openpose_model_dir=openpose_model_folder,
    )
    run_command(cmd)


def list_files_in_folder(dir_path, extension):
    rs = []
    for r, d, files in os.walk(dir_path):
        for f in files:
            for ext in extension:
                if ext in f:
                    rs.append(os.path.join(r, f))
    return rs


@timeit
def raw_video_to_subclip(debug=False):
    print('Split raw video into subclips')
    videos = list_files_in_folder(raw_video_folder, video_extension)
    for video in videos:
        os.rename(video, video.replace(' ', '-'))
    videos = list_files_in_folder(raw_video_folder, video_extension)

    if debug:
        return videos

    def subclip(f_path):
        full_video = VideoFileClip(f_path)
        duration = full_video.duration
        print(duration)
        clip_start = 0
        idx = 0
        video_name, _ = Path(f_path).name.split('.')
        clips = []
        while clip_start < duration:
            clip_end = clip_start + subclip_duration
            if clip_end > duration:
                clip_end = duration
            clip = full_video.subclip(clip_start, clip_end)
            name = "{}/{}-{}.mp4".format(subclip_video_folder, video_name, idx)
            try:
                clip.write_videofile(name, codec="libx264", temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac')
            except Exception as e:
                print(e)
                clip_start = clip_end
                continue
            idx += 1
            clip_start = clip_end
            clips.append(name)
        print(clips)
        return clips
    clips = []
    for v in videos:
        clips += subclip(v)
    return clips


@timeit
def subclip_to_frame():
    print('Export subclips into frames')
    frame_rate = float(1.0/float(fps))
    videos = list_files_in_folder(subclip_video_folder, video_extension)

    def video_to_frame(f_path):
        vidcap = cv2.VideoCapture(f_path)
        video_name, _ = Path(f_path).name.split('.')
        def getFrame(sec):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
            hasFrames, image = vidcap.read()
            if hasFrames:
                # save frame as PNG file
                f_name = "{}/{}-frame-{}.png".format(raw_frame_folder, video_name, count)
                cv2.imwrite(f_name, image)
            return hasFrames
        sec = 0
        count=1
        success = getFrame(sec)
        while success:
            count = count + 1
            sec = sec + frame_rate
            sec = round(sec, 2)
            success = getFrame(sec)

    for f_path in videos:
        video_to_frame(f_path)


def pick_frames_for_prediction(videos):
    frames = []
    for video_path in videos:
        start_frame = 1
        continue_loop = True
        frame_prefix = video_path.replace(subclip_video_folder, stack_frame_folder).replace(raw_video_folder, stack_frame_folder).replace('.mp4', '').replace('.avi', '')
        frame_path = "{}-frame-{}.png"
        frame = [frame_path.format(frame_prefix, start_frame)]
        while continue_loop:
            last_frame = start_frame + fps - 1
            f_path = frame_path.format(frame_prefix, last_frame)
            if os.path.exists(f_path):
                frame.append(f_path)
            else:
                continue_loop = False
            start_frame = last_frame + 1
        frames.append(frame)
    return frames


def display_alert(frames, event_type):
    rd.set(event_type + '_detected', '1', ex=5)
    data = {'eventType': event_type}
    for idx, frame in enumerate(frames):
        suffix = idx + 1
        raw_url = frame.replace(subclip_video_folder, raw_frame_folder).replace(
            stack_frame_folder, raw_frame_folder).replace(frontend_base_dir, '').replace('static/', '')
        data['snapshotURL{}'.format(suffix)] = raw_url
    r = requests.post(display_alert_url, json=data)
    print(r.status_code, r.text)


def display_frame(frame, prediction_result):
    # use relative path
    raw_url = frame.replace(subclip_video_folder, raw_frame_folder).replace(
        stack_frame_folder, raw_frame_folder).replace(frontend_base_dir, '')
    frame_url = frame.replace(subclip_video_folder, stack_frame_folder).replace(frontend_base_dir, '')
    fighting, falling = prediction_result
    data = {
        'snapshotRawURL': raw_url,
        'snapshotHeatURL': frame_url,
        'fallingAccuracy': float(falling),
        'fightingAccuracy': float(fighting)
    }
    r = requests.post(display_frame_url, json=data)
    print(r.status_code, r.text)


def display_crowd(path, crowd_flag):
    # use relative path
    path = path.replace(frontend_base_dir, '')
    print(path)
    data = {
        'snapshotURL': path,
        'crowdFlag': crowd_flag
    }
    r = requests.post(display_crowd_url, json=data)
    print(r.status_code, r.text)
