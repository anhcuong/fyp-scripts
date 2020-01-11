import os
import time
import sys
import shutil
import glob

from config import (
    raw_frame_folder, openpose_processing_folder, fps,
    sleep_time, frame_prefix, rd, subclip_video_folder, heatmap_folder,
    cnn_frames_per_prediction, keypoint_folder)
from utils import (
    extract_frames_from_stream, generate_heatmap_with_openpose,
    generate_keypoints_with_openpose, raw_video_to_subclip, subclip_to_frame,
    pick_frames_for_prediction,)
from models import run_cnn_model, get_cnn_model, run_crowd_detection_model


FIGHTING_FALLING_FRAME_RS_PREFIX = 'fighting_falling_'
CROWD_FRAME_RS_PREFIX = 'crowd_'


def build_path(folder, idx):
    return os.path.join(folder, '{}{}.png'.format(frame_prefix, idx))


def is_op_in_progress():
    val = rd.get('OPENPOSE_IN_PROGRESS')
    return int(val) == 1


def is_dir_empty(dir_path):
    return len(os.listdir(dir_path) ) == 0


def empty_folder(dir_path):
    try:
        files = glob.glob('{}/*'.format(dir_path))
        for f in files:
            os.remove(f)
    except:
        print('{} does not exist'.format(dir_path))


def copy_raw_to_processing():
    print('Copy data to processing folder')
    start_frame = 1
    while True:
        last_frame = start_frame + fps
        if is_op_in_progress():
            print('Wait...')
        else:
            print('On...')
        if not is_op_in_progress() and os.path.isfile(build_path(raw_frame_folder, last_frame-1)):
            for i in range(start_frame, last_frame):
                shutil.copy(build_path(raw_frame_folder, i), build_path(openpose_processing_folder, i))
            start_frame = last_frame
        time.sleep(sleep_time)


def generate_heatmap_realtime(with_keypoint=False):
    while True:
        if not is_dir_empty(openpose_processing_folder):
            rd.set('OPENPOSE_IN_PROGRESS', '1')
            generate_heatmap_with_openpose(openpose_processing_folder, with_keypoint)
            empty_folder(openpose_processing_folder)
            rd.set('OPENPOSE_IN_PROGRESS', '0')
        time.sleep(sleep_time)


def generate_keypoints_realtime():
    while True:
        if not is_dir_empty(openpose_processing_folder):
            rd.set('OPENPOSE_IN_PROGRESS', '1')
            generate_keypoints_with_openpose(openpose_processing_folder)
            empty_folder(openpose_processing_folder)
            rd.set('OPENPOSE_IN_PROGRESS', '0')
        time.sleep(sleep_time)


def generate_heatmap_batch():
    generate_heatmap_with_openpose(raw_frame_folder)


def predict_in_batch_with_cnn(videos):
    model = get_cnn_model()
    x = pick_frames_for_prediction(videos)
    for video_frames in x:
        total_frames = len(video_frames)
        for start in range(total_frames):
            end = start + cnn_frames_per_prediction
            input_frames = video_frames[start:end]
            total_input_frames = len(input_frames)
            if total_input_frames < cnn_frames_per_prediction:
                input_frames = input_frames + [input_frames[-1]] * (cnn_frames_per_prediction-total_input_frames)
            rs = run_cnn_model(input_frames, model)
            print('Result for {}: {}'.format(input_frames, rs))


def run_batch_pipeline():
    videos = raw_video_to_subclip()
    subclip_to_frame()
    generate_heatmap_batch()
    predict_in_batch_with_cnn(videos)


def update_result(prefix, frame, result):
    rd.set('{}_{}'.format(prefix, frame), str(result))


def predict_fighting_falling_realtime():
    model = get_cnn_model()
    start_frame = 1
    f_name = 'raw_{}_rendered.png'
    while True:
        end_frame = start_frame + cnn_frames_per_prediction*fps
        input_frames = []
        for i in range(start_frame, end_frame, fps):
            input_frames.append(os.path.join(heatmap_folder, f_name.format(i)))
        while(not os.path.isfile(input_frames[-1])):
            time.sleep(sleep_time)
        rs = run_cnn_model(input_frames, model)
        update_result(FIGHTING_FALLING_FRAME_RS_PREFIX, start_frame, rs)
        start_frame = end_frame


def predict_crowding_realtime():
    start_frame = 1
    f_name = 'raw_{}_rendered.json'
    while True:
        end_frame = start_frame + crowding_frames_per_prediction*fps
        for i in range(start_frame, end_frame, fps):
            f = os.path.join(keypoint_folder, f_name.format(i))
            while(not os.path.isfile(f)):
                time.sleep(sleep_time)
            rs = run_crowd_detection_model(f)
            update_result(CROWD_FRAME_RS_PREFIX, i, rs)
        start_frame = end_frame


def reset():
    empty_folder(raw_frame_folder)
    empty_folder(heatmap_folder)
    empty_folder(keypoint_folder)
    empty_folder(subclip_video_folder)


if __name__== "__main__":
    arg = sys.argv[1]
    if arg == 'copy_raw_processings':
        copy_raw_to_processing()
    elif arg == 'extract_frames_from_stream':
        extract_frames_from_stream()
    elif arg == 'generate_heatmap_realtime':
        generate_heatmap_realtime()
    elif arg == 'generate_heatmap_with_keypoints_realtime':
        generate_heatmap_realtime(with_keypoint=True)
    elif arg == 'predict_fighting_falling_realtime':
        predict_fighting_falling_realtime()
    elif arg == 'predict_crowding_realtime':
        predict_crowding_realtime()
    elif arg == 'run_batch_pipeline':
        run_batch_pipeline()
    elif arg == 'reset':
        reset()
    else:
        pass
