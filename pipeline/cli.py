import os
import time
import sys
import shutil
import glob

from config import (
    raw_frame_folder, openpose_processing_folder, fps,
    sleep_time, frame_prefix, rd, subclip_video_folder, heatmap_folder,
    cnn_frames_per_prediction, keypoint_folder, stack_frame_folder,
    crowding_frames_per_prediction, prediction_threshold)
from utils import (
    extract_frames_from_stream, generate_heatmap_with_openpose,
    generate_keypoints_with_openpose, raw_video_to_subclip, subclip_to_frame,
    pick_frames_for_prediction, display_frame, display_alert)
from crowd_model import observe_and_process as crowding_observe_and_process
from process_images import observe_and_process as prepare_image_observe_and_process


FIGHTING_FALLING_FRAME_RS_PREFIX = 'fighting_falling_'
CROWD_FRAME_RS_PREFIX = 'crowd_'


def build_path(folder, idx):
    return os.path.join(folder, '{}{}.png'.format(frame_prefix, idx))


def stacking_images_in_progress():
    val = rd.get('STACKING_IN_PROGRESS')
    return val != None and int(val) == 1


def is_the_same_alert(event_type):
    val = rd.get(event_type + '_detected')
    return val != None and int(val) == 1


def is_op_in_progress():
    val = rd.get('OPENPOSE_IN_PROGRESS')
    return val != None and int(val) == 1


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


def generate_heatmap_batch(with_keypoints):
    generate_heatmap_with_openpose(raw_frame_folder, with_keypoints)


def predict_in_batch_with_cnn(videos):
    from cnn_models import run_cnn_model, get_cnn_model
    model = get_cnn_model()
    while(stacking_images_in_progress()):
        print('Waiting for stacking images process')
        time.sleep(sleep_time)
    print('Start prediction!')
    x = pick_frames_for_prediction(videos)
    prediction = {}
    alert = {'fighting': [], 'falling': []}
    for video_frames in x:
        total_frames = len(video_frames)
        for start in range(total_frames):
            end = start + cnn_frames_per_prediction
            input_frames = video_frames[start:end]
            total_input_frames = len(input_frames)
            if total_input_frames < cnn_frames_per_prediction:
                input_frames = input_frames + [input_frames[-1]] * (cnn_frames_per_prediction-total_input_frames)
            rs = run_cnn_model(input_frames, model)
            fighting, falling = rs
            if falling >= prediction_threshold:
                alert['falling'].append(input_frames)
            if fighting >= prediction_threshold:
                alert['fighting'].append(input_frames)
            for frame in input_frames:
                prediction[frame] = rs
    return prediction, alert


def display_prediction_by_frame(predict):
    for k, v in predict.items():
        time.sleep(1)
        display_frame(k, v)


def display_alert_to_ui(alert):
    for frames in alert['fighting']:
        display_alert(frames, 'fighting')
        time.sleep(1)
    for frames in alert['falling']:
        display_alert(frames, 'falling')
        time.sleep(1)


def run_batch_pipeline():
    videos = raw_video_to_subclip()
    subclip_to_frame()
    generate_heatmap_batch(with_keypoints=True)
    predict, alert = predict_in_batch_with_cnn(videos)
    display_prediction_by_frame(predict)
    display_alert_to_ui(alert)


def update_result(prefix, frame, result):
    rd.set('{}_{}'.format(prefix, frame), str(result))


def predict_fighting_falling_realtime():
    from cnn_models import run_cnn_model, get_cnn_model
    model = get_cnn_model()
    start_frame = 1
    f_name = 'raw_{}.png'
    while True:
        end_frame = start_frame + cnn_frames_per_prediction*fps
        input_frames = []
        alert = {'fighting': [], 'falling': []}
        for i in range(start_frame, end_frame, fps):
            input_frames.append(os.path.join(stack_frame_folder, f_name.format(i)))
        while(not os.path.isfile(input_frames[-1])):
            print('Wait for {}'.format(input_frames[-1]))
            time.sleep(sleep_time)
        rs = run_cnn_model(input_frames, model)
        fighting, falling = rs
        if falling >= prediction_threshold:
            alert['falling'].append(input_frames)
        if fighting >= prediction_threshold:
            alert['fighting'].append(input_frames)
        for frame in input_frames:
            display_frame(frame, rs)
        display_alert_to_ui(alert)
        start_frame = end_frame


def run_process_images():
    prepare_image_observe_and_process()


def run_crowding_model():
    crowding_observe_and_process()


def reset():
    empty_folder(raw_frame_folder)
    empty_folder(heatmap_folder)
    empty_folder(keypoint_folder)
    empty_folder(subclip_video_folder)
    empty_folder(stack_frame_folder)
    print('Reset successfully!')


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
    elif arg == 'predict_crowding':
        run_crowding_model()
    elif arg == 'run_batch_pipeline':
        run_batch_pipeline()
    elif arg == 'process_images':
        run_process_images()
    elif arg == 'reset':
        reset()
    else:
        pass
