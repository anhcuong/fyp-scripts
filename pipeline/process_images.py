import cv2
import numpy as np
import re
import time
import os
from collections import deque

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import (
    raw_frame_folder, openpose_processing_folder, fps,
    sleep_time, frame_prefix, rd, subclip_video_folder, heatmap_folder,
    cnn_frames_per_prediction, keypoint_folder, image_dir, stack_frame_folder)

BG_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=5, detectShadows=False)
IMAGES_DIR = image_dir
RAW_FOLDER = raw_frame_folder
HM_FOLDER = heatmap_folder
KP_FOLDER = keypoint_folder
TARGET_FOLDER = stack_frame_folder


def resize_and_extract_hhb(raw_path, hm_path, output_path):

    """
    Read the raw, hm images as grayscale.
    Use raw image for background subtraction.
    Stack the images to get HHB representation.
    """
    raw_img = cv2.imread(raw_path, 0)
    raw_img = cv2.resize(raw_img, (224,224))
    should_continue = True
    time.sleep(0.5)
    while(should_continue):
        try:
            hm_img = cv2.imread(hm_path, 0)
            hm_img = cv2.resize(hm_img, (224,224))
            should_continue = False
        except Exception as e:
            print(e)
            # Warm up time for openpose to complete the heatmap
            time.sleep(0.5)
            should_continue = True
    bg_mask = BG_SUBTRACTOR.apply(raw_img)
    if (bg_mask == 255).all():
        bg_mask -= 255
    stacked_3ch_img = np.dstack((hm_img, hm_img, bg_mask))
    print('Saving file to {}'.format(output_path))
    cv2.imwrite(output_path, stacked_3ch_img)
    return

def resize_and_extract_hkb(raw_path, hm_path, kp_path, hkb_dir):

    """
    NOT CURRENT IN USED
    Read the raw, hm and kp images as grayscale.
    Use raw image for background subtraction.
    Stack the images to get HKB representation.
    """
    time.sleep(2)
    output_filename = raw_path.split('\\')[-1] # to be updated
    raw_img = cv2.imread(raw_path, 0)
    raw_img = cv2.resize(raw_img, (224,224))

    hm_img = cv2.imread(hm_path, 0)
    hm_img = cv2.resize(hm_img, (224,224))

    kp_img = cv2.imread(kp_path, 0)
    kp_img = cv2.resize(kp_img, (224,224))

    bg_mask = BG_SUBTRACTOR.apply(raw_img)

    if (bg_mask == 255).all():
    	bg_mask -= 255

    stacked_3ch_img = np.dstack((hm_img, kp_img, bg_mask))

    dst_path = os.path.join(IMAGES_DIR, hkb_dir, output_filename)
    print('Saving file to {}'.format(dst_path))
    cv2.imwrite(dst_path, stacked_3ch_img)

    return


def on_created_custom(event):

    """
    Event handling logic:
    - Track only the heatmap and keypoint folders
    - Save the frame ID to the HM or KP deque, depending on the created file's folder
    - If file was created on heatmap folder, check keypoint folder if the file of the same index is already there
    - When both heatmap file and keypoint file of the same index are found in their deques, call the processing function

    Note : Raw folder is not tracked, as the images are populated much faster than KP and HM.
    """
    folder_updated, file_name = os.path.split(event.src_path)

    if folder_updated != HM_FOLDER:
        # Only check the heat map folder
        return
    # set expire time in redis for 10s
    rd.set('STACKING_IN_PROGRESS', '1', ex=10)
    file_name = file_name.replace('_rendered.png', '')
        # .replace('_keypoints.json', '')
    raw_path = os.path.join(RAW_FOLDER, file_name + '.png')
    hm_path = os.path.join(HM_FOLDER, file_name + '_rendered.png')
    output_path = os.path.join(TARGET_FOLDER, file_name + '.png')
    resize_and_extract_hhb(raw_path, hm_path, output_path)
    return

def observe_and_process():

    '''
    Whenever a new folder is created in the observed folder, trigger the event_handler
    (function : on_created_custom)
    '''

    observer = Observer()
    event_handler = FileSystemEventHandler()
    event_handler.on_created = on_created_custom
    observer.schedule(event_handler, IMAGES_DIR, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

# if __name__ == '__main__':
#     observe_and_process()
