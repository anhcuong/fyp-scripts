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

DEQUE = dict()
DEQUE[HM_FOLDER] = deque(maxlen=100)
DEQUE[KP_FOLDER] = deque(maxlen=100)


def resize_and_extract_hkb(raw_path, hm_path, kp_path, hkb_dir):

    """
    Read the raw, hm and kp images as grayscale.
    Use raw image for background subtraction.
    Stack the images to get HKB representation.
    """

    output_filename = raw_path.split('\\')[-1] # to be updated

    raw_img = cv2.imread(raw_path,0)
    raw_img = cv2.resize(raw_img, (224,224))

    hm_img = cv2.imread(hm_path,0)
    hm_img = cv2.resize(hm_img, (224,224))


    kp_img = cv2.imread(kp_path,0)
    kp_img = cv2.resize(kp_img, (224,224))

    bg_mask = BG_SUBTRACTOR.apply(raw_img)

    if (bg_mask == 255).all():
    	bg_mask -= 255

    stacked_3ch_img = np.dstack((hm_img, kp_img, bg_mask))

    dst_path = os.path.join(IMAGES_DIR, hkb_dir, output_filename)
    print(f'Saving file to {dst_path}')
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

    folder_updated = event.src_path.split('\\')[1]

    if folder_updated == TARGET_FOLDER:

        print('ignoring file update - hhb folder')
        return

    folder_check = HM_FOLDER if folder_updated == KP_FOLDER else KP_FOLDER

    file_name = event.src_path.split('\\')[2]

    new_frame_idx = re.match('[\w]*frame(?P<new_frame_idx>[0-9]{3})[\w]*',
        file_name).group('new_frame_idx')

    print(f'Folder updated : {folder_updated}\n'
          f'Folder to check : {folder_check}\n'
          f'New File Frame ID : {new_frame_idx}\n'
          f'File Name : {file_name}')

    DEQUE[folder_updated].append(new_frame_idx)

    if new_frame_idx in DEQUE[folder_check]:

        print('Running processing function')

        raw_path = os.path.join(RAW_FOLDER,file_name)
        hm_path = os.path.join(HM_FOLDER,file_name)
        kp_path = os.path.join(KP_FOLDER,file_name)
        resize_and_extract_hkb(raw_path, hm_path, kp_path, TARGET_FOLDER)

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
