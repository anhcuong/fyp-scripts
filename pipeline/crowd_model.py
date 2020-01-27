import os
import math
import time
from collections import deque
import json

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    cnn_model_location, crowd_folder, keypoint_folder, image_dir)
from utils import display_crowd


def get_json_len(path):
    """
    Faster way to get number of people than pd.read_json()
    """
    print(path)
    should_continue = True
    time.sleep(0.5)
    while(should_continue):
        try:
            with open(path) as f:
                json_file = json.load(f)
                persons_detected= json_file['people']

                conf_list = []
                for ind, person in enumerate(persons_detected):
                    confidence = persons_detected[ind].get('pose_keypoints_2d')[2::3]
                    avg_conf = np.mean(confidence)
                    if avg_conf > 0.05:
                        conf_list.append(avg_conf)
                counts = len(conf_list)
                should_continue= False
        except Exception as e:
            print(e)
            # Warm up time for openpose to complete the heatmap
            time.sleep(0.5)
            should_continue = True
    return counts


def calculate_ucl(pax_deque):
    mean = np.mean(pax_deque)
    std = np.std(pax_deque)
    return mean + 3 * std


def is_crowd(new_pax_count, pax_deque, lookback=3):
    """
    Abnormal crowd algorithm - test if the new_pax_count is consider abnormal crowd.
    Input:
    - new_pax_count : single pax count
    - pax_deque : list of past pax counts
    - lookback : pax counts from *lookback* seconds ago is used to check for % increase
    Algorithm:
    1. Check if new_pax_count is more than the upper control limit
    2. Check if new_pax_count is 50% more than the pax_count from 3 seconds ago
    3. Exclude if new_pax_count is less than 3
    """
    # skip if we only have 2 seconds worth of pax counts
    if len(pax_deque) < lookback:
        return False, calculate_ucl(pax_deque)
    # check if new count is out of control
    if new_pax_count > calculate_ucl(pax_deque) and new_pax_count > 3:
        # check if increase is > 150%  in the past 3s
        if new_pax_count > pax_deque[-lookback] * 1.5:
            return True, calculate_ucl(pax_deque)
        else:
            return False, calculate_ucl(pax_deque)
    else:
        return False, calculate_ucl(pax_deque)


pax_hist = 120
pax_count_graph = deque(maxlen=500)
pax_count_deque = deque(maxlen=pax_hist)
pax_count_past_2_mins = deque(maxlen=pax_hist)


def run_scenario_and_plot_sudden_crowd(path):
    """
    Function to run simulation on sample scenario.
    Output : plot of number of pax vs timestamp, with red dotted lines indicating detected abnormal crowds
    """
    if len(pax_count_past_2_mins) < pax_hist:
        return
    # initialize list to store detected abnormal timestamp
    abnormal_crowd_timestamp = []
    ucl_timestamp = []
    ucl = 0
    for index, sample in enumerate(pax_count_deque):
        flag, ucl = is_crowd(sample, pax_count_past_2_mins)
        ucl_timestamp.append([index, ucl])
        if flag:
            abnormal_crowd_timestamp.append(index)
        pax_count_past_2_mins.append(sample)
    # plot the scenarios and detected abnormal crowd
    plt.figure(figsize=(20, 8))
    plt.plot(pax_count_deque, label='Person Count')
    plt.plot(*list(zip(*ucl_timestamp)), ls='--', label='UCL')
    for time in abnormal_crowd_timestamp:
        plt.vlines(x=time, ymin=min(pax_count_deque), ymax=max(pax_count_deque), color='red', linestyle='--')
    plt.ylabel('Number of people detected')
    plt.xlabel('Timestamp (second)')
    plt.legend(loc="lower right")
    plt.xlim(index-500, index)
    plt.ylim(0, ucl*2)
    plt.savefig(path)
    return abnormal_crowd_timestamp


def on_created_custom(event):
    folder_updated, file_name = os.path.split(event.src_path)
    if folder_updated != keypoint_folder:
        # Only check the heat map folder
        return
    new_pax_count = get_json_len(event.src_path)
    crowd_flag = is_crowd(new_pax_count, pax_count_deque)
    pax_count_deque.append(new_pax_count)
    pax_count_graph.append(new_pax_count)
    crowd_path = os.path.join(crowd_folder, '1.png')
    run_scenario_and_plot_sudden_crowd(crowd_path)
    display_crowd(path, str(crowd_flag))
    return

def observe_and_process():

    '''
    Whenever a new folder is created in the observed folder, trigger the event_handler
    (function : on_created_custom)
    '''

    observer = Observer()
    event_handler = FileSystemEventHandler()
    event_handler.on_created = on_created_custom
    observer.schedule(event_handler, image_dir, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
