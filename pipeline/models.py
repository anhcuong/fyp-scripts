import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
# from keras.models import load_model
from tensorflow.keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from config import cnn_model_location, crowd_folder



tf_config =  tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)
set_session(session)


def roc_auc_score_modified_multi_lab(y_true, y_pred):
    # to tackle problems where shuffled batches only contains 1 class
    # return 0.5, an underestimate of the actual metric
    #print(y_true.shape, y_pred.shape)

    col_score = []

    for col_true, col_pred in zip(y_true, y_pred):
        try:
            col_score.append(roc_auc_score(y_true, y_pred))
        except ValueError:
            col_score.append(0.5)

    return np.mean(col_score)


def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score_modified_multi_lab, (y_true, y_pred), tf.double)


def get_cnn_model():
    model = load_model(cnn_model_location)
    # model = load_model(cnn_model_location, custom_objects={'auroc': auroc})
    return model


def run_cnn_model(frame_paths, model):

    '''
    frame_paths : list of frames
    model : keras model
    '''
    assert len(frame_paths) == 5, "Check and ensure the length of frame path list is 5."
    img_seq = np.zeros(shape=(5,224,224,3))
    for i, path in enumerate(frame_paths):
        img = cv2.imread(path)
        img_seq[i] = img
    results = model.predict(np.expand_dims(img_seq, axis=0), batch_size=1)
    assert results.shape == (1,2), "Incorrect output shape, check model"
    return results[0]


def get_json_len(path):
    """
    Faster way to get number of people than pd.read_json()
    """
    with open(path, 'rb') as handle:
        json_file = json.load(handle)
        persons_detected= json['people']

        conf_list = []
        for ind, person in enumerate(persons_detected):
            confidence = persons_detected[ind].get('pose_keypoints_2d')[2::3]
            avg_conf = np.mean(confidence)
            if avg_conf > 0.05:
                conf_list.append(avg_conf)
        counts = len(conf_list)
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


def run_scenario_and_plot_sudden_crowd(sample_scenario, path):
    """
    Function to run simulation on sample scenario.
    Input : sample_scenario - a list of pax counts (each pax count correspond to a timestamp)
    Output : plot of number of pax vs timestamp, with red dotted lines indicating detected abnormal crowds
    """
    # initialize the deque to store past 2 minutes pax counts
    pax_hist = 120
    pax_count_past_2_mins = deque(maxlen=pax_hist)
    # initialize list to store detected abnormal timestamp
    abnormal_crowd_timestamp = []
    ucl_timestamp = []

    for index, sample in enumerate(sample_scenario):
        # skip the calculations for first sample
        if len(pax_count_past_2_mins) < pax_hist:
            pax_count_past_2_mins.append(sample)
            continue

        flag, ucl = is_crowd(sample, pax_count_past_2_mins)
        ucl_timestamp.append([index, ucl])
        if flag:
            abnormal_crowd_timestamp.append(index)
        pax_count_past_2_mins.append(sample)
    # plot the scenarios and detected abnormal crowd
    plt.figure(figsize=(20,8))
    plt.plot(sample_scenario, label='Person Count')
    plt.plot(*list(zip(*ucl_timestamp)), ls='--', label='UCL')
    for time in abnormal_crowd_timestamp:
        plt.vlines(x=time, ymin=min(sample_scenario), ymax=max(sample_scenario), color='red', linestyle='--')
    plt.ylabel('Number of people detected')
    plt.xlabel('Timestamp (second)')
    plt.legend(loc="lower right")
    plt.xlim(index-500, index)
    plt.ylim(0, ucl*2)
    plt.show()
    plt.savefig(path)
    # if flag:
    #     return abnormal_crowd_timestamp
    print(abnormal_crowd_timestamp)


def run_crowd_detection_model(file_path):
    # Actual deployment will be something like this
    # define a global deque
    pax_count_graph = deque(maxlen=500)
    pax_count_deque = deque(maxlen=120)
    # control loop handled by Frank's pipeline, run the following everytime new json is generated
    new_pax_count = get_json_len(file_path)
    crowd_flag = is_crowd(new_pax_count, pax_count_deque)
    pax_count_deque.append(new_pax_count)
    pax_count_graph.append(new_pax_count)
    run_scenario_and_plot_sudden_crowd(pax_count_graph, crowd_folder)
