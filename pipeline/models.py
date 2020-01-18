import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import roc_auc_score

from config import cnn_model_location



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
    model = load_model(cnn_model_location, custom_objects={'auroc': auroc})
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


def json_len(json_file):
    json = pd.read_json(json_file)
    counts = len(json)
    return counts


def initial_P_check(file, person_count, person_list):
    person_list.append([person_count, file])
    if len(person_list) % 200 == 0:
        perform_Pchart = 'Y'
        return perform_Pchart, person_list
    else:
        return 'Nil', person_list


def rows_delete(person_list):
    if len(person_list) % 600 == 0:
        person_list = person_list[200:601]
    return person_list


# Determine Upper Control Limit to detect anomaly.
def two_min_pchart(chunk):
    counter_1 = pd.DataFrame(chunk[-200:])
    rolling_3s = counter_1[counter_1.columns[0]].rolling(5).max()
    list_3s = rolling_3s.values.tolist()

    ts_list = []
    for i in list_3s:
        if math.isnan(i):
            ts_list.append(0)
    else:
        ts_list.append(int(i))
    array_2m = np.array(ts_list)
    std_2m = np.std(array_2m)
    mean_2m = np.mean(array_2m)
    # 3-sigma UCL used for detection. UCL rounded down to nearest integer.
    UCL_2m = math.floor(std_2m*3 + mean_2m)
    return mean_2m,std_2m,UCL_2m


# Flag Out-of-Control instants
def out_of_cont_ins(f, pax, ucl):
    if pax > ucl:
        return f, pax, ("OOC")
    else:
        return f, pax, ("Normal")


# Flag Sudden Crowds, Crowds
def rate_of_change(flag, f, pax, person_list):
    if flag == "OOC":
            prev_3s_pax_count = person_list[0][-5]
            if prev_3s_pax_count != 0:
                rate_of_change_perc = (pax - prev_3s_pax_count)/prev_3s_pax_count
                # Rate at which Sudden Crowds are determined
                if rate_of_change_perc >= 0.5:
                    return f, rate_of_change_perc, "Sudden Crowd"
                else:
                    return f, rate_of_change_perc, "Crowd"
            else:
                return f, rate_of_change_perc, "None"
    else:
        return f, "None", "None"


def run_crowd_detection_model(f):
    # Initialize
    person_list = []
    ucl_prev = 0
    # When each file comes in
    person_count = json_len(f)
    person_list = rows_delete(person_list)
    p_check_flag, person_list = initial_P_check(f, person_count, person_list)
    # Check for need to perform statistics (every 2m)
    if p_check_flag == 'Y':
        m_prev, std_dev, ucl_prev = two_min_pchart(person_list)
    if ucl_prev > 0:
        # Check for Out-of-Control
        file_OOC, count_OOC, flag_OOC = out_of_cont_ins(f, person_count, ucl_prev)
        # Flag for Crowd
        file_id, rate_crowd, flag_Crowd = rate_of_change(flag_OOC, file_OOC, count_OOC, person_list)
        # Output to Pipeline
        output = file_id, flag_Crowd
        return flag_Crowd
    return None
