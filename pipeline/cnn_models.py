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

from config import cnn_model_location, crowd_folder

tf_config =  tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)
set_session(session)

print("GPU available: {}".format(tf.test.is_gpu_available()))


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
    retry = 0
    while retry < 20:
        try:
            for i, path in enumerate(frame_paths):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_seq[i] = img
            img_seq /= 255
            results = model.predict(np.expand_dims(img_seq, axis=0), batch_size=1)
            assert results.shape == (1,2), "Incorrect output shape, check model"
            return results[0]
        except:
            time.sleep(0.5)
            retry = retry + 1
