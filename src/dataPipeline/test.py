#%%
import sys
sys.path.append('/home/anirudhan/project/fusion')

import tensorflow as tf
import jax
import tensorflow_datasets as tfds
import os
import yaml
import  albumentations as A
import pandas as pd
import os
from functools import partial
from config.data_pipeline_config import get_test_pipeline_config

cfg = get_test_pipeline_config()

def decode_imges(images):
    imaegs = dict(map(lambda item:
                   (item[0], tf.io.decode_jpeg(item[1], channels=3)),
                   images.items()))
    return imaegs

def resize_imges(images):
    images = dict(map(lambda item:
                   (item[0], tf.image.resize(item[1], [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE])),
                   images.items()))
    return images

def load_pathes(file_paths):
    images = dict(map(lambda item:
                (item[0],  tf.io.read_file(item[1])),
                file_paths.items()))
    return images

def normalize_val_image(images):

    def normalize(image):
        image -= tf.constant(cfg.MEAN, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(cfg.STD, shape=[1, 1, 3], dtype=image.dtype)
        return image

    images = dict(map(lambda item:(item[0], normalize(item[1]) ), images.items()))
    return images

def val_data(file_path, batch_size=32):
    df = pd.read_csv(file_path)
    for column in df.columns:
        df[column] = df[column].apply(lambda x: os.path.join(cfg.FOLDER, x))
    data = tf.data.Dataset.from_tensor_slices(dict(df))
    return data.map(load_pathes).map(decode_imges).map(resize_imges).map(normalize_val_image).batch(batch_size).repeat()