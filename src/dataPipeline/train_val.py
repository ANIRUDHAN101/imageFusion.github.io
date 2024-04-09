#%%
import sys
sys.path.append('/home/anirudhan/project/fusion')
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import yaml
import  albumentations as A
import pandas as pd
import os
from functools import partial
from utils.data import _bytes_feature, _float_feature, _int64_feature, feature_description
from config.data_pipeline_config import get_train_val_pipeline_config
#%%
cfg = get_train_val_pipeline_config()

def set_normalization_transforms():
    if os.path.exists(cfg.YAML_FILE_PATH):
        with open(cfg.YAML_FILE_PATH, 'r') as file:
            return yaml.safe_load(file)

def tfrecord_size(parsed_dataset):
    total_records = 0
    for _ in parsed_dataset:
        total_records += 1
    return total_records

def _parse_function(images):
# Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(images, feature_description)

def _reshape(images):

    images = dict(
        map(lambda item:
        (item[0],tf.reshape(tf.io.decode_raw(item[1], tf.uint8), cfg.INPUT_SHAPE)),
        images.items()))
    return images



def normalize_image(images, split='train'):

    def normalize(image, split):
        variabel = set_normalization_transforms()[f'{split}_images']
        MEAN_RGB = list(map(int, variabel['mean']))
        STDDEV_RGB = list(map(int, variabel['std']))

        image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
        return image

    images['mask'] = (images['mask'] - tf.reduce_min(images['mask'])) / (tf.reduce_max(images['mask']) - tf.reduce_min(images['mask']))

    images['image'] = normalize(images['image'], split)
    images['input_img_1'] = normalize(images['input_img_1'], split)
    images['input_img_2'] = normalize(images['input_img_2'], split)

    return images

def _resize(images):
    images = dict(map(lambda item :
    (item[0], tf.image.resize(item[1], size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))),
    images.items()))
    return images

def uint8_to_float32(image):
    images = dict(
        map(lambda item :
        (item[0], tf.image.convert_image_dtype(item[1], dtype=tf.float32)),
        image.items()))
    return images



def create_split(
    dataset_path,
    batch_size,
    split,
    dtype=tf.float32,
    image_size=cfg.IMAGE_SIZE,
    cache=False,
    shuffle_buffer_size=2_000,
    prefetch=10,
    cfg=None
):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.

    Args:
        dataset_builder: TFDS dataset builder for ImageNet.
        batch_size: the batch size returned by the data pipeline.
        train: Whether to load the train or evaluation split.
        dtype: data type of the image.
        image_size: The target size of the images.
        cache: Whether to cache the dataset.
        shuffle_buffer_size: Size of the shuffle buffer.
        prefetch: Number of items to prefetch in the dataset.
    Returns:
        A `tf.data.Dataset`.
    """
    dataset = tf.data.TFRecordDataset(dataset_path, compression_type=cfg.COMPRESSION, num_parallel_reads=16)
    no_train_samples = tfrecord_size(dataset)# // jax.process_count()
    # start = jax.process_index() * split_size
    # print(split_size, start)
    # dataset.shard(split_size, start)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds = dataset.with_options(options)

    if cache:
        ds = ds.cache()

    if split == 'train':
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size, seed=0)

    ds = ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if split == 'train':
        ds = ds.map(_reshape).map(_resize).map(uint8_to_float32).map(lambda x: normalize_image(x, 'train'))
        # ds = ds.map(_resize)
    elif split == 'val':
        ds = ds.map(_reshape).map(lambda x: normalize_image(x, 'val')).map(_resize)
        # ds = ds.map(_resize)

    ds = ds.batch(batch_size, drop_remainder=True)

    if not split == 'train':
        ds = ds.repeat()

    # ds = ds.prefetch(prefetch)

    # ds = ds.take(tfrecord_size(dataset)//batch_size) # tf record doesnot contain infomation of dataset size, so manully computed and given

    return ds, no_train_samples

# val data pipeline

