import tensorflow as tf
from dataPipeline.test import create_split

options = tf.data.Options()
options.threading.private_thread_size = 48

dataset_path = ''

train_dataset, no_train_samples = create_split(dataset_path, 32, 'train', shuffle_buffer)