import cv2
import tensorflow as tf
import numpy as np

def denormalize_image(image, segment, variables):
    """Denormalizes an image based on its segment type and normalization method.

    Args:
        image: The normalized image to be denormalized.
        segment: The type of segment (e.g., 'train_masks', 'val_images', etc.).
        variables: A dictionary containing mean and standard deviation values for each segment.

    Returns:
        The denormalized image.
    """

    if segment in ('train_masks', 'val_masks'):

        if np.all(image == 0) or np.all(image == 1):
            center = np.array(image.shape[:2]) // 2
            image[center[0], center[1]] = 0.5


        # Min-max scaling was used, so rescale to 0-255
        denormalized_image = cv2.normalize(
            image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
    else:
        # Standardization was used, so revert back using mean and std
        mean = variables[segment]['mean']
        std = variables[segment]['std']
        denormalized_image = (image * std) + mean

    return denormalized_image


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_example(image, mask, input_img_1, input_img_2):

    feature = {
        'image': _bytes_feature(image),
        'mask': _bytes_feature(mask),
        'input_img_1': _bytes_feature(input_img_1),
        'input_img_2': _bytes_feature(input_img_2),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))
