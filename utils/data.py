import tensorflow as tf
from matplotlib import pyplot as plt
import os

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'mask': tf.io.FixedLenFeature([], tf.string),
    'input_img_1': tf.io.FixedLenFeature([], tf.string),
    'input_img_2': tf.io.FixedLenFeature([], tf.string),
}

def save_data_plots(dataset, output_folder, split, no_samples=4):

    if split != 'test':
        for i, parsed_record in enumerate(dataset):
            # Access images directly (assuming parsed_record already contains decoded images)
            img = parsed_record['image']
            mask = parsed_record['mask']
            input_img_1 = parsed_record['input_img_1']
            input_img_2 = parsed_record['input_img_2']

            # Create a grid of images with Matplotlib
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            axes[0].imshow(img[0])
            axes[0].set_title("Image")
            axes[1].imshow(mask[0])
            axes[1].set_title("Mask")
            axes[2].imshow(input_img_1[0])
            axes[2].set_title("Input Image 1")
            axes[3].imshow(input_img_2[0])
            axes[3].set_title("Input Image 2")

            # Tight layout for consistent spacing
            plt.tight_layout()

            # Save grid as an image to the output folder
            grid_filename = f"{split}_{i}.png"  # Or any other desired format
            grid_filepath = os.path.join(output_folder, grid_filename)
            plt.savefig(grid_filepath)
            plt.close(fig)  # Close figure to avoid memory issues

            if i == no_samples:  break
    if split == 'test':
        for i, parsed_record in enumerate(dataset):
            print(parsed_record.keys())
            fusion = parsed_record['Fusion']
            imageA = parsed_record['imageA']
            imageB = parsed_record['imsgeB']
            # Create subplots with labels
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust figsize as needed
            axes[0].imshow(fusion[0])
            axes[0].set_title("Fusion")
            axes[1].imshow(imageA[0])
            axes[1].set_title("image A")
            axes[2].imshow(imageB[0])
            axes[2].set_title("image B")

            # Tight layout for consistent spacing
            plt.tight_layout()

            # Save grid as an image to the output folder
            grid_filename = f"{split}_{i}.png"  # Or any other desired format
            grid_filepath = os.path.join(output_folder, grid_filename)
            plt.savefig(grid_filepath)
            plt.close(fig)  # Close figure to avoid memory issues

            if i == no_samples : break