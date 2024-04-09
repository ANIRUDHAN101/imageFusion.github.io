import sys
sys.path.append('/home/anirudhan/project/fusion')


from matplotlib import pyplot as plt
from config.data_pipeline_config import get_test_pipeline_config
from config.jax_train_config import get_default_configs
import numpy as np
import torch

config = get_test_pipeline_config()
MEAN = config.MEAN
STD = config.STD
PLOT_SAVE_DIR = get_default_configs().plots_save_dir
NAME = get_default_configs().name

def convert_grayscale_mask_to_multiclass(grayscale_mask, num_classes):
    """
    Converts a grayscale image mask to a multi-class mask image with channels.

    Args:
        grayscale_mask (torch.Tensor): The grayscale mask tensor (H, W), where each pixel value represents a class.
        num_classes (int): The number of classes in the multi-class segmentation task.

    Returns:
        torch.Tensor: The multi-class mask image with channels (H, W, num_classes), one-hot encoded.
    """

    # Ensure grayscale mask is a LongTensor with values in the range [0, num_classes - 1]
    grayscale_mask = grayscale_mask.long()
    grayscale_mask = grayscale_mask * 2
    if grayscale_mask.min() < 0 or grayscale_mask.max() >= num_classes:
        raise ValueError("Grayscale mask values must be in the range [0, num_classes - 1]")

    # Create a one-hot encoded tensor using torch.nn.functional.one_hot
    multiclass_mask = torch.nn.functional.one_hot(grayscale_mask, num_classes=num_classes)

    # Move the channel dimension to the first position
    multiclass_mask = multiclass_mask.permute(2, 0, 1)

    return multiclass_mask

def mask_to_multiclass(grayscale_image, num_classes=3):
    """ Convert grayscale mask to multi-class mask with channels."""    
    lower_bound = .1
    upper_bound = .9

    mask_below = (grayscale_image < lower_bound).float()
    condition_within = (grayscale_image >= lower_bound) & (grayscale_image <= upper_bound)
    mask_above = 1-(grayscale_image <= upper_bound).float()  # Exclude values below lower_bound
    mask = torch.cat((mask_below[:, 0].unsqueeze(1), condition_within[:, 0].unsqueeze(1), mask_above[:, 0].unsqueeze(1)), dim=1)
    return mask

def check_and_replace_nan(tensor):
    """
    Checks if a PyTorch tensor contains NaN values and replaces them with ones of the same shape.

    Args:
        tensor: The PyTorch tensor to check.

    Returns:
        A new PyTorch tensor with NaN values replaced by ones, or the original tensor if no NaNs were found.
    """

    # Check if any element is NaN
    if torch.isnan(tensor).any():
        # Create a mask of NaN values
        nan_mask = torch.isnan(tensor)

        # Replace NaN values with ones using masked_fill
        return tensor.masked_fill(nan_mask, 1.0)
    else:
        # No NaN values found, return the original tensor
        return tensor
