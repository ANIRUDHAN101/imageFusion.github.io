import sys
sys.path.append('/home/anirudhan/project/fusion')

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from config.data_pipeline_config import get_test_pipeline_config
from config.jax_train_config import get_default_configs
import numpy as np

config = get_test_pipeline_config()
MEAN = config.MEAN
STD = config.STD
PLOT_SAVE_DIR = get_default_configs().plots_save_dir
NAME = get_default_configs().name

def check_and_replace_nan(tensor):
  """
  Checks if a Jax tensor contains NaN values and replaces them with ones of the same shape.

  Args:
      tensor: The Jax tensor to check.

  Returns:
      A new Jax tensor with NaN values replaced by ones, or the original tensor if no NaNs were found.
  """

  # Check if any element is NaN
  if jnp.any(jnp.isnan(tensor)):
    # Create a mask of NaN values
    nan_mask = jnp.isnan(tensor)

    # Replace NaN values with ones using masked_fill
    return jnp.where(nan_mask, 1.0, tensor)
  else:
    # No NaN values found, return the original tensor
    return tensor
  
def denormalize_val_image(image, mean=MEAN, std=STD):
    return (image * std + mean).clip(min=0.0, max=255.0).astype(jnp.uint8)


def denormalize(image):
    image = image * STD + MEAN
    return image


def denormalize_images(func):
    def wrapper(*args, **kwargs):
        denormalized_args = []
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.ndim == 3:
                denormalized_args.append(denormalize(arg))
            else:
                denormalized_args.append(arg)
        return func(*denormalized_args, **kwargs)

    return wrapper


cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


@denormalize_images
def save_plot(image1, image2, prediction, ground_truth):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(image1.astype(np.uint8))
    axes[0, 0].set_title("Image 1")
    axes[0, 1].imshow(image2.astype(np.uint8))
    axes[0, 1].set_title("Image 2")
    axes[1, 0].imshow(prediction.astype(np.uint8))
    axes[1, 0].set_title("Prediction")
    axes[1, 1].imshow(ground_truth.astype(np.uint8))
    axes[1, 1].set_title("Ground Truth")
    plt.tight_layout()
    plt.savefig(f"{PLOT_SAVE_DIR}/{NAME}validation_plot.png")
    plt.close()
