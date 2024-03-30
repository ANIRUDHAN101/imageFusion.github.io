import sys
sys.path.append('/home/anirudhan/project/fusion')

import jax
import jax.numpy as jnp
import einops
import jaxwt as jwt
import numpy as np
from skimage.metrics import structural_similarity
from config.jax_train_config import get_default_configs

config = get_default_configs()

@jax.jit
def charbonnier_loss(predicted_image, groundtruth_image):
            return jnp.mean(jnp.sqrt(jnp.abs(predicted_image - groundtruth_image + 1e-3)))
@jax.jit
def wavelet_loss(predicted_image, gt_image):
    gt_image = einops.rearrange(gt_image, "n h w c -> n c h w")
    gt_image_transforms = jwt.wavedec2(gt_image, "haar", level=config.level, mode="reflect")
    predicted_image = einops.rearrange(predicted_image, "n h w c -> n c h w")
    predicted_image_dwt = jwt.wavedec2(predicted_image, "haar", level=config.level, mode="reflect")
    fused_transforms = [jnp.abs(predicted_image_dwt.pop(0)-gt_image_transforms.pop(0) +  1e-3)]

    for image1_coefficents, image2_coefficents in zip(predicted_image_dwt, gt_image_transforms):
        fused_transforms + list(map(lambda x, y: jnp.abs(x-y+ 1e-3), image1_coefficents, image2_coefficents))

    return jnp.mean(jnp.array(fused_transforms, dtype=jnp.float32))

def ssim_loss(image1, image2):
    assert image1.shape == image2.shape
    n = image1.shape[0]
    loss = []
    for i in range(n):
        loss.append(structural_similarity(image1[i], image2[i],
                    channel_axis=2))
    return np.mean(loss)