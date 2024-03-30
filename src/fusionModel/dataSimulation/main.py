import sys
sys.path.append('/home/anirudhan/project/fusion')

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tensorflow as tf
from torch.utils.data import DataLoader
from config.data_simulation_config import get_simulation_config
from algorithm.util.gaussianBlur import RandomGaussianBlur
from dataloader.dataset import AM2KDataset
from dataloader.util import denormalize_image, image_example
from tqdm import tqdm
import time

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

# Get simulation config
cfg = get_simulation_config()

# Define Albumentations transform
albumentation_transform = A.Compose([
    #A.Resize(height=800, width=600),
    A.Rotate(limit=45),
    A.CenterCrop(height=1000, width=1000),
    A.HorizontalFlip(p=0.5),
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'background': 'image'},
is_check_shapes=False,
)

# Create AM2K dataset
am2k = AM2KDataset(
    split=cfg.SPLIT,
    base_dir=cfg.TRAIN_IMG_DIR,
    transform=albumentation_transform,
    blur_simulation=False,
    image_height=cfg.IMAGE_HEIGHT,
    image_width=cfg.IMAGE_WIDTH
)

# Create AM2K DataLoader
am2k_dataLoader = DataLoader(
    am2k,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    prefetch_factor=cfg.PREFETCH_FACTOR,
    pin_memory=cfg.PIN_MEMORY
)

# Create Gaussian blur module
blur_module_gpu = RandomGaussianBlur(filter_size=cfg.FILTER_SIZE, deviation=cfg.DEVIATION, device=DEVICE)

# Set normalization transforms
variables = am2k.set_normalization_transforms()

# Set TFRecord options
options = tf.io.TFRecordOptions(compression_type=cfg.COMPRESSION)

start_time = time.time()
# Write data to TFRecord file
with tf.io.TFRecordWriter(cfg.TF_RECORD_DIR, options) as writer:
    for i in tqdm(range(cfg.NUM_EPOCHS)):
        img = next(iter(am2k_dataLoader))
        print(f"Retrieved batch {i}")

        sim_data = blur_module_gpu(
            img['image'].to(DEVICE),
            img['mask'].to(DEVICE),
            multiple_blur_choices=cfg.MULTIPLE_BLUR_CHOICES,
            dtype=torch.float16
        )

        for i in range(sim_data['image'].shape[0]):
            values = {}
            if not torch.isnan(sim_data['mask'][i]).any():
                for key, value in sim_data.items():
                    values[key] = np.clip(
                        denormalize_image(
                            sim_data[key][i].permute(1, 2, 0).cpu().numpy(),
                            cfg.SPLIT + '_' + 'images' if key != 'mask' else cfg.SPLIT + '_' + 'masks',
                            variables
                        ),
                        0,
                        255
                    ).astype(np.uint8).tobytes()
                tf_data = image_example(**values)
                writer.write(tf_data.SerializeToString())
            else:
                print(f'Found NaN value at: {i}')

    # Calculate total time taken
    
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time} seconds")
