import gc
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import albumentations as A
import torch
from src.dataset.train_val import create_split
from src.dataset.coco_dataset import COCODataset
from src.dataset.test import MFI_Dataset 
from src.fusionModel.dataSimulation.algorithm.util.gaussianBlur import RandomGaussianBlur
from config.data_pipeline_config import get_train_val_pipeline_config
from utils.data import save_data_plots
from config.data_pipeline_config import get_test_pipeline_config
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from utils.train import convert_grayscale_mask_to_multiclass, mask_to_multiclass, check_and_replace_nan
from torch import optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms


IMAGE_SIZE = 128
MULTIPLE_BLUR_CHOICES = 50
options = tf.data.Options()
options.threading.private_threadpool_size = 16

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

train_val_cfg = get_train_val_pipeline_config()
train_dataset_path = '/home/anirudhan/project/image-fusion/data/memmaps/train_images100.tfrecords.gz'

albumentation_transform = A.Compose([
        # A.augmentations.geometric.transforms.Affine(
        #     # scale=[0.6,1.0],
        #     translate_percent=0.5,
        #     # translate_px=[0,1],
        #     rotate=[-45,45],
        #     shear=[-45,45],
        #     # interpolation=1,
        # )
        A.Resize(height=256, width=256),
        A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Rotate(limit=45),
        # A.CenterCrop(height=1000, width=1000),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    additional_targets={'image':'image','input_img_1':'image', 'input_img_2':'image', 'mask':'image'},
    is_check_shapes=False,
    )

def numpy_to_torch(data):
    return dict(map(lambda item: (item[0], torch.from_numpy(item[1].copy()).permute(0, 3, 1, 2)), data.items()))

def augment(data):
    temp_data = {'image': [], 'mask': [], 'input_img_1': [], 'input_img_2': []}
    for i in range(data['image'].shape[0]):
        temp = albumentation_transform(image=data['image'][i],
                                        mask=data['mask'][i],
                                        input_img_1=data['input_img_1'][i],
                                        input_img_2=data['input_img_2'][i])
        temp_data['image'].append(temp['image'])
        temp_data['mask'].append(temp['mask'])
        temp_data['input_img_1'].append(temp['input_img_1'])
        temp_data['input_img_2'].append(temp['input_img_2'])

    data['image'] = np.stack(temp_data['image'], axis=0)
    data['mask'] = np.stack(temp_data['mask'], axis=0)
    data['input_img_1'] = np.stack(temp_data['input_img_1'], axis=0)
    data['input_img_2'] = np.stack(temp_data['input_img_2'], axis=0)
    return data

def train(data='coco', batch_ize=20):
    if data=='coco':
        rand_gausian_blur = RandomGaussianBlur(30, 'cpu')
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = COCODataset(
            data_dir='/home/anirudhan/project/image-fusion/data/coco/images/train2017',
            mask_dir='/home/anirudhan/project/image-fusion/data/coco/images/train_2017mask', 
            simulation=rand_gausian_blur,
            multiple_blur_choices=MULTIPLE_BLUR_CHOICES,
            crop_size=512, need_crop=False, need_rotate=True, need_flip=True)

        return DataLoader(train_dataset, batch_size=batch_ize, shuffle=True, persistent_workers=True, num_workers=16, pin_memory=True, prefetch_factor=2)
    
    else:
        train_dataset, no_train_samples = create_split(train_dataset_path, batch_ize, 'train', cfg=train_val_cfg)
        train_dataset = train_dataset.as_numpy_iterator()
        train_dataset = map(augment, train_dataset)
        return map(numpy_to_torch, train_dataset)

def val(batch_size=20):
    train_val_cfg.COMPRESSION = 'GZIP'
    val_dataset_path = '/home/anirudhan/project/image-fusion/data/memmaps/val_images.tfrecords.gz'
    val_dataset, no_train_samples = create_split(val_dataset_path, 20, 'val', cfg=train_val_cfg)
    val_dataset = val_dataset.as_numpy_iterator()
    return map(numpy_to_torch, val_dataset)

def test(dataset,batch_size=2):
    if dataset == 'RealMFF':
        datasetPath = '/home/anirudhan/project/image-fusion/data/valid/RealMFF'
        phase = 'full_clear'

    elif dataset == 'MFI-WHU':
        datasetPath = '/home/anirudhan/project/image-fusion/data/valid/MFI-WHU/MFI-WHU'
        phase = None

    elif dataset == 'Lytro':
        datasetPath = '/home/anirudhan/project/image-fusion/data/valid/Lytro'
        phase = None
    
    elif dataset == 'MFFW':
        datasetPath = '/home/anirudhan/project/image-fusion/data/valid/MFFW'
        phase = None

    else:
        print("dataset doesnot exist")

    test_dataset = MFI_Dataset(
        datasetPath=datasetPath,
        phase=phase,
        use_dataTransform=True,
        resize=None,
        imgSzie=None)

    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=2, pin_memory=True, prefetch_factor=2)
