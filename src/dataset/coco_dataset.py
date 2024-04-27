#%%
import PIL.Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import cv2
import random
import numpy as np
import albumentations as A

class COCODataset(Dataset):
    def __init__(self, data_dir, mask_dir, 
                simulation,
                multiple_blur_choices=20,
                crop_size=256,
                transform=None, need_crop=False, need_rotate=False, need_flip=False):
        
        self._images_basename = os.listdir(mask_dir)
        if '.ipynb_checkpoints' in self._images_basename:
            self._images_basename.remove('.ipynb_checkpoints')
        
        self._data_address = [os.path.join(data_dir, item) for item in sorted(self._images_basename)]
        self._mask_address = [os.path.join(mask_dir, item) for item in sorted(self._images_basename)]
        self._mask_address = self._mask_address
        self._crop_size = crop_size
        self._transform = A.Compose([
        # A.augmentations.geometric.transforms.Affine(
        #     # scale=[0.6,1.0],
        #     translate_percent=0.5,
        #     # translate_px=[0,1],
        #     rotate=[-45,45],
        #     shear=[-45,45],
        #     # interpolation=1,
        # )
        A.Resize(height=256, width=256),
        A.RandomCrop(height=self._crop_size, width=self._crop_size),
        A.Rotate(limit=45),
        # A.CenterCrop(height=1000, width=1000),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    additional_targets={'image':'image', 'mask':'image'},
    is_check_shapes=False,
    )
        self._origin_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self._need_rotate = need_rotate
        self._need_crop = need_crop
        self._need_flip = need_flip
        self._simulation = simulation
        self.MULTIPLE_BLUR_CHOICES = multiple_blur_choices

    def __len__(self):
        return len(self._mask_address)

    def __getitem__(self, idx):
        #print(idx)
        data = cv2.imread(self._data_address[idx]) 
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        # data = cv2.resize(data, (256, 256))  / 255.0
        mask = cv2.imread(self._mask_address[idx]) 
        # mask = cv2.resize(mask, (256, 256))  
        
        roi_tensor = self._transform(image=data, mask=mask)
        roi_image_tensor = self._origin_transform(roi_tensor['image'])
        roi_mask_tensor = self._mask_transform(roi_tensor['mask'])

        roi_mask_tensor[roi_mask_tensor < 0.5] = 0
        roi_mask_tensor[roi_mask_tensor > 0.5] = 1

        data = self. _simulation(roi_image_tensor, roi_mask_tensor, self.MULTIPLE_BLUR_CHOICES)

        return data

    
#%%
#%%
