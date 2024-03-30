#%%
import os
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from torchvision import transforms 
import torch
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import json
import logging
from torchvision.transforms import functional
from algorithm.util.gaussianBlur import RandomGaussianBlur, blur_simulation
import yaml

class AM2KDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

    Args:
        split (str): The type of dataset to load ('train', 'val', 'test').
        base_dir (str): The directory where the dataset is located.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        img_dir (str): The directory where the images are located.
        split (str): The type of dataset to load ('train', 'val', 'test').
    """

    BG20_FILE_PATH = '/home/anirudhan/project/fusion/data/BG20k/train'
    VAL_IMG_PATH = '/home/anirudhan/project/fusion/data/AM_2k/valSim'
    ANNOTATIONS_PATH = '/home/anirudhan/project/fusion/data/AM_2k/annotations'
    NORM_YAML = '/home/anirudhan/project/fusion/config/dataset.yml'
    
    def __init__(self, split, 
                 base_dir,
                 transform=None,
                 blur_simulation=True,
                 image_height=512,
                 image_width=512):

        super(AM2KDataset, self).__init__()
        print(split)
        print(base_dir)
        ann_file = os.path.join(base_dir, f'{self.ANNOTATIONS_PATH}/{split}.json')
        self.img_dir = os.path.join(base_dir, f'{split}/')
        self.split = split
        self.base_dir = base_dir
        self.bg_2k_files = os.listdir(self.BG20_FILE_PATH)
        self.image_resize = transforms.Compose([transforms.Resize([image_height,image_width], antialias=True)])
        self.variables =  self.set_normalization_transforms()
        self.blur_simulation = blur_simulation

        if split == 'val':
            self.load_val_images()

        with open(ann_file, 'r') as f:
            self._image_dict = json.load(f)

        self._transform = transform
    
    def __len__(self):
        if self.split == 'train':
            return len(self._image_dict)

        return len(self.val_images)

    def __getitem__(self, idx):
        data = {}

        if self.split == 'train':
        
            image, mask = self.get_image_and_mask(idx)
            background_image = self.get_background_image()
            

            # this function fuses the image and background image using the mask and 
            # returns the fused image , and its corresponding mask        
            if self._transform is not None:
                image, mask = self.train_image_generator(image=image, mask=mask, background_image=background_image)
            else:
                data['background_image'] = self.set_image_to_tensor(background_image)
                
            image = self.set_image_to_tensor(image)
            mask = self.set_image_to_tensor(mask)
            data['mask'] = mask
            data['image'] = image

            if self.blur_simulation:
                input_img_1, input_img_2, mask = blur_simulation(image, mask, deviation=None, filter_size=None, multiple_blur=20)
                data['image'] = image
                data['input_img_1'] = input_img_1
                data['input_img_2'] = input_img_2
                data['mask'] = mask

        elif self.split == 'val':
            input_img_1, image, mask, input_img_2 = self.get_val_images(idx)
            
            data['image'] = image
            data['input_img_1'] = input_img_1
            data['input_img_2'] = input_img_2
            data['mask'] = mask
            

        else:
            raise WrongSplitArgument(f"Used wroing split:{self.split} should use 'val' or 'train'")
        
        return data

    def get_val_images(self, idx):
        input_img_1_path = os.path.join(self.VAL_IMG_PATH, 'baground_blur', self.val_baground_blurs[idx])
        image_path = os.path.join(self.VAL_IMG_PATH, 'image', self.val_images[idx])
        mask_path = os.path.join(self.VAL_IMG_PATH, 'mask', self.val_masks[idx])
        input_img_2_path = os.path.join(self.VAL_IMG_PATH, 'object_blur', self.val_object_blurs[idx])
        
        if self._transform is not None:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, 0)
            input_img_1 = cv2.imread(input_img_1_path)
            input_img_1 = cv2.cvtColor(input_img_1, cv2.COLOR_BGR2RGB)
            input_img_2 = cv2.imread(input_img_2_path)
            input_img_2 = cv2.cvtColor(input_img_2, cv2.COLOR_BGR2RGB)
            data = self._transform(image=image, mask=mask, input_img_1=input_img_1, input_img_2=input_img_2)
            return data['input_img_1'], data['image'], data['mask'], data['input_img_2']
        
        input_img_1 = Image.open(input_img_1_path).convert('RGB')
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Assuming mask is a single-channel grayscale image
        input_img_2 = Image.open(input_img_2_path).convert('RGB')
        

        # Apply transformations
        input_img_1 = self.set_image_to_tensor(input_img_1)
        image = self.set_image_to_tensor(image)
        mask = self.set_image_to_tensor(mask)
        input_img_2 = self.set_image_to_tensor(input_img_2)

        input_img_1 = self.image_resize(input_img_1)
        image = self.image_resize(image)
        mask = self.image_resize(mask)
        mask = mask.repeat(3, 1, 1) # for returning a tneor of shape equalent to 3 channels 
        input_img_2 = self.image_resize(input_img_2)
        return input_img_1, image, mask, input_img_2
        
    def load_val_images(self):
        self.val_baground_blurs = os.listdir(os.path.join(self.VAL_IMG_PATH, 'baground_blur'))
        self.val_images = os.listdir(os.path.join(self.VAL_IMG_PATH, 'image'))
        self.val_masks = os.listdir(os.path.join(self.VAL_IMG_PATH, 'mask'))
        self.val_object_blurs  = os.listdir(os.path.join(self.VAL_IMG_PATH, 'object_blur'))
        
        self.val_masks.sort()
        self.val_images.sort()
        self.val_object_blurs.sort()
        self.val_baground_blurs.sort()

        if not (self.val_masks == self.val_images == self.val_object_blurs == self.val_baground_blurs):
            raise ValidationDataMismatch("Lists do not contain the same values after sorting")

    def set_normalization_transforms(self):
        if os.path.exists(self.NORM_YAML):
            with open(self.NORM_YAML, 'r') as file:
                return yaml.safe_load(file)
                # self.normalizations = {}
                # for key in self.data.keys():
                #     self.normalizations[key] = transforms.Normalize(mean=self.variables[key]['mean'], std=self.variables[key]['std'])
    
    def set_image_to_tensor(self, image):
        if isinstance(image, np.ndarray) and image.shape[-1] == 3:
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)  # Move the channel dimension to the beginning
        
        elif not isinstance(image, torch.Tensor) :
            image = F.to_tensor(image)

        return image
        
    def train_image_generator(self, image: np.array, mask: np.array, background_image: np.array) -> torch.tensor:
        """
        Generates a training image by applying transformations, masking, and resizing.

        Args:
            image (np.array): The original image.
            mask (np.array): The mask to apply to the image.
            background_image (np.array): The background image to use where the mask is not applied.

        Returns:
            torch.Tensor: The resulting image after applying the transformations, masking, and resizing.
        """
        augmentation = self._transform(image=image, background=background_image, mask=mask)
        image = augmentation['image']
        background = augmentation['background']
        mask = augmentation['mask']

        if len(mask.shape) == 3:
            mask = mask.permute(2, 0, 1)
        else:
            c, _, _ = image.shape
            mask = mask.repeat(c, 1, 1)

        image = image * mask + background * (1 - mask)

        image = self.image_resize(image)
        mask = self.image_resize(mask)

        return image, mask

    def normalizations(self, image: np.array, segment: str) -> any:
        """
        Normalizes a mask image to the range [0, 1].

        Args:
            mask (np.array): The mask image to normalize. This should be a 2D numpy array.

        Returns:
            np.array: The normalized mask image. This is a 2D numpy array with the same shape as the input, but with values normalized to the range [0, 1].
        """

        if segment == 'train_masks':
            # return (image - self.variables['train_masks']['mean']) / self.variables['train_masks']['std']
            return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
        
        elif segment == 'val_masks':
            return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
        elif segment == 'train_images':
            return (image - self.variables['train_images']['mean']) / self.variables['train_images']['std']
        
        elif segment == 'val_images':
            return (image - self.variables['val_images']['mean']) / self.variables['val_images']['std']
        
        elif segment == 'train_background_images':
            return (image - self.variables['train_background_images']['mean']) / self.variables['train_background_images']['std']
        
        elif segment == 'val_background_images':
            return (image - self.variables['val_background_images']['mean']) / self.variables['train_background_images']['std']
        
    def get_image_and_mask(self, index: int) -> tuple[np.array, np.array]:
        """
        Retrieves a pair of image and its corresponding mask

        Args:
            index (int): The index of the list.

        Return:
            image, mask (np.array, np.array): The images are converted from BGR to RGB.

        Raises:
            FileNotFoundError: If file cannot be read.
            FileNameMismatchError: If the image and mask filenames do not match.
        """

        image_file_name = self._image_dict[str(index)]['image']
        mask_file_name = self._image_dict[str(index)]['mask']
        
        if os.path.splitext(image_file_name)[0] != os.path.splitext(mask_file_name)[0]:
            raise FileNameMismatchError("Image and mask filenames do not match")
        
        image_file_name = os.path.join(self.base_dir, self.split, 'original', image_file_name)
        mask_file_name = os.path.join(self.base_dir, self.split, 'mask', mask_file_name)

        if not os.path.exists(image_file_name):
            logging.error(f"Image not found at {image_file_name}")
            raise FileNotFoundError(f"Image not found at {image_file_name}")

        if not os.path.exists(mask_file_name):
            logging.error(f"Image not found at {mask_file_name}")
            raise FileNotFoundError(f"Image not found at {mask_file_name}")

        image = cv2.imread(image_file_name)
        mask = cv2.imread(mask_file_name, 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.normalizations(image, f'{self.split}_images')
        mask = self.normalizations(mask, f'{self.split}_masks')
    
        assert (image.shape[0], image.shape[1]) == (mask.shape[0], mask.shape[1])

        return image, mask
    
    def get_background_image(self, min_image_size: int = 1000) -> np.array:
        """
        Retrives a random background image ensuring its smallest dimenion is atleast 'min_image_size'.

        Args:
            min_image_size (int, optional): Minimum size for the small dimension of the image. Default to 1000.
        
        Return:
            np.array: The selected background image as a numpy array, converted from BGR to RGB.
        
        Raises:
            FileNotFoundError:  if the image file cannot be read.
        """
        if not hasattr(self, 'sutable_bg_files'):
            self.sutable_bg_files = self.bg_2k_files.copy()

        background_img_size = 0

        while background_img_size < min_image_size and self.sutable_bg_files:
            random_index = random.randint(0, len(self.bg_2k_files)-1)
            image_path = os.path.join(self.BG20_FILE_PATH, self.bg_2k_files[random_index])
            background_image = cv2.imread(image_path)
            
            if background_image is None:
                logging.error(f"Image file not found: {image_path}")
                self.sutable_bg_files.pop(random_index)
                continue

            h, w, _ = background_image.shape
            if min(h,w) < min_image_size:
                self.sutable_bg_files.pop(random_index)
                continue

            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
            return self.normalizations(background_image, f'{self.split}_background_images')
        
        raise FileNotFoundError("No sutable image found")
    
class FileNameMismatchError(Exception):
    """Raised when the image and mask filenames do not match"""
    pass

class WrongSplitArgument(Exception):
    """Raised when argument used is either 'train' or 'val'"""
    pass

class ValidationDataMismatch(Exception):
    """Raised when The filenames of images masks blur images doesnot match"""
    pass
#%%
# import matplotlib.pyplot as plt
# import os
# import  albumentations as A
# from  albumentations.pytorch.transforms import ToTensorV2
# albumentation_transform = A.Compose([
#     #A.Resize(height=800, width=600),
#     # A.Rotate(limit=45),
#     A.CenterCrop(height=1000, width=1000),
#     # A.HorizontalFlip(p=0.5),
#     #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2(),
# ],
# additional_targets={'background': 'image'},
# is_check_shapes=False,
# )
# am2k = AM2KDataset(split='val',
#                    base_dir=f'{os.getcwd()}/data/AM-2k',
#                    transform=albumentation_transform,
#                    blur_simulation=False)

# am2k_dataLoader = DataLoader(am2k,batch_size=1, shuffle=True)
# def saveImage(image, filename):
#     image = Image.fromarray(image.squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
#     image.save(f'{filename}.png')
# #%%
# device = 'cuda'
# blur_module_gpu = RandomGausianBlur(filter_size = 3, deviation = 10, device=device)

# img = next(iter(am2k_dataLoader))
# # for key, value in img.items():
# #     print(f"for {key}: {value.shape}, {value}")

# # sim_data = blur_module_gpu(img['image'].to('cuda'), img['mask'].to('cuda'),  multiple_blur_choices=50)

# # img = sim_data
# # for key, value in sim_data.items():
# #     print(f"for {key}: {value.shape}, {value}")


# from PIL import Image
# # saveImage(img['image'],'image')
# # saveImage(img['input_img_1'], 'input_img_1')
# # saveImage(img['input_img_2'], 'input_img_2')
# # saveImage(img['mask'], 'mask.jpg')
# save_image(img['image'][0], 'image.jpg')
# save_image(img['input_img_1'][0], 'input_img_1.jpg')
# save_image(img['input_img_2'][0], 'input_img_2.jpg')

# save_image(img['mask'][0], 'mask.jpg')

# # image_np = img['image'].squeeze(0).permute(1, 2, 0).numpy()
# # image_np = (image_np * 255).astype(np.uint8)  # convert to uint8
# # image = Image.fromarray(image_np)
# # image.save("image.jpg")

# # image = Image.fromarray(img['input_img_1'].to('cpu').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
# # image.save("input_image1.jpg")

# # image = Image.fromarray(img['input_img_2'].to('cpu').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
# # image.save("input_image2.jpg")
# # Values less than 100 are set to 0
# mask = img['mask'].to('cpu').squeeze(0)
# mask[mask <= 0] = 0

# Values between 100 and 200 are set to 100
# mask[(mask > .1) & (mask < .7)] = .5

# # Values greater than or equal to 200 are set to 255
# mask[mask >= 1] = 1
# image = Image.fromarray((mask.permute(1, 2, 0).numpy()*255).astype(np.uint8))
# image.save("mask.jpg")
#%%
# coco_augment = DataAugment(coco_dataLoader)

# %%
# image_batch = next(iter(coco_augment))

# image_batch = torch.stack([image_batch[0], 
#                            image_batch[1].to('cpu'),
#                            image_batch[2].to('cpu'),
#                            image_batch[3].to('cpu')])
# # %%
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# for i in range(4):
#     axs[i].imshow(image_batch[i].squeeze(0).permute( 0, 1, 2).numpy())
#     axs[i].axis('off')

# plt.show()

# %%
