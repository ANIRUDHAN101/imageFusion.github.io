#%%
import os
import random
import PIL.Image
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
from model.model_utility.gaussianBlur import RandomGausianBlur, blur_simulation

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
    BG20_FILE_PATH = '//content/drive/MyDrive/fusion/BG20k/val'
    AM2k_VAL = '/content/drive/MyDrive/fusion/AM_2k/val'
    ANN_FILE = '/content/drive/MyDrive/fusion/val.json'

    def __init__(self, split, 
                 base_dir,
                 transform=None,
                 blur_simulation=True):
        super(AM2KDataset, self).__init__()
        
       
        self.bg_2k_files = os.listdir(self.BG20_FILE_PATH)
        self.am_2k_files = os.listdir(self.AM2k_VAL)
        self.base_dir = self.AM2k_VAL
        self.image_resize = transforms.Compose([transforms.Resize([512,512], antialias=True)])
        self.blur_simulation = blur_simulation
        
        with open(self.ANN_FILE, 'r') as f:
            self._image_dict = json.load(f)

        self._transform = transform
    
    def __len__(self):
        return len(self._image_dict)

    def __getitem__(self, idx):
        
        image, mask = self.get_image_and_mask(idx)
        background_image = self.get_background_image()
        mask = self.normalize_mask(mask)

        if self._transform is not None:

            image, mask = self.train_image_generator(image=image, mask=mask, background_image=background_image)
                
            
        image = self.set_image_to_tensor(image)
        mask = self.set_image_to_tensor(mask)

        data = {}
        data['image'] = image
        data['mask'] = mask

        if self.blur_simulation:
            input_img_1, input_img_2, mask = blur_simulation(image, mask, deviation=None, filter_size=None, multiple_blur_choices=100)
            data['image'] = image
            data['input_img_1'] = input_img_1
            data['input_img_2'] = input_img_2
            data['mask'] = mask
        # print(f'sutable bg files: {self.sutable_bg_files}')
        return data

    def set_image_to_tensor(self, image):
        if isinstance(image, np.ndarray) and image.shape[-1] == 3:
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)  # Move the channel dimension to the beginning
        
        if isinstance(image,np.ndarray) and len(image.shape) < 3:
            image = torch.from_numpy(image)
        
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


    def normalize_mask(self, mask: np.array) -> np.array:
        """
        Normalizes a mask image to the range [0, 1].

        Args:
            mask (np.array): The mask image to normalize. This should be a 2D numpy array.

        Returns:
            np.array: The normalized mask image. This is a 2D numpy array with the same shape as the input, but with values normalized to the range [0, 1].
        """
        return cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
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
        
        image_file_name = os.path.join(self.base_dir, 'images', image_file_name)
        mask_file_name = os.path.join(self.base_dir, 'masks', mask_file_name)

        if not os.path.exists(image_file_name):
            logging.error(f"Image not found at {image_file_name}")
            raise FileNotFoundError(f"Image not found at {image_file_name}")

        if not os.path.exists(mask_file_name):
            logging.error(f"Image not found at {mask_file_name}")
            raise FileNotFoundError(f"Image not found at {mask_file_name}")

        image = cv2.imread(image_file_name)
        mask = cv2.imread(mask_file_name, 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

            return cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
        
        raise FileNotFoundError("No sutable image found")

    
class FileNameMismatchError(Exception):
    """Raised when the image and mask filenames do not match"""
    pass
#%%
import matplotlib.pyplot as plt
import os
import  albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2
albumentation_transform = A.Compose([
    #A.Resize(height=800, width=600),
    A.Rotate(limit=45),
    A.CenterCrop(height=1000, width=1000),
    # A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'background': 'image'},
is_check_shapes=False,
)
am2k = AM2KDataset(split='train',
                   base_dir=f'{os.getcwd()}/data/AM-2k',
                   transform=albumentation_transform,
                   blur_simulation=True)

am2k_dataLoader = DataLoader(am2k,batch_size=182, shuffle=False)
def saveImage(image, filename):
    image = Image.fromarray(image.squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
    image.save(f'{filename}.png')
#%%
device = 'cuda'
blur_module_gpu = RandomGausianBlur(filter_size = 3, deviation = 10, device=device, )

img = next(iter(am2k_dataLoader))
# for key, value in img.items():
#     print(f"for {key}: {value.shape}, {value}")

# sim_data = blur_module_gpu(img['image'].to('cuda'), img['mask'].to('cuda'), no_of_blurs=17)

# img = sim_data
# for key, value in sim_data.items():
#     print(f"for {key}: {value.shape}, {value}")


from PIL import Image
import shutil
# saveImage(img['image'],'image')
# saveImage(img['input_img_1'], 'input_img_1')
# saveImage(img['input_img_2'], 'input_img_2')
# saveImage(img['mask'], 'mask.jpg')

save_dir_img = '/content/drive/MyDrive/fusion/AM_2k/valSim/image'
save_dir_mask = '/content/drive/MyDrive/fusion/AM_2k/valSim/mask'
save_dir_object_blur = '/content/drive/MyDrive/fusion/AM_2k/valSim/object_blur'
save_dir_baground_blur = '/content/drive/MyDrive/fusion/AM_2k/valSim/baground_blur'
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
clear_directory(save_dir_img)
clear_directory(save_dir_mask)
clear_directory(save_dir_object_blur)
clear_directory(save_dir_baground_blur)
def save_batch(batch_images, index):
    for i in range(batch_images['image'].shape[0]):
        save_image(img['image'][i], f'{save_dir_img}/{index}.jpg')
        save_image(img['input_img_1'][i], f'{save_dir_object_blur}/{index}.jpg')
        save_image(img['input_img_2'][i], f'{save_dir_baground_blur}/{index}.jpg')
        save_image(img['mask'][i], f'{save_dir_mask}/{index}.jpg')
        index +=1
    return index

index = 0
for i, batch in enumerate(am2k_dataLoader):
    print(f"saving batch:{i}")
    index = save_batch(batch, index)


# image_np = img['image'].squeeze(0).permute(1, 2, 0).numpy()
# image_np = (image_np * 255).astype(np.uint8)  # convert to uint8
# image = Image.fromarray(image_np)
# image.save("image.jpg")

# image = Image.fromarray(img['input_img_1'].to('cpu').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
# image.save("input_image1.jpg")

# image = Image.fromarray(img['input_img_2'].to('cpu').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
# image.save("input_image2.jpg")
# Values less than 100 are set to 0
mask = img['mask'].to('cpu').squeeze(0)
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
