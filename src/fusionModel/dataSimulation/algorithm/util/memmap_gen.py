from model.dataset.dataset import AM2KDataset
import os
import  albumentations as A
from  albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from model.model_utility.gaussianBlur import RandomGausianBlur
import torch
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import numpy as np

albumentation_transform = A.Compose([
    #A.Resize(height=800, width=600),
    # A.Rotate(limit=45),
    A.CenterCrop(height=1000, width=1000),
    # A.HorizontalFlip(p=0.5),
    #A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'background': 'image'},
is_check_shapes=False,
)
am2k = AM2KDataset(split='train',
                   #base_dir='/home/anirudhan/Documents/project/fusion/Fusion/AM_2k_dataset_colletor/AM-2k',
                   base_dir=f'{os.getcwd()}/data/AM-2k',
                   transform=albumentation_transform,
                   blur_simulation=False)

am2k_dataLoader = DataLoader(am2k,batch_size=20, shuffle=True,num_workers=12)
def saveImage(image, filename):
    image = Image.fromarray(image.squeeze(0).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
    image.save(f'{filename}.png')
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
blur_module_gpu = RandomGausianBlur(filter_size = 3, deviation = 10, device=device)


for i in range(1000):
    img = next(iter(am2k_dataLoader))
    print(f"retreived batch {i}")
    # for key, value in img.items():
    #     print(f"for {key}: {value.shape}, {value}")

    sim_data = blur_module_gpu(img['image'].to(device), img['mask'].to(device),  multiple_blur_choices=50)

    if 'data' not in locals():
        data = {}
        for key, value in sim_data.items():
            data[key] = value
    
    else:
        for key, value in sim_data.items():
            data[key] = torch.cat((data[key], value), dim=0)
print('Saving')
for key, value in data.items():
    fp = np.memmap(f'{key}.npy', dtype=np.float32, mode='w+', shape=value.shape)
    fp[:] = value[:].detach().to('cpu').numpy()
    fp.flush()



    

    

# for key, value in sim_data.items():
#     print(f"for {key}: {value.shape}, {value}")


from PIL import Image
# saveImage(img['image'],'image')
# saveImage(img['input_img_1'], 'input_img_1')
# saveImage(img['input_img_2'], 'input_img_2')
# saveImage(img['mask'], 'mask.jpg')
# save_image(img['image'][0], 'image.jpg')
# save_image(img['input_img_1'][0], 'input_img_1.jpg')
# save_image(img['input_img_2'][0], 'input_img_2.jpg')

# save_image(img['mask'][0], 'mask.jpg')

# image_np = img['image'].squeeze(0).permute(1, 2, 0).numpy()
# image_np = (image_np * 255).astype(np.uint8)  # convert to uint8
# image = Image.fromarray(image_np)
# image.save("image.jpg")

# image = Image.fromarray(img['input_img_1'].to('cpu').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
# image.save("input_image1.jpg")

# image = Image.fromarray(img['input_img_2'].to('cpu').squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
# image.save("input_image2.jpg")
# Values less than 100 are set to 0
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


