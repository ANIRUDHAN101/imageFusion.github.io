#%%
from skimage.metrics import structural_similarity, normalized_mutual_information, variation_of_information
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torchvision.utils import save_image
from src.dataset.get_dataset import test
import os
import pathlib

transform = transforms.Compose([
    # transforms.Resize((128, 12)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=MEAN, std=STD)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

OUT_DIR = '/home/anirudhan/project/image-fusion/experiments'

#%%
def test_step(model, dataset, device, val_data, exp_name):
    pathlib.Path(f'{OUT_DIR}/{exp_name}/{val_data}/full_clear').mkdir(parents=True, exist_ok=True) 
    pathlib.Path(f'{OUT_DIR}/{exp_name}/{val_data}/mask').mkdir(parents=True, exist_ok=True) 

    model = model.eval()
    with torch.no_grad():
        for i, image in enumerate(dataset):  # Iterate through the dataset
            image['source_1'] = transform(image['source_1'])
            image['source_2'] = transform(image['source_2'])
            # image['full_clear'] = transform(image['full_clear'])

            prediction, mask, _ = model(
                image['source_1'].to(device), 
                image['source_2'].to(device), 
                image['source_1'].to(device), 
                gt_mask = torch.ones_like( image['source_1'].to(device)))
            
            output_image_path = os.path.join(OUT_DIR, exp_name, val_data,'full_clear', f'output_{i+1}.png')
            output_mask_path = os.path.join(OUT_DIR, exp_name, val_data, 'mask', f'mask_{i+1}.png')

            prediction = inv_normalize(prediction)

            save_image(prediction[:,[2,1,0],:,:], output_image_path)
            save_image(mask, output_mask_path)
            print(i)
            
#%%
from src.fusionModel.nn.segment import SegmentFocus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegmentFocus([32, 32, 32, 32, 32], 2)
model = model.to(device)
opt_model = model
opt_model = torch.compile(model)
opt_model.load_state_dict(torch.load(f'/home/anirudhan/project/image-fusion/experiments/segment_focus_v4/checkpoints/model_97.pth')['model_state_dict'])

test_step(opt_model, dataset=test('Lytro', batch_size=1), device=device, val_data='Lytro', exp_name='segment_focus_v4')

# %%
