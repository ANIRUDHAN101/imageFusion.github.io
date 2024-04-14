#%%
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
from src.dataPipeline.train_val import create_split
from config.data_pipeline_config import get_train_val_pipeline_config
from utils.data import save_data_plots
from src.dataPipeline.test import val_data 
from config.data_pipeline_config import get_test_pipeline_config
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from src.fusionModel.nn.segment import GACNFuseNet
from utils.train import convert_grayscale_mask_to_multiclass, mask_to_multiclass, check_and_replace_nan
from src.loss.train_loss import GALoss
from torch import optim
from torchvision.utils import make_grid
from src.dataPipeline.coco_dataset import COCODataset
from src.fusionModel.dataSimulation.algorithm.util.gaussianBlur import RandomGaussianBlur
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

IMAGE_SIZE = 128
MULTIPLE_BLUR_CHOICES = 5

torch.set_float32_matmul_precision('medium')
train_val_cfg = get_train_val_pipeline_config()

options = tf.data.Options()
options.threading.private_threadpool_size = 48

output_folder = '/home/anirudhan/project/image-fusion/results/plots/data' 
train_dataset_path = '/home/anirudhan/project/image-fusion/data/memmaps/train_images100.tfrecords.gz'
split = 'train'

rand_gausian_blur = RandomGaussianBlur(10, 'cpu')
train_data_iter = COCODataset(
    data_dir='/home/anirudhan/project/image-fusion/data/coco/images/train2017',
    mask_dir='/home/anirudhan/project/image-fusion/data/coco/images/train_2017mask', 
    simulation=rand_gausian_blur,
    multiple_blur_choices=MULTIPLE_BLUR_CHOICES,
    crop_size=128, need_crop=False, need_rotate=True, need_flip=True)

train_dataloader = DataLoader(train_data_iter, batch_size=20, shuffle=True, persistent_workers=True, num_workers=16, pin_memory=True, prefetch_factor=2)

train_val_cfg.COMPRESSION = 'GZIP'
val_dataset_path = '/home/anirudhan/project/image-fusion/data/memmaps/val_images.tfrecords.gz'
split = 'val'
# dataset = tf.data.TFRecordDataset(dataset_path, compression_type="GZIP").with_options(options)
# dataset = dataset.map(_parse_function).map(_reshape).map(lambda x: normalize_image(x, 'train')).map(_resize)
val_dataset, no_train_samples = create_split(val_dataset_path, 20, split, cfg=train_val_cfg)
val_dataset = val_dataset.as_numpy_iterator()
save_data_plots(val_dataset, output_folder, split,no_samples=1)

test_cfg = get_test_pipeline_config()

test_dataset= val_data('/home/anirudhan/project/image-fusion/data/RealMFF/data.csv', batch_size=2).as_numpy_iterator()
save_data_plots(test_dataset, output_folder, 'test', no_samples=4)

# dataloader 

def numpy_to_torch(data):
    return dict(map(lambda item: (item[0], torch.from_numpy(item[1].copy()).permute(0, 3, 1, 2)), data.items()))


val_data_iter = map(numpy_to_torch, val_dataset)
test_data_iter = map(numpy_to_torch, test_dataset)

# model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = SegmentFocus([16, 192, 384, 768], 8)
model = GACNFuseNet()
model = model.to(device)
opt_model = model
opt_model = torch.compile(model)
# dummpy_out = model(torch.randn((1,3,12

# Training

train_step = 200
grad_acc = 1

val_step = 5
diffusion_mask_w = 0.5

# fabric = Fabric(accelerator="auto", strategy="auto", devices=1, precision='16-mixed')
#tensorboard_logger = TensorBoardLogger(root_dir='/content/lightning_logs')
# fabric.launch()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SegmentFocus([32, 64, 128], 16)
# model = GACNFuseNet()
weight_decay = 0.001
optimizer = optim.Adam(opt_model.parameters(), lr=0.0001, weight_decay=weight_decay)

# opt_mdodel, optimizer = fabric.setup(opt_mdodel, optimizer)

# criterion1 = FusionLoss().to(device)
criterion2 = GALoss().to(device)
criterion3 = torch.nn.CrossEntropyLoss().to(device)
grad_acc = 2


writer = SummaryWriter('/home/anirudhan/project/image-fusion/results/logs')
CHECKPOINT_PATH = '/home/anirudhan/project/image-fusion/results/checkpoints'
opt_model.load_state_dict(torch.load('/home/anirudhan/project/image-fusion/results/checkpoints/model_72.pth')['model_state_dict'])
opt_model.train()
#%%
for epoch in range(300):
    train_loss = 0
    val_loss = 0
    opt_model.train()
    for i, data in enumerate(train_dataloader):
        data['mask'] = check_and_replace_nan(data['mask'])
        # mask = mask_to_one_hot(data['mask'][:,0,:,:]).to(device)
        mask = mask_to_multiclass(data['mask'], num_classes=3).to(device)
        # mask = data['mask'].to(device)
        # gt_image = data['image'].to(device)
        image1 = data['input_img_1'].to(device)
        image2 = data['input_img_2'].to(device)
        # optimizer.zero_grad()
        output = opt_model(image1, image2)
        loss = criterion2(output, mask) + criterion3(output, mask) #+ diffusion_mask_w*criterion2(diffusion_mask, mask[:,2,:,:])
        loss.backward()
        # optimizer.step()
        train_loss += loss.item()

        if i % grad_acc == 0:
            # print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")
            optimizer.step()
            optimizer.zero_grad()

        # this code reoves local varibles to clear gpu ram
        # del loss
        # del output
        # del gt_image
        # del image1
        # del image2
        # del mask
        # del data

        # torch.cuda.empty_cache()
        # gc.collect()

        if i % train_step == 0 and i != 0: break

    opt_model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_data_iter):
            data['mask'] = check_and_replace_nan(data['mask'])
            # mask = mask_to_one_hot(data['mask'][:,0,:,:]).to(device)
            mask = mask_to_multiclass(data['mask'], num_classes=3).to(device)
            # mask = data['mask'].to(device)
            # gt_image = data['image'].to(device)
            output = opt_model(data['input_img_1'].to(device), data['input_img_2'].to(device))
            loss = criterion2(output, mask) + criterion3(output, mask) #+ diffusion_mask_w*criterion2(diffusion_mask, mask[:,2,:,:])
            val_loss += loss.item()
            if i % val_step == 0 and i != 0: break
    writer.add_scalar('Loss/train', train_loss/train_step, epoch)
    writer.add_scalar('Loss/val', val_loss/val_step, epoch)
    
    # val_visual = make_grid([output[0], mask[0]]).permute(1,2,0).cpu().numpy()
    val_visual = torch.stack([output[0], mask[0]], dim=0)
    writer.add_images('val image and predicted images', val_visual, epoch)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/train_step}, Val Loss: {val_loss/val_step}")

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': opt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        },CHECKPOINT_PATH+f'/model_{epoch}.pth'
    )

writer.close()
# %%
