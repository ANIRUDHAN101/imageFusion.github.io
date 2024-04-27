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
from src.dataset.train_val import create_split
from src.dataset.coco_dataset import COCODataset
from src.dataset.test import MFI_Dataset 
from src.fusionModel.dataSimulation.algorithm.util.gaussianBlur import RandomGaussianBlur
from config.data_pipeline_config import get_train_val_pipeline_config
from utils.data import save_data_plots
from config.data_pipeline_config import get_test_pipeline_config
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from src.fusionModel.nn.segment import SegmentFocus
from utils.train import convert_grayscale_mask_to_multiclass, mask_to_multiclass, check_and_replace_nan
from src.loss.train_loss import GALoss
from src.loss.losses import FFTLoss, EdgeLoss, MSELoss
from torch import optim
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from train_step import train_step
from val_step import val_step
from torch.utils.tensorboard import SummaryWriter
from src.dataset.get_dataset import train, val, test
import pathlib

IMAGE_SIZE = 128
MULTIPLE_BLUR_CHOICES = 5

exp_name = 'segment_focus_v5'
OUT_DIR = '/home/anirudhan/project/image-fusion/experiments'
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

pathlib.Path(f'{OUT_DIR}/{exp_name}/checkpoints').mkdir(parents=True, exist_ok=True)
pathlib.Path(f'{OUT_DIR}/{exp_name}/logs').mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = f'{OUT_DIR}/{exp_name}/checkpoints'
LOG_PATH = f'{OUT_DIR}/{exp_name}/logs'

train_data_iter = train(data='coco', batch_ize=8)
val_data_iter = val(batch_size=8)
# test_data_iter = test(dataset='RealMFF', batch_size=20)

# model
start_step = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegmentFocus()
model = model.to(device)
opt_model = model
opt_model = torch.compile(model)
# opt_model.load_state_dict(torch.load(f'{CHECKPOINT_PATH}/model_{36}.pth')['model_state_dict'])
# dummpy_out = model(torch.randn((1,3,12
# Pytorch hooks for model debuging
# activation = {}
# def getActivation(name):
#     def hook(model, input, output):
#         activation[name] = output
#     return hook

# encoder_hooks = []
# decoder_hooks = []
# encoder_hooks.append(opt_model.initial_feature_extractor(getActivation('initial_feature_extractor')))
# encoder_hooks.append(opt_model.encoder_blocks[0](getActivation('encoder1')))
# encoder_hooks.append(opt_model.encoder_blocks[1](getActivation('encoder2')))
# encoder_hooks.append(opt_model.encoder_blocks[2](getActivation('encoder3')))
# encoder_hooks.append(opt_model.encoder_blocks[3](getActivation('encoder4')))
# encoder_hooks.append(opt_model.encoder_blocks[4](getActivation('encoder5')))
# decoder_hooks.append(opt_model.decoder_blocks[0](getActivation('decoder1')))
# decoder_hooks.append(opt_model.decoder_blocks[1](getActivation('decoder2')))
# decoder_hooks.append(opt_model.decoder_blocks[2](getActivation('decoder3')))
# decoder_hooks.append(opt_model.decoder_blocks[3](getActivation('decoder4')))

# decoder_hooks.append(opt_model.output_conv_final(getActivation('output_conv_final')))
# decoder_hooks.append(opt_model.guided_filter(getActivation('guided_filter')))

# Training

train_steps = 150
grad_acc = 24

val_steps = 5
diffusion_mask_w = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_decay = 0.0001
optimizer = optim.Adam(opt_model.parameters(), lr=0.0001, weight_decay=weight_decay)


# criterion1 = GALoss().to(device)
criterion2 = FFTLoss().to(device)
criterion3 = MSELoss().to(device)
criterion1 = EdgeLoss(channels=3).to(device)
criterions = [criterion1, criterion2, criterion3]
criterion_weights = [0.333, 0.333, 0.333]

grad_acc = 3

opt_model.train()
writer = SummaryWriter(LOG_PATH)
#%%
for epoch in range(start_step+1, 300):
    train_loss = 0
    val_loss = 0
    opt_model.train()
    for i, data in enumerate(train_data_iter):
        model, optimizer, train_loss = train_step(data, opt_model, criterions, criterion_weights, optimizer, grad_acc, start_step, writer, device)

        if i % grad_acc == 0:
            # print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")
            optimizer.step()
            optimizer.zero_grad()

        if i % train_steps == 0 and i != 0: break

    val_loss = val_step(val_data_iter, opt_model, criterions, criterion_weights, val_steps, epoch, writer, device)

    writer.add_scalar('Loss/train', train_loss/train_steps, epoch)
    writer.add_scalar('Loss/val', val_loss/val_steps, epoch)
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss/train_steps}, Val Loss: {val_loss/val_steps}")

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': opt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        },CHECKPOINT_PATH+f'/model_{epoch}.pth'
    )

writer.close()
# %%
