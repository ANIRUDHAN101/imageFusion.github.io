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
from src.fusionModel.nn.segment import SegmentFocus
from utils.train import convert_grayscale_mask_to_multiclass, mask_to_multiclass, check_and_replace_nan
from src.loss.train_loss import GALoss
from src.loss.losses import FFTLoss, EdgeLoss, MSELoss
from torch import optim
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter

IMAGE_SIZE = 128

torch.set_float32_matmul_precision('medium')
train_val_cfg = get_train_val_pipeline_config()

options = tf.data.Options()
options.threading.private_threadpool_size = 48

output_folder = '/home/anirudhan/project/image-fusion/results/plots/data' 
train_dataset_path = '/home/anirudhan/project/image-fusion/data/memmaps/train_images100.tfrecords.gz'
split = 'train'
# dataset = tf.data.TFRecordDataset(dataset_path, compression_type="GZIP").with_options(options)
# dataset = dataset.map(_parse_function).map(_reshape).map(lambda x: normalize_image(x, 'train')).map(_resize)
train_dataset, no_train_samples = create_split(train_dataset_path, 20, split, shuffle_buffer_size=5_00, cfg=train_val_cfg)
train_dataset = train_dataset.as_numpy_iterator()

save_data_plots(train_dataset, output_folder, split,no_samples=1)

train_val_cfg.COMPRESSION = 'GZIP'
val_dataset_path = '/home/anirudhan/project/image-fusion/data/memmaps/val_images.tfrecords.gz'
split = 'val'
# dataset = tf.data.TFRecordDataset(dataset_path, compression_type="GZIP").with_options(options)
# dataset = dataset.map(_parse_function).map(_reshape).map(lambda x: normalize_image(x, 'train')).map(_resize)
val_dataset, no_train_samples = create_split(val_dataset_path, 20, split, cfg=train_val_cfg)
val_dataset = val_dataset.as_numpy_iterator()
save_data_plots(train_dataset, output_folder, split,no_samples=1)

test_cfg = get_test_pipeline_config()

test_dataset= val_data('/home/anirudhan/project/image-fusion/data/RealMFF/data.csv', batch_size=2).as_numpy_iterator()
save_data_plots(test_dataset, output_folder, 'test', no_samples=4)

# dataloader 

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

numpy_train_data = map(augment, train_dataset)
train_data_iter = map(numpy_to_torch, numpy_train_data)


val_data_iter = map(numpy_to_torch, val_dataset)
test_data_iter = map(numpy_to_torch, test_dataset)

# model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SegmentFocus([16, 16, 32, 32], 16)
model = model.to(device)
opt_model = model
opt_model = torch.compile(model)
# dummpy_out = model(torch.randn((1,3,12

# Training

train_step = 150
grad_acc = 1

val_step = 5
diffusion_mask_w = 0.5

# fabric = Fabric(accelerator="auto", strategy="auto", devices=1, precision='16-mixed')
#tensorboard_logger = TensorBoardLogger(root_dir='/content/lightning_logs')
# fabric.launch()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SegmentFocus([32, 64, 128], 16)
# model = GACNFuseNet()
weight_decay = 0.1
optimizer = optim.Adam(opt_model.parameters(), lr=0.0001, weight_decay=weight_decay)

# opt_mdodel, optimizer = fabric.setup(opt_mdodel, optimizer)

# criterion1 = FusionLoss().to(device)
# criterion2 = GALoss().to(device)
# criterion3 = torch.nn.CrossEntropyLoss().to(device)
criterion1 = MSELoss().to(device)
criterion2 = FFTLoss().to(device)
criterion3 = EdgeLoss().to(device)
grad_acc = 2

opt_model.train()
writer = SummaryWriter('/home/anirudhan/project/image-fusion/results/logs')
CHECKPOINT_PATH = '/home/anirudhan/project/image-fusion/results/checkpoints'
#%%
for epoch in range(300):
    train_loss = 0
    val_loss = 0
    opt_model.train()
    for i, data in enumerate(train_data_iter):
        data['mask'] = check_and_replace_nan(data['mask'])
        # mask = mask_to_one_hot(data['mask'][:,0,:,:]).to(device)
        mask = mask_to_multiclass(data['mask'], num_classes=3).to(device)
        # mask = data['mask'].to(device)
        gt_image = data['image'].to(device)
        image1 = data['input_img_1'].to(device)
        image2 = data['input_img_2'].to(device)
        # optimizer.zero_grad()
        output, output_mask = opt_model(image1, image2, gt_image, mask)
        loss = .3*criterion1(output, gt_image) + .3*criterion2(output, gt_image) + .3*criterion3(output, gt_image) 
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
            gt_image = data['image'].to(device)
            output, output_mask = opt_model(data['input_img_1'].to(device), data['input_img_2'].to(device), gt_image, mask)
            loss = .3*criterion1(output, gt_image) + .3*criterion2(output, gt_image) + .3*criterion3(output, gt_image)
            val_loss += loss.item()
            if i % val_step == 0 and i != 0: break
    writer.add_scalar('Loss/train', train_loss/train_step, epoch)
    writer.add_scalar('Loss/val', val_loss/val_step, epoch)
    
    # val_visual = make_grid([output[0], mask[0]]).permute(1,2,0).cpu().numpy()
    val_visual = torch.stack([output[0], gt_image[0], output_mask[0], mask[0]], dim=0)
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
