#%%
import os
import torch
import torch.nn as nn
import PIL.Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import morphology,io
from skimage.color import rgb2gray
from typing import Sequence
from torch import Tensor
import cv2
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision import models
#%%
swin_transformer = models.swin_t(models.Swin_T_Weights.DEFAULT)
swin_neck = swin_transformer.features
#%%
import cv2
import torch
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# Open the image using PIL
image1 = Image.open('/home/anirudhan/Documents/project/fusion/datasets/RealMFF/imageA/012_A.png')
image2 = Image.open('/home/anirudhan/Documents/project/fusion/datasets/RealMFF/imageB/012_B.png')
# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transformations to the image
image1 = transform(image1).unsqueeze(0)
image2 = transform(image2).unsqueeze(0)
#%%
# forward hooks
activation = {}
def getActivation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

h0 = swin_neck[1].register_forward_hook(getActivation('swin 0'))
h1 = swin_neck[3].register_forward_hook(getActivation('swin 1'))
h2 = swin_neck[5].register_forward_hook(getActivation('swin 2'))
h3 = swin_neck[7].register_forward_hook(getActivation('swin 3'))
output = swin_neck(image1)
#%%
h0.remove()
h1.remove()
h2.remove()
h3.remove()
#%%
# . Discrete Tchebichef Moments polynimials
e0 = torch.asarray([-0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333])
e1 = torch.asarray([-0.5164, -0.3873, -0.2582, -0.1291, 0, 0.1281, 0.2582, 0.3873, 0.5164])
e2 = torch.asarray([0.5318, 0.1330, -0.1519, -0.3229, 0, -0.3229, -0.1519, 0.1330, 0.5318])
e3 = torch.asarray([-0.4449, 0.2225, 0.4132, 0.2860, 0, -0.2860, -0.4132, -0.2225, 0.4449])
e4 = torch.asarray([0.3219, -0.4693, -0.2458, 0.2011, 0, 0.2011, -0.2458, -0.4693, 0.3129])
e5 = torch.asarray([-0.1849, 0.5085, -0.1849, -0.4610, 0, 0.4610, 0.1849, -0.5085, 0.1849])
e6 = torch.asarray([0.0899, -0.3820, 0.4944, 0.0225, 0, 0.0225, 0.4944, -0.3820, 0.0899])
e7 = torch.asarray([-0.0341, 0.2048, -0.4780, 0.4780, 0, -0.4780, 0.4780, -0.2048, 0.0341])
e8 = torch.asarray([0.0088, -0.0705, 0.2468, -0.4936, 0, -0.4936, 0.2468, -0.0705, 0.0088])
e = torch.stack([e0 ,e1, e2, e3, e4, e5, e6, e7, e8])
#%%
"""
Model:
    normalize image,
    apply dtm kernel
    apply convolution
    apply crm
"""
class NormalizeImageBlock(nn.Module):
    def __init__(self, block_size=9):
        super().__init__()
        
        self.block_size = block_size
        
        self.w = torch.ones((1, 1, self.block_size, self.block_size))
    
    def forward(self, image):
        image = torch.pow(image, 2)
        image = F.pad(image, (4,4,4,4,0,0,0,0))
        image = F.conv2d(image, self.w)
        
        return image
normalize = NormalizeImageBlock()
normalize(torch.randn((1,  1, 256, 256)))
#%%
class DTMConvBlock(nn.Module):
    def __init__(self, p=5, block_size=9):
        super().__init__()
        
        w = []
        for i in range(0, p+1):
            for j in range(0, p-i+1):
                w.append(e[i].unsqueeze(1) @ e[j].unsqueeze(0))
        
        self.n = len(w)
        self.w = torch.stack(w).view(self.n, 1, block_size, block_size)
        
    def forward(self, image):
        image = image.repeat(1, self.n, 1, 1)
        image = F.conv2d(image, self.w, groups=self.n)
        return image

dtm = DTMConvBlock()
dtm_features = dtm(torch.randn((1,  1, 256, 256)))

#%%
class FocusSegmrntation(nn.Module):
    def __init__(self, p=5, no_classes=3):
        super().__init__()
        self.normalize = NormalizeImageBlock()
        self.dtm = DTMConvBlock(p=p)
        in_channels = self.dtm.n
        
        self.convBolock1 = nn.Conv2d(in_channels*2, 64, kernel_size=1)
        self.convBolock2 = nn.Conv2d(64, 64, kernel_size=1)
        self.convFinal = nn.conv2d(64, no_classes, kernel_size=1)
        
        self.segmenter = nn.Sequential(
            [self.convBolock1(),
             nn.GELU(),
             self.convBolock2(),
             nn.GELU(),
             self.convFinal(),
             nn.softmax()])
        
    def forward(self, image1, image2):
        #normalize images
        image1 = self.normalize(image1)
        image2 = self.normalize(image2)
        
        # extract features STM block
        image1 = self.dtm(image1)
        image2 = self.dtm(image2)
                
        image_features = torch.cat([image1, image2], dim=1)
        # apply conv block
        focus_maps = self.segmenter(image_features)
        
        return focus_maps
    
#%%
# Create an instance of the SegmentFocus model
model = GACNFuseNet()

# Create a dummy input
#image1 = torch.randn(1, 3, 256, 256)  # Assuming input image size is 256x256
#image2 = torch.randn(1, 3, 256, 256)

# Pass the dummy input through the model
output = model(image1, image2)

# Print the output
print(output)