#%%
import os
import torch
import torch.nn as nn
import PIL.Image
import torch.nn.functional as f
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
from crfseg import CRF

# Open the image using PIL
# image1 = Image.open('/home/anirudhan/Documents/project/fusion/datasets/RealMFF/imageA/012_A.png')
# image2 = Image.open('/home/anirudhan/Documents/project/fusion/datasets/RealMFF/imageB/012_B.png')
# # Define the transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Apply the transformations to the image
# image1 = transform(image1).unsqueeze(0)
# image2 = transform(image2).unsqueeze(0)
# #%%
# # forward hooks
# activation = {}
# def getActivation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

# h0 = swin_neck[1].register_forward_hook(getActivation('swin 0'))
# h1 = swin_neck[3].register_forward_hook(getActivation('swin 1'))
# h2 = swin_neck[5].register_forward_hook(getActivation('swin 2'))
# h3 = swin_neck[7].register_forward_hook(getActivation('swin 3'))
# output = swin_neck(image1)
# #%%
# h0.remove()
# h1.remove()
# h2.remove()
# h3.remove()
#%%
class GACNFuseNet(nn.Module):
    """
    The Class of SESFuseNet
    """
    def __init__(self):
        super(GACNFuseNet, self).__init__()
        
        # feature_extraction
        self.feature_extraction_conv0 = self.conv_block(1, 16, name="feature_extraction_conv0")
        self.se_0 = CSELayer(16, 8)
        self.feature_extraction_conv1 = self.conv_block(16, 16, name="feature_extraction_conv1")
        self.se_1 = CSELayer(16, 8)
        self.feature_extraction_conv2 = self.conv_block(32, 16, name="feature_extraction_conv2")
        self.se_2 = CSELayer(16, 8)
        self.feature_extraction_conv3 = self.conv_block(48, 16, name="feature_extraction_conv3")
        self.se_3 = CSELayer(16, 8)
        
        # decision_path
        self.se_4 = SSELayer(64)
        self.se_5 = SSELayer(48)
        self.se_6 = SSELayer(32)
        self.se_7 = SSELayer(16)
        self.se_8 = SSELayer(3)
        self.decision_path_conv1 = self.conv_block(64, 48, name="decision_path_conv1")
        self.decision_path_conv2 = self.conv_block(48, 32, name="decision_path_conv2")
        self.decision_path_conv3 = self.conv_block(32, 16, name="decision_path_conv3")
        self.decision_path_conv4 = self.conv_block(16, 3,  name="decision_path_conv4")

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, relu=True, batchnorm=True, name=None):
        """
        The conv block of common setting: conv -> relu -> bn
        In conv operation, the padding = 1
        :param in_channels: int, the input channels of feature
        :param out_channels: int, the output channels of feature
        :param kernel_size: int, the kernel size of feature
        :param relu: bool, whether use relu
        :param batchnorm: bool, whether use bn
        :param name: str, name of the conv_block
        :return:
        """
        block = torch.nn.Sequential()

        block.add_module(name+"_Conv2d", torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                                                        out_channels=out_channels, padding=kernel_size // 2))
        if relu:
            block.add_module(name+"_ReLu", torch.nn.ReLU())
        if batchnorm:
            block.add_module(name+"_BatchNorm", torch.nn.BatchNorm2d(out_channels))
        return block
    
    @staticmethod
    def concat(f1, f2):
        """
        Concat two feature in channel direction
        """
        return torch.cat((f1, f2), 1)
    
    @staticmethod
    def fusion_channel_sf(f1, f2, kernel_radius=5):
        """
        Perform channel sf fusion two features
        """
        device = f1.device
        b, c, h, w = f1.shape
        r_shift_kernel = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])\
            .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])\
            .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_r_shift = f.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = f.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = f.conv2d(f2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = f.conv2d(f2, b_shift_kernel, padding=1, groups=c)
        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
        f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)
        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
        kernel_padding = kernel_size // 2
        f1_sf = f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c)
        f2_sf = f.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c)

        # get decision map
        bimap = torch.sigmoid(1000 * (f1_sf - f2_sf))
        # bimap = torch.nn.Softmax(dim=1)(f1_sf - f2_sf)
        # bimap = torch.nn.ReLU()(f1_sf - f2_sf)
        return bimap
    
    def forward(self, img1, img2):
        """
        Train or Forward for two images
        :param img1: torch.Tensor
        :param img2: torch.Tensor
        :return: output, torch.Tensor
        """
        img1 = rgb_to_grayscale(img1)
        img2 = rgb_to_grayscale(img2)
        
        # Feature extraction c1
        feature_extraction_conv0_c1 = self.feature_extraction_conv0(img1)
        se_feature_extraction_conv0_c1 = self.se_0(feature_extraction_conv0_c1)
        feature_extraction_conv1_c1 = self.feature_extraction_conv1(se_feature_extraction_conv0_c1)
        se_feature_extraction_conv1_c1 = self.se_1(feature_extraction_conv1_c1)
        se_cat1_c1 = self.concat(se_feature_extraction_conv0_c1, se_feature_extraction_conv1_c1)
        feature_extraction_conv2_c1 = self.feature_extraction_conv2(se_cat1_c1)
        se_feature_extraction_conv2_c1 = self.se_2(feature_extraction_conv2_c1)
        se_cat2_c1 = self.concat(se_cat1_c1, se_feature_extraction_conv2_c1)
        feature_extraction_conv3_c1 = self.feature_extraction_conv3(se_cat2_c1)
        se_feature_extraction_conv3_c1 = self.se_3(feature_extraction_conv3_c1)
        
        # Feature extraction c2
        feature_extraction_conv0_c2 = self.feature_extraction_conv0(img2)
        se_feature_extraction_conv0_c2 = self.se_0(feature_extraction_conv0_c2)
        feature_extraction_conv1_c2 = self.feature_extraction_conv1(se_feature_extraction_conv0_c2)
        se_feature_extraction_conv1_c2 = self.se_1(feature_extraction_conv1_c2)
        se_cat1_c2 = self.concat(se_feature_extraction_conv0_c2, se_feature_extraction_conv1_c2)
        feature_extraction_conv2_c2 = self.feature_extraction_conv2(se_cat1_c2)
        se_feature_extraction_conv2_c2 = self.se_2(feature_extraction_conv2_c2)
        se_cat2_c2 = self.concat(se_cat1_c2, se_feature_extraction_conv2_c2)
        feature_extraction_conv3_c2 = self.feature_extraction_conv3(se_cat2_c2)
        se_feature_extraction_conv3_c2 = self.se_3(feature_extraction_conv3_c2)

        # SF fusion
        cat_1 = torch.cat((se_feature_extraction_conv0_c1, se_feature_extraction_conv1_c1, 
                           se_feature_extraction_conv2_c1, se_feature_extraction_conv3_c1), axis=1)
        cat_2 = torch.cat((se_feature_extraction_conv0_c2, se_feature_extraction_conv1_c2, 
                           se_feature_extraction_conv2_c2, se_feature_extraction_conv3_c2), axis=1)
        fused_cat = self.fusion_channel_sf(cat_1, cat_2, kernel_radius=5)
        se_f = self.se_4(fused_cat)

        # Decision path
        decision_path_conv1 = self.decision_path_conv1(se_f)
        se_decision_path_conv1 = self.se_5(decision_path_conv1)
        decision_path_conv2 = self.decision_path_conv2(se_decision_path_conv1)
        se_decision_path_conv2 = self.se_6(decision_path_conv2)
        decision_path_conv3 = self.decision_path_conv3(se_decision_path_conv2)
        se_decision_path_conv3 = self.se_7(decision_path_conv3)
        decision_path_conv4 = self.decision_path_conv4(se_decision_path_conv3)
        # se_decision_path_conv4 = self.se_8(decision_path_conv4)
        
        # Boundary guided filter
        #output_origin = torch.sigmoid(1000 * se_decision_path_conv4)
        
        return decision_path_conv4

class SSELayer(nn.Module):
    def __init__(self, channel):
        super(SSELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, bias=False, padding=[1, 1]),
            nn.Conv2d(channel, channel, kernel_size=3, bias=False, padding=[1, 1]),
            nn.Conv2d(channel, 1, kernel_size=3, bias=False, padding=[1, 1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y


class CSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SegmentPostProcessing(nn.Module):
    def __init__(self, spatial_dim=2):
        super(SegmentPostProcessing, self).__init__()
        model = GACNFuseNet()
        model = torch.compile(model)
        model.load_state_dict(torch.load('/home/anirudhan/project/image-fusion/results/checkpoints/model_60.pth')['model_state_dict'])
        for parm in model.parameters():
            parm.requires_grad = False

        self.segmentNN = model
        self.crf = CRF(n_spatial_dims=spatial_dim)
    
    def forward(self, image1, image2):
        x = self.segmentNN(image1, image2)
        return self.crf(x)
    
#%%:

# Create an instance of the SegmentFocus model
# model = SegmentPostProcessing()

# # # Create a dummy input
# image1 = torch.randn(1, 1, 256, 256)  # Assuming input image size is 256x256
# image2 = torch.randn(1, 1, 256, 256)

# # # Pass the dummy input through the model
# output = model(image1, image2)
# torch.compile()
# Print the output
# print(output)