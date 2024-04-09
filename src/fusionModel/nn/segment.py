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
#%%
class EncoderBLock(nn.Module):
    """
    This class implements a basic feature extractor block with residual connections 
    and optional downsampling.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        depth (int): Number of residual blocks within the encoder block.
        downsample (bool, optional): Whether to perform downsampling. Defaults to True.
    """

    def __init__(self, in_channels, out_channels, depth, downsample=True):
        super(EncoderBLock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.input_conv_block = nn.Sequential(
            nn.LazyConv2d(self.out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU()
        )

        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            conv_block = nn.Sequential(
                nn.LazyConv2d(self.out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1,bias=False, groups=out_channels),
                nn.BatchNorm2d(self.out_channels),
                nn.GELU(),
                CSELayer(self.out_channels, reduction=2)
            )
            self.layers.append(conv_block)

        if downsample:
            self.downsample = nn.MaxPool2d(2)

        self.dropout = nn.Dropout(0.3)

    def forward(self, image, features=None):
        if hasattr(self, 'downsample'):
            image = self.downsample(image)

        # Handle varying input feature dimensions
        if features is not None:
            if features.shape[2:] != image.shape[2:]:
                # Implement logic for handling different feature dimensions (e.g., separate convolution branch)
                raise ValueError("Input feature and image shapes do not match.")
            x = torch.cat([features, image], dim=1)
        else:
            x = image

        x = self.input_conv_block(x)
        residual = [x]

        for i in range(self.depth):
            x = self.layers[i](x)
            x += residual.pop()
            residual.append(x)
        x = self.dropout(x)
        return x

class SpatialFrequency(nn.Module):

    def forward(self, f1, f2, kernel_radius=5):
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
        bimap = torch.nn.functional.softmax(f1_sf - f2_sf, dim=1)
        return bimap
    
class DecoderBlock(nn.Module):
    
    def __init__(self, input_dim : int, dim :int, depth :int, upsample: str = 'bilinear') -> None:
        """
        Initialize the DecoderBlock.
        
        Args:
            input_dim (int): The input dimension.
            dim (int): The dimension of the block.
            depth (int): The depth of the block.
            upsample (str, optional): The type of upsampling. Defaults to 'bilinear'.
        """
        
        super(DecoderBlock, self).__init__()
        self.dim = dim
        self.depth = depth

        self.spatial_freaquency_extractor = SpatialFrequency()
        
        self.conv_block = nn.Sequential(
                nn.LazyConv2d(self.dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.dim),
                nn.GELU(),
                nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, bias=False, groups=dim//2),
                nn.BatchNorm2d(self.dim),
                nn.GELU(),
                SSELayer(self.dim)
            )
        
        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            conv_block = nn.Sequential(
                nn.LazyConv2d(self.dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.dim),
                nn.GELU(),
                nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=dim//2, bias=False),
                nn.BatchNorm2d(self.dim),
                nn.GELU(),
                SSELayer(self.dim)
            )
            self.layers.append(conv_block)

        self.normalize = nn.Sequential(
                nn.LazyConv2d(self.dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.dim),
            )

        if upsample == 'convTranspose':
            self.upsample = nn.ConvTranspose2d(self.dim, self.dim, kernel_size=(2, 2), stride=(2, 2), bias=False)

        elif upsample is not None:
            self.upsample = nn.Upsample(scale_factor=2, mode=upsample)

        self.dropout = nn.Dropout(0.3)

    def forward(self, image1_features: torch.Tensor, image2_features: torch.Tensor, prev_features: torch.Tensor):
        """
        Perform forward pass through the decoder block.
        
        Args:
            image1_features (torch.Tensor): Features from image 1.
            image2_features (torch.Tensor): Features from image 2.
            prev_features (torch.Tensor): Features from previous block.
        
        Returns:
            torch.Tensor: Output features.
        """
        # image1_features = self.normalize(image1_features)
        # image2_features = self.normalize(image2_features)

        spatial_fused_features = self.spatial_freaquency_extractor(image1_features, image2_features)

        if prev_features is not None:
            assert spatial_fused_features.shape[2:] == prev_features.shape[2:], "Spatial fused features and previous features do not have the same shape."
            spatial_fused_features = torch.cat([spatial_fused_features, prev_features], dim=1)

        spatial_fused_features = self.conv_block(spatial_fused_features)

        if hasattr(self, 'upsample'):
            spatial_fused_features = self.upsample(spatial_fused_features)

        residual = [spatial_fused_features]
        for i in range(self.depth):
            spatial_fused_features = self.layers[i](spatial_fused_features)
            spatial_fused_features += residual.pop()
            residual.append(spatial_fused_features)

        del residual
        spatial_fused_features = self.dropout(spatial_fused_features)

        return spatial_fused_features
    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()

        self.aspp_blocks = nn.ModuleList()

        for dilation_rate in dilation_rates:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation_rate, dilation=dilation_rate,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            self.aspp_blocks.append(block)

        # Global context branch with 1x1 convolution
        self.global_context = nn.Sequential(
            nn.Conv2d(out_channels*len(dilation_rates), out_channels, 1, bias=False),  # 1x1 convolution for global context
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = torch.cat([block(x) for block in self.aspp_blocks], dim=1)
        x = torch.cat([x, self.global_context(x)], dim=1)  # Concatenate with global context
        return x

class SegmentFocus(nn.Module):
    def __init__(self, feature_dim: Sequence[int], depth :int):
        super(SegmentFocus, self).__init__()      
        self.feature_dim = feature_dim
        self.depth = depth
        self.xfm = DWTForward(J=len(feature_dim), mode='zero', wave='haar')

        # input channel is set to one the encoder takes images as grayscale images 
        self.initial_feature_extractor = EncoderBLock(in_channels = 1, 
                                                      out_channels=feature_dim[0]*2,
                                                      depth=2,
                                                      downsample=False
                                                      )
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        self.aspp = ASPP(in_channels=feature_dim[0], out_channels=feature_dim[0], dilation_rates=[2, 4, 8])

        for i in range(1, len(feature_dim)):
            self.encoder_blocks.append(EncoderBLock(in_channels = feature_dim[i-1], 
                                                    out_channels=feature_dim[i],
                                                    depth=depth
                                                    )
                                        )
            
        self.decoder_blocks.append(DecoderBlock(input_dim = feature_dim[0],
                                                dim = feature_dim[0],
                                                depth = depth,
                                                upsample=None
                                                )
                                    )
        for i in range(1, len(feature_dim)):
            self.decoder_blocks.append(DecoderBlock(input_dim = feature_dim[i],
                                                    dim = feature_dim[i-1],
                                                    depth = depth
                                                    )
                                        )
        # self.output_conv_map = nn.Sequential(
        #     nn.Conv2d(feature_dim[0], 1, kernel_size=3, padding=1),
        # )

        self.output_conv_f = nn.Sequential(
            nn.Conv2d(feature_dim[0], feature_dim[0], kernel_size=3, padding=1, bias=False, groups=feature_dim[0]),
            nn.Conv2d(feature_dim[0], feature_dim[0], kernel_size=3, padding=1, bias=False, groups=feature_dim[0]),
            nn.Conv2d(feature_dim[0], feature_dim[0], kernel_size=3, padding=1, groups=feature_dim[0]),
            nn.BatchNorm2d(feature_dim[0]),
            nn.ReLU(),
            nn.Conv2d(feature_dim[0], 3, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )

        # self.output_conv_d = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=1, padding=0),
        #     nn.Sigmoid()
        # )


        
    def forward(self, image1, image2):
        image1_gray = rgb_to_grayscale(image1)
        image2_gray = rgb_to_grayscale(image2)
        
        image1_initial_features = self.initial_feature_extractor(image1_gray)
        image2_initial_features = self.initial_feature_extractor(image2_gray)


        image1_features = [image1_initial_features[:, :self.feature_dim[0]]]
        image2_features = [image2_initial_features[:, :self.feature_dim[0]]]

        # image1_features[0] = self.aspp(image1_features[0])
        # image2_features[0] = self.aspp(image2_features[0])

        _, dwt_image1 = self.xfm(image1)
        _, dwt_image2 = self.xfm(image2)
        
        for i in range(len(self.feature_dim)-1):
            image1_features.append(self.encoder_blocks[i](image1_features[i], dwt_image1[i][:, 0, :, :, :]))
            image2_features.append(self.encoder_blocks[i](image2_features[i], dwt_image2[i][:, 0, :, :, :]))
        
        fused_features = None

        for i in reversed(range(len(self.feature_dim)-1)):
            fused_features = self.decoder_blocks[i+1](image1_features[i+1], image2_features[i+1], fused_features)
        
        # to extract global features from images using aspp
        # image1_features = self.aspp(image1_initial_features[:, self.feature_dim[0]:])
        # image2_features = self.aspp(image1_initial_features[:, self.feature_dim[0]:])

        fused_features = torch.cat((image1_initial_features[:, self.feature_dim[0]:], image1_initial_features[:, self.feature_dim[0]:]), dim=1)
        fused_features = self.decoder_blocks[0](image1_features[0], image2_features[0], fused_features)

        # fused_features = self.output_conv_map(fused_features) 

        focus_regions = self.output_conv_f(fused_features)
        
        # diffusion_regions = self.output_conv_d(fused_features)

        # diffusion_regions = diffusion_regions*(1-focus_regions) + diffusion_regions*focus_regions
        
        # fused_image = focus_regions*image1 + (1-focus_regions)*image2
        # focus_regions[:,2] = 1 - focus_regions[:,0] - focus_regions[:,1]
        return focus_regions

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
        self.se_4 = SSELayer(1)
        self.se_5 = SSELayer(48)
        self.se_6 = SSELayer(32)
        self.se_7 = SSELayer(16)
        self.se_8 = SSELayer(1)
        self.decision_path_conv1 = self.conv_block(1, 48, name="decision_path_conv1")
        self.decision_path_conv2 = self.conv_block(48, 32, name="decision_path_conv2")
        self.decision_path_conv3 = self.conv_block(32, 16, name="decision_path_conv3")
        self.decision_path_conv4 = self.conv_block(16, 1,  name="decision_path_conv4")
        self.squeeze_features = self.conv_block(64, 1, name="feature_squeeze")
        # self.guided_filter = GuidedFilter(3, 0.1)
        # self.gaussian = GaussBlur(8, 4)
   
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
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        b_shift_kernel = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])\
            .cuda(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_r_shift = f.conv2d(f1, r_shift_kernel, padding=1, groups=c)
        f1_b_shift = f.conv2d(f1, b_shift_kernel, padding=1, groups=c)
        f2_r_shift = f.conv2d(f2, r_shift_kernel, padding=1, groups=c)
        f2_b_shift = f.conv2d(f2, b_shift_kernel, padding=1, groups=c)
        f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
        f2_grad = torch.pow((f2_r_shift - f2), 2) + torch.pow((f2_b_shift - f2), 2)
        kernel_size = kernel_radius * 2 + 1
        add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().cuda(device)
        kernel_padding = kernel_size // 2
        f1_sf = f.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c)
        f2_sf = f.conv2d(f2_grad, add_kernel, padding=kernel_padding, groups=c)

        # get decision map
        bimap = torch.sigmoid(1000 * (f1_sf - f2_sf))
        return bimap
    
    def forward(self, img1, img2):
        """
        Train or Forward for two images
        :param img1: torch.Tensor
        :param img2: torch.Tensor
        :return: output, torch.Tensor
        """
        # Feature extraction c1
        img1 = rgb_to_grayscale(img1)
        img2 = rgb_to_grayscale(img2)

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
        cat_1 = self.squeeze_features(cat_1)
        cat_2 = self.squeeze_features(cat_2)

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
        se_decision_path_conv4 = self.se_8(decision_path_conv4)
        
        # Boundary guided filter
        output_origin = torch.sigmoid(1000 * se_decision_path_conv4)
        
        return output_origin


class SSELayer(nn.Module):
    def __init__(self, channel):
        super(SSELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=7, bias=False, padding=[3, 3]),
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
    
#%%
# Create an instance of the SegmentFocus model
model = SegmentFocus(feature_dim=[16, 16], depth=2)

# Create a dummy input
image1 = torch.randn(1, 3, 256, 256)  # Assuming input image size is 256x256
image2 = torch.randn(1, 3, 256, 256)

# Pass the dummy input through the model
output = model(image1, image2)

# Print the output
print(output)