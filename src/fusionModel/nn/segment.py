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
from .guided_filter import GuidedFilter
from typing import Optional, Tuple, Type
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
                features = f.interpolate(features, size=image.shape[2:], mode='bilinear')
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
        # bimap = torch.nn.functional.softmax(10 * (f1_sf - f2_sf), dim=1)
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
            self.upsample = nn.Upsample(scale_factor=8, mode=upsample)

        self.dropout = nn.Dropout(0.3)

    def forward(self, spatial_fused_features: torch.Tensor, prev_features: torch.Tensor):
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

        if prev_features is not None:
            if spatial_fused_features.shape[2:] != prev_features.shape[2:]:
                spatial_fused_features = f.interpolate(spatial_fused_features, size=prev_features.shape[2:], mode='bilinear')
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
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
        )

        self.conv_block2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.GELU()
        )

        self.conv_residual1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.conv_residual2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

        self.cse = CSELayer(out_channels, reduction=2)

    def forward(self, x):
        x1 = self.conv_block1(x)
        x = self.conv_residual1(x) + x1
        x2 = self.conv_block2(x)
        x = self.conv_residual2(x) + x2
        x = self.cse(x)
        return x

class CrossConvBlock(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_sizes=[3, 5, 7, 11, 15, 17, 21], stride=1, padding=1, bias=False):
        super(CrossConvBlock, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.conv_layers = nn.ModuleDict()
        self.out_channels = len(kernel_sizes)
        for kernel_size in kernel_sizes:
            convs = []
            # its an Arithamtic progression (3 +2n) , so to get no of consicutive 3x3 conv n = (kernel_size - 3 )/ 2
            for i in range(int((kernel_size-3) / 2)-1):
                convs.append(nn.Conv2d(in_channels, in_channels, 3, stride, padding, bias=bias))
            convs.append(nn.Conv2d(in_channels, 1, 3, stride, padding, bias=bias))

            self.conv_layers[str(kernel_size)] = nn.Sequential(
                *convs,
                nn.BatchNorm2d(1),
            )

        self.cse = CSELayer(self.out_channels, reduction=2)
        self.output_conv = nn.Conv2d(self.out_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x1, x2):
        featues = []
        for kernel_size in self.kernel_sizes:
            featues.append(self.conv_layers[str(kernel_size)](x1) - self.conv_layers[str(kernel_size)](x2))

        x1_features = torch.cat(featues, dim=1)
        x2_features = -x1_features

        x1_features = nn.Sigmoid()(x1_features)
        x2_features = nn.Sigmoid()(x2_features)

        x1_features = self.cse(x1_features)
        x2_features = self.cse(x2_features)

        x1_features = self.output_conv(x1_features)
        x2_features = self.output_conv(x2_features)

        return x1_features, x2_features
    
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, channel):
        super(FeatureExtractor, self).__init__()
        self.depth = len(channel) - 1
        self.conv_blocks = nn.ModuleList()
        self.corss_conv_blocks = nn.ModuleList()
        self.conv_residuals = nn.ModuleList()

        for i in range(self.depth):
            self.conv_blocks.append(
                nn.Sequential(
                    ConvBlock(in_channels if i==0 else channel[i], channel[i+1]),
                    # SSELayer(channel[i+1])
                ))

            self.corss_conv_blocks.append(
                CrossConvBlock(channel[i+1], channel[i+1])
            )
            self.conv_residuals.append(
                nn.Conv2d(channel[i], channel[i+1], kernel_size=1, stride=1, padding=0, bias=False)
            )
        
    def forward(self, image1_input_features, image2_input_features):
        for i in range(self.depth):
            image1_features = self.conv_blocks[i](image1_input_features)
            image2_features = self.conv_blocks[i](image2_input_features)
            # shortct connection
            image1_input_features = self.conv_residuals[i](image1_input_features) + image1_features
            image2_input_features = self.conv_residuals[i](image2_input_features) + image2_features

            image1_coss_weight, image2_coss_weight = self.corss_conv_blocks[i](image1_input_features, image2_input_features)
            image1_input_features = image1_input_features * image1_coss_weight
            image2_input_features = image2_input_features * image2_coss_weight

        return image1_input_features, image2_input_features
        
class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, image1_features, image2_features):
        if image1_features.shape[2:] != image2_features.shape[2:]:
            image1_features = self.upsample(image1_features)
        
        x = torch.cat([image1_features, image2_features], dim=1)
        x = self.conv_block(x)
        return x

class SegmentFocus(nn.Module):
    def __init__(self, channel: Tuple[int] = (4, 8, 16, 8, 3)):
        super(SegmentFocus, self).__init__()      
        self.depth = len(channel) - 1
        self.xfm = DWTForward(J=1, mode='reflect', wave='haar')
        
        self.feature_extractor = FeatureExtractor(channel)
        self.fusion_layer = FusionBlock(channel[-1]+1, channel[-1]) 
        
        # self.graysacle_to_channel = nn.Conv2d(1, channel[-1], kernel_size=1)

        self.output_conv_final = nn.Conv2d(channel[-1]*2, 3, kernel_size=3, padding=1)
        
        self.self_guideding_filter_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=1),
        )

        self.guided_filter = GuidedFilter(10, 0.1)

    def forward(self, image1, image2, gt_image, gt_mask):
        image1_gray = rgb_to_grayscale(image1)
        image2_gray = rgb_to_grayscale(image2)
    
        ll_image1, dwt_image1 = self.xfm(image1_gray)
        ll_image2, dwt_image2 = self.xfm(image2_gray)

        image1_input_features = torch.cat([ll_image1, dwt_image1[0][:, 0, :, :, :]], dim=1)
        image2_input_features = torch.cat([ll_image2, dwt_image2[0][:, 0, :, :, :]], dim=1)

        image1_input_features, image2_input_features= self.feature_extractor(image1_input_features, image2_input_features)
        
        image1_input_features = self.fusion_layer(image1_input_features, image1_gray)
        image2_input_features = self.fusion_layer(image2_input_features, image2_gray)

        final_fused_features = torch.cat([image1_input_features, image2_input_features], dim=1)
        focus_regions = self.output_conv_final(final_fused_features)

        # fusion of features
        focus_regions = f.softmax(focus_regions, dim=1)
        g = self.self_guideding_filter_layer(focus_regions)
        focus_regions = self.guided_filter(g, focus_regions)
        focus_regions = torch.clamp(focus_regions, 0, 1)

        masked_gt_image = gt_image*gt_mask[:,1:2]
        
        fused_image = focus_regions[:,0:1]*image1 + focus_regions[:,1:2]*masked_gt_image + focus_regions[:,2:3]*image2
        return fused_image, focus_regions.data, None
        # return fused_image, focus_regions.data, (list(map(lambda x: x.data, image1_features)))

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
    
#%%
# Create an instance of the SegmentFocus model
# model = SegmentFocus().cuda()

# # Create a dummy input
# image1 = torch.randn(1, 3, 256, 256).cuda()  # Assuming input image size is 256x256
# image2 = torch.randn(1, 3, 256, 256).cuda()
# gt_image = torch.randn(1, 3, 256, 256).cuda()
# gt_mask = torch.randn(1, 3, 256, 256).cuda()
# # Pass the dummy input through the model
# output = model(image1, image2, gt_image, gt_mask)

# # Print the output
# print(output)
# # %%
