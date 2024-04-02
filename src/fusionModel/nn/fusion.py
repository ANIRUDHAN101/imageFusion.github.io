#%%
import jax
import jax.numpy as jnp
import einops
import jaxwt as jwt
from functools import partial
import flax.linen as nn
from typing import Any, Callable, Optional, Tuple, Type
#%%
Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

Conv3x3 = partial(nn.Conv, kernel_size=(3,3), padding='SAME')
Conv7x7 = partial(nn.Conv, kernel_size=(7,7), padding='SAME')
Conv1x1 = partial(nn.Conv, kernel_size=(1,1), padding='SAME')

rearange_width_first = lambda x: einops.rearrange(x, "n h w c -> n c h w")
rearange_height_first = lambda x: einops.rearrange(x, "n c h w -> n c w h")

x_to_patch = partial(einops.rearrange, pattern="n no_heads (h patch1) (w patch2) c -> n no_heads c h w patch1 patch2")
patch_to_x = partial(einops.rearrange, pattern="n no_heads c h w patch1 patch2 -> n no_heads (h patch1) (w patch2) c")

patch_to_fft = partial(einops.rearrange, pattern = "n no_heads c h w patch1 patch2 -> n no_heads h w patch1 patch2 c")
fft_to_patch = partial(einops.rearrange, pattern = "n no_heads h w patch1 patch2 c -> n no_heads c h w patch1 patch2")

def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw c)",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, patch_size, image_height, image_width):
    grid_height = image_height // patch_size[0]
    grid_width = image_width // patch_size[1]
    """patches to images."""
    x = einops.rearrange(
    x, "n (gh gw) (fh fw c )-> n (gh fh) (gw fw) c",
    gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x

class ScaleImage(nn.Module):
    """Scale image to the desired size( downsample or upsample).
    Attributes:
        scale_factor: the scale factor of the image.
    """
    scale_factor: float
    scale_channel_factor: int = 2
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = False):
        """Applies the ScaleImage module.
        Args:
            x: input image tensor.
        Returns:
            Output tensor with shape [batch_size, new_height, new_width, num_channels].
        """
        new_size = (
            x.shape[0],                             # batch size
            int(x.shape[1] * self.scale_factor),    # new height
            int(x.shape[2] * self.scale_factor),    # new width
            x.shape[3])                             # number of channels
        
        image = jax.image.resize(x, new_size, 'bilinear')
        image = Conv3x3(x.shape[-1], param_dtype=self.dtype)(image)
        image = Conv3x3(x.shape[-1], param_dtype=self.dtype)(image)
        image = nn.BatchNorm(use_running_average = not train)(image)
        # image = Conv3x3(x.shape[-1]//self.scale_channel_factor)(image)
        return image
    
class AddPositionalEmbs(nn.Module):
    """ Adds learnable positional embeddings to the inputs.
    
    Attributes:
        posemb_init: positioanl embedding initializer.
        param_dtype: the datatype of the positional embeddings.
    """
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        """Applies the AddPositionalEmbs module.

        Args:
            inputs: inputs to the layer .
        
        Returns:
            Output tensor with shape [batch_size, seq_length, in_dim].
        """
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pos_emb = self.param('pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
        return inputs + pos_emb

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(inputs) 
        x = nn.activation.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(x)
        
        output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)
        return output

class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""
    dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = Conv3x3(self.dim, param_dtype=self.param_dtype)(x)
        x = nn.activation.gelu(x)
        x = nn.BatchNorm(use_running_average = not train)(x)
        x = Conv3x3(self.dim, param_dtype=self.param_dtype)(x)
        x = nn.activation.gelu(x)
        x = nn.BatchNorm(use_running_average = not train)(x)
        return x

class FrequencyWeightedDotProduct(nn.Module):
    """Frequency weighted dot product attention."""
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x_patch_fft):
        x_patch_fft = self.param('fft projection wight', nn.initializers.xavier_normal(), x_patch_fft.shape, self.dtype) * (x_patch_fft)
        x_patch_fft = x_patch_fft + self.param('fft projection bias', nn.initializers.xavier_uniform(), x_patch_fft.shape, self.dtype) 
        x_patch_fft = nn.activation.gelu(x_patch_fft)
        return x_patch_fft
    
class LinearProjection(nn.Module):
    """Linear projection Layer

    Args:
        dim (int): The dimension of the output feature.
        no_heads (int, optional): The number of attention heads. If None, the output will not have a head dimension. Defaults to None.
        patch_size (int): The size of the patch.
        dtype (Dtype, optional): The data type of the layer's parameters. Defaults to jnp.float32.
        param_dtype (Dtype, optional): The data type of the layer's parameters. Defaults to jnp.float32.
        kernel_init (Callable[[PRNGKey, Shape, Dtype], Array], optional): The initializer for the kernel weights. Defaults to nn.initializers.xavier_uniform().
        bias_init (Callable[[PRNGKey, Shape, Dtype], Array], optional): The initializer for the bias weights. Defaults to nn.initializers.normal(stddev=1e-6).
        bias (bool, optional): Whether to include a bias term. Defaults to False.

    Returns:
        Array: The output tensor after linear projection. If no_heads is None, the shape will be [batch_size, height, width, channels]. Otherwise, the shape will be [batch_size, num_heads, height, width, channels].

    """

    dim: int
    patch_size: int
    no_heads: int = None
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)
    bias: bool = False

    @nn.compact
    def __call__(self, x):
        if self.no_heads is None:
            x = nn.Conv(
                features=self.dim*self.patch_size**2,
                kernel_size=(self.patch_size, self.patch_size),
                strides=(self.patch_size, self.patch_size),
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            )(x)
            x = einops.rearrange(x, 
                                 "b h w (c p0 p1)-> b (h p0) (w p1) c", 
                                 p0=self.patch_size, p1=self.patch_size,
                                 c = self.dim)
            return x
        
        x = nn.Conv(
            features=self.dim * self.no_heads*self.patch_size**2,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(x)
        x = einops.rearrange(x, 
                             "b h w (c p0 p1 head)-> b head (h p0) (w p1) c", 
                             p0=self.patch_size, p1=self.patch_size, 
                             c=self.dim,
                             head=self.no_heads)
        return x
    
class DFFN(nn.Module):
    dim :int 
    no_heads: int = 8
    patch_size :int = 8
    ffn_expansion_factor :int = 2
    bias :bool = False
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image, train: bool = False):
        # x of dimentiion [batch_size, height, width, channels] is linearly projected to
        # [batch_size, num_heads, height, width, channels]
        x  = LinearProjection(self.dim, self.no_heads, self.patch_size)(image)

        # for applying fft over the image the image is convrted to pathes and patch dimention is
        # [batch_size, num_heads, height, width, channels] ->
        # [batch_size, num_heads, channels,height, width, patch_size, patch_size]
        x_patch = x_to_patch(x, patch1=self.patch_size, patch2=self.patch_size, no_heads=self.no_heads)
        x_patch_fft = jnp.fft.rfft2(x_patch)

        # this layers extracts the necessary frequencies from the image
        x_patch_fft = FrequencyWeightedDotProduct(name='fft weighted dot product')(x_patch_fft)

        x_patch = jnp.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # [batch_size, num_heads, height, width, channels]
        x = patch_to_x(x_patch, patch1=self.patch_size, patch2=self.patch_size, no_heads = self.no_heads)
        x = nn.LayerNorm()(x)
        #x = nn.Dense(self.dim * self.ffn_expansion_factor)(x)
        x = einops.rearrange(x, "b head h w c -> b h w (head c)")
        x = LinearProjection(self.dim, no_heads=None, patch_size=self.patch_size)(x)
        x = nn.activation.gelu(x) + image
        
        return x
    
class FSAS(nn.Module):
    dim :int 
    no_heads: int = 8
    bias :bool = False
    patch_size :int = 8
    dropout_rate :float = 0.1
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = False):
        # q k and v of dimentiion [batch_size, height, width, channels] is linearly projected to
        # [batch_size, num_heads, height, width, channels]
        q = LinearProjection(name='q linear projection', dim=self.dim, no_heads=self.no_heads, patch_size=self.patch_size)(x)
        k = LinearProjection(name='k linear projection', dim=self.dim, no_heads=self.no_heads, patch_size=self.patch_size)(x)
        v = LinearProjection(name='v linear projection', dim=self.dim, no_heads=self.no_heads, patch_size=self.patch_size)(x)
        
        # for applying fft over the image the image is convrted to pathes and patch dimention is
        # [batch_size, num_heads, height, width, channels] ->
        # [batch_size, num_heads, channels,height, width, patch_size, patch_size]
        q = x_to_patch(q, patch1=self.patch_size, patch2=self.patch_size, no_heads=self.no_heads, )
        q_fft = jnp.fft.rfft2(q)
        # q_fft = patch_to_fft(q_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)
        
        k = x_to_patch(k, patch1=self.patch_size, patch2=self.patch_size, no_heads=self.no_heads)
        k_fft = jnp.fft.rfft2(k)
        # k_fft = patch_to_fft(k_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)
        
        # perform corelation in the frequency domain
        out = q_fft * k_fft
        # out = fft_to_patch(out, patch1=self.patch_size, patch2=self.patch_size//2+1)
        out = jnp.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        
        # convert the image back to the original dimention 
        # [batch_size, num_heads, height, width, channels]
        out = patch_to_x(out)
        out = nn.LayerNorm()(out)
        
        output = v * out

        output = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(output)

        # convert the output to dimentions [batch_size, height, width, channels * no_heads] and apply
        # linear projection to get the output of dimentions [batch_size, height, width, channels]
        output = einops.rearrange(output, "b head h w c -> b h w (head c)")
        output = LinearProjection(name='output linear projection', dim=self.dim, no_heads=None, patch_size=self.patch_size)(output)
        
        return output
    
class FrequencyTransformer(nn.Module):
    dim: int
    no_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Dtype = jnp.float32
    patch_size: int = 8
    encoder: bool = True
    mlp:bool = False

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        # Attention block
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = FSAS(
            dim=self.dim,
            no_heads=self.no_heads,
            dropout_rate=self.attention_dropout_rate,
            patch_size=self.patch_size,
            dtype=self.dtype
        )(x, train=train)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        if x.shape[-1] != inputs.shape[-1]:
            inputs = Conv1x1(x.shape[-1])(inputs)
        x = x + inputs

        # if using as a decoder
        if not self.encoder:
            x_ff = nn.LayerNorm(dtype=self.dtype)(x)
            x_ff = DFFN(dim=self.dim)(x, train=train)
            x_ff = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            x = x + x_ff
        
        # MLP block
        if self.mlp:
            y = block_images_einops(x, patch_size =  [self.patch_size, self.patch_size])
            y = nn.LayerNorm(dtype=self.dtype)(y)
            y = MlpBlock(
                self.dim,
                dtype=self.dtype,   
                dropout_rate=self.dropout_rate
            )(y, train)
            y = unblock_images_einops(y, [self.patch_size, self.patch_size], image_height=x.shape[-3], image_width=x.shape[-2])
            x = x + y

        return x
    
class FrequencyFusionBlock(nn.Module):
    dim: int
    bias: bool = False
    no_heads: int = 0
    patch_size: int = 8
    dtype: Dtype = jnp.float32
    
    @nn.compact
    def __call__(self, image1, image2, train=False):

        q1 = LinearProjection(self.dim, no_heads=self.no_heads, patch_size=self.patch_size, dtype=self.dtype)(image1)
        k1 = LinearProjection(self.dim, no_heads=self.no_heads, patch_size=self.patch_size, dtype=self.dtype)(image1)
        v1 = LinearProjection(self.dim, no_heads=self.no_heads, patch_size=self.patch_size, dtype=self.dtype)(image1)

        q2 = LinearProjection(self.dim, no_heads=self.no_heads, patch_size=self.patch_size, dtype=self.dtype)(image2)
        k2 = LinearProjection(self.dim, no_heads=self.no_heads, patch_size=self.patch_size, dtype=self.dtype)(image2)
        v2 = LinearProjection(self.dim, no_heads=self.no_heads, patch_size=self.patch_size, dtype=self.dtype)(image2)


        q1_patch = x_to_patch(q1, patch1=self.patch_size, patch2=self.patch_size, no_heads=self.no_heads)
        k1_patch = x_to_patch(k1, patch1=self.patch_size, patch2=self.patch_size, no_heads=self.no_heads)

        q2_patch = x_to_patch(q2, patch1=self.patch_size, patch2=self.patch_size, no_heads=self.no_heads)
        k2_patch = x_to_patch(k2, patch1=self.patch_size, patch2=self.patch_size, no_heads=self.no_heads)

        q1_fft = jnp.fft.rfft2(q1_patch)
        # q1_fft = patch_to_fft(q1_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        k1_fft = jnp.fft.rfft2(k1_patch)
        # k1_fft = patch_to_fft(k1_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        q2_fft = jnp.fft.rfft2(q2_patch)
        # q2_fft = patch_to_fft(q2_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        k2_fft = jnp.fft.rfft2(k2_patch)
        # k2_fft = patch_to_fft(k2_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        # frequency filter takes the necessary freqnecies before corelation
        q1_patch = FrequencyWeightedDotProduct(name='fft weighted dot product q1')(q1_fft)
        k1_patch = FrequencyWeightedDotProduct(name='fft weighted dot product k1')(k1_fft)
        q2_patch = FrequencyWeightedDotProduct(name='fft weighted dot product q2')(q2_fft)
        k2_patch = FrequencyWeightedDotProduct(name='fft weighted dot product k2')(k2_fft)

        out1 = q1_patch * k2_patch
        out2 = q2_patch * k1_patch
        
        # out1 = fft_to_patch(out1, patch1=self.patch_size, patch2=self.patch_size//2+1)
        out1 = jnp.fft.irfft2(out1, s=(self.patch_size, self.patch_size))
        out1 = patch_to_x(out1)

        # out2 = fft_to_patch(out2, patch1=self.patch_size, patch2=self.patch_size//2+1)
        out2 = jnp.fft.irfft2(out2, s=(self.patch_size, self.patch_size))
        out2 = patch_to_x(out2)

        out1 = nn.LayerNorm()(out1)
        out2 = nn.LayerNorm()(out2)
        
        out1 = einops.rearrange(out1, "b head h w c -> b h w (head c)")
        out2 = einops.rearrange(out2, "b head h w c -> b h w (head c)")
        out1 = LinearProjection(self.no_heads, no_heads=None, patch_size=self.patch_size, dtype=self.dtype)(out1)
        out2 = LinearProjection(self.no_heads, no_heads=None, patch_size=self.patch_size, dtype=self.dtype)(out2)
        out1 = einops.rearrange(out1, "b h w (head c) -> b head h w c", head=self.no_heads, c=1)
        out2 = einops.rearrange(out2, "b h w (head c) -> b head h w c", head=self.no_heads, c=1)
        
        out = jnp.concatenate([out1, out2], axis=-1)
        out = nn.activation.softmax(out, axis=-1)

        corelation1, corelation2 = jnp.split(Conv1x1(v1.shape[-1] + v2.shape[-1], use_bias=False)(out), 2, axis=-1)
        fused_features = v1 * corelation1 + v2 * corelation2
        
        fused_features = einops.rearrange(fused_features, "b head h w c -> b h w (head c)")
        fused_features = LinearProjection(3, no_heads=None, patch_size=self.patch_size, dtype=self.dtype)(fused_features)
        
        return fused_features

class FusionBlock(nn.Module):
    dim: int
    no_heads: int
    levels: int = 3
    patch_size: int = 8
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, feature1, feature2, train=False):
        x = Conv1x1(
            features=feature1.shape[-1]*2,
            name='conv1x1',
            param_dtype=self.dtype
            )(jnp.concatenate([feature1, feature2], axis=-1))\
        
        x = nn.LayerNorm(name='layer_norm')(x)
        x = FrequencyTransformer(
            dim=self.dim,
            no_heads=self.no_heads,
            patch_size=self.patch_size,
            encoder=False,
            mlp=True,
            dtype=self.dtype
        )(x, train)
        x = Conv1x1(
            features=self.dim*2,
            name='conv1x1_2',
            param_dtype=self.dtype
        )(x)

        a, b = jnp.split(x, 2, axis=-1)
        a = nn.activation.gelu(a)
        output = a + b

        return output

class EncoderFusionBlock(nn.Module):
    dim: int
    no_heads: int
    levels: int = 3
    patch_size: int = 8
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, image, train=False):
        frequency_transformer_hypers = {
            'dim': self.dim,
            'no_heads': self.no_heads,
            'patch_size': self.patch_size,
            'dtype': self.dtype,
            'encoder': True,
        }
       
        for i in range(self.levels-1):
            print(i)
            image = FrequencyTransformer(
                name=f"encoder {i}",
                mlp=False,
                **frequency_transformer_hypers
            )(image)

        image = FrequencyTransformer(
            name=f"encoder {self.levels-1}",
            mlp=True,
            **frequency_transformer_hypers,
        )(image)
        return image

class DencoderFusionBlock(nn.Module):
    dim: int
    no_heads: int
    levels: int = 3
    patch_size: int = 8
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, image, train=False):
        frequency_transformer_hypers = {
            'dim': self.dim,
            'no_heads': self.no_heads,
            'patch_size': self.patch_size,
            'dtype': self.dtype,
            'encoder': False,
        }

        for i in range(self.levels-1):
            image = FrequencyTransformer(
                name=f"encoder {i}",
                mlp=False,
                **frequency_transformer_hypers
            )(image)
        image = FrequencyTransformer(
            name=f"encoder {self.levels-1}",
            mlp=True,
            **frequency_transformer_hypers,
        )(image)
        return image
    
class ImageFusion(nn.Module):
    dim: int 
    no_heads: int 
    stack_levels: int = 3
    downsampling_levels: int = 3
    patch_size: int = 4
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image1, image2, train=False):
        
        image_features = [None]*self.downsampling_levels
        encoder = [None]*self.downsampling_levels
        encoder_fusion = [None]*self.downsampling_levels

        transformer_hypers = {
            'dim': self.dim,
            'no_heads': self.no_heads,
            'patch_size': self.patch_size,
            'dtype': self.dtype,
        }

        # Encoder part
        for i in range(self.downsampling_levels):
            # extract featues from the images and fuse the featues from the images
            encoder[i] = EncoderFusionBlock(
                name=f'encoder block level_{i}',
                levels=self.stack_levels,
                **transformer_hypers
            )
            image1_feature = encoder[i](image1, train)
            image2_feature = encoder[i](image2, train)

            if image_features[i-1] is not None:
                encoder_fusion[i] = FusionBlock(name=f'encoder feature fusion _{i}', **transformer_hypers)
                downscaled_feature = ScaleImage(0.5, dtype=self.dtype)(image_features[i-1])
                image1_feature = encoder_fusion[i](downscaled_feature, image1_feature, train)
                image2_feature = encoder_fusion[i](downscaled_feature, image2_feature, train)

            fused_features = FrequencyFusionBlock(
                name=f'fusion block level_{i}',
                **transformer_hypers
            )(image1_feature, image2_feature, train)

            image_features[i] = fused_features

            # downsample the images by 2
            image1 = ScaleImage(0.5, dtype=self.dtype)(image1, train=train)
            image2 = ScaleImage(0.5, dtype=self.dtype)(image2, train=train)

        # Bottlenext part
        image_features[-1] = EncoderFusionBlock(
            name='bottleck layer',
            levels=self.stack_levels*2,
            **transformer_hypers
        )(image_features[-1], train)
    
        # Decoder part
        for i in reversed(range(1, self.downsampling_levels)):
            image_features[i] = ScaleImage(2, dtype=self.dtype)(image_features[i], train=train)
            image_features[i-1] = FusionBlock(
                name=f'decoder fusion block level_{i}',
                **transformer_hypers
            )(image_features[i-1], image_features[i], train)

            image_features[i-1] = DencoderFusionBlock(
                name=f'decoder level_{i}',
               **transformer_hypers
            )(image_features[i-1], train)

        fused_image = Conv1x1(3)(image_features[0])
        return fused_image
        

dffn = ImageFusion(3, 16)
rng = jax.random.PRNGKey(0)
state = dffn.init(rng, jnp.ones((1, 512, 512, 3)), jnp.ones((1, 512, 512, 3)))    
#state = create_train_state(rng, config, DFFN(3), learning_rate_fn=None)
#%%
tabulate_fn = nn.tabulate(
    dffn, rng, compute_flops=True, compute_vjp_flops=True)
x = jnp.ones((32, 512, 512, 3))
print(tabulate_fn(x, x))
#%%
# class ConvBlock(nn.Module):
#     dimention : int

#     @nn.compact
#     def __call__(self, image, train=False):
#         feature = Conv3x3(self.dimention)(image)
#         feature = nn.activation.gelu(feature)
#         feature = nn.BatchNorm(use_running_average = not train)(feature)
#         return feature

# class FusionModel(nn.Module):
#     @nn.compact
#     def __call__(self, image1, image2, train=False):
#         feature1 = ConvBlock(16)(image1, train)
#         feature2 = ConvBlock(16)(image2, train)
#         return feature1 + feature2
    
# workdir = '/home/anirudhan/project/fusion/results'
# # checkpoint_manager = create_checkpoints_manager(config, '/content')
# config = get_default_configs()

# rng = jax.random.PRNGKey(0)

# writer = metric_writers.create_default_writer(
#       logdir=workdir, just_logging=jax.process_index() != 0
#   )

# if config.batch_size % jax.device_count() > 0:
#     raise ValueError('Batch size must be divisible by the number of devices')
# local_batch_size = config.batch_size // jax.process_count()
# print(rng)
# #%%
# state = create_train_state(rng, config, FusionModel(), learning_rate_fn=None)