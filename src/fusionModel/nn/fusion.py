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
x_to_patch = partial(einops.rearrange, pattern="n (h patch1) (w patch2) (c heads) -> n heads c h w patch1 patch2")
patch_to_x = partial(einops.rearrange, pattern="n heads c h w patch1 patch2 -> n (h patch1) (w patch2) (c heads)")
patch_to_fft = partial(einops.rearrange, pattern = "n heads c h w patch1 patch2 -> n heads h w patch1 patch2 c")
fft_to_patch = partial(einops.rearrange, pattern = "n heads h w patch1 patch2 c -> n heads c h w patch1 patch2")

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

    @nn.compact
    def __call__(self, x):
        """Applies the ScaleImage module.
        Args:
            x: input image tensor.
        Returns:
            Output tensor with shape [batch_size, new_height, new_width, num_channels].
        """
        new_size = (int(x.shape[1] * self.scale_factor), int(x.shape[2] * self.scale_factor))
        image = nn.image.resize(x, new_size, 'bilinear')
        image = Conv3x3(x.shape[-1])(image)
        image = Conv3x3(x.shape[-1]//self.scale_channel_factor)(image)
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
    def __call__(self, inputs, train: bool = True):
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
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(x)
        
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=not train)
        return output


class DFFN(nn.Module):
    dim :int 
    heads: int = 8
    patch_size :int = 8
    ffn_expansion_factor :int = 2
    bias :bool = False
    @nn.compact
    def __call__(self, x):
        
        hidden_features = self.ffn_expansion_factor * self.dim
        x = Conv1x1(hidden_features * 2, use_bias=self.bias)(x)
        x_patch = x_to_patch(x, patch1=self.patch_size, patch2=self.patch_size)
        x_patch_fft = jnp.fft.rfft2(x_patch)
        x_patch_fft = patch_to_fft(x_patch_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)
        x_patch_fft = nn.Dense(hidden_features * 2)(x_patch_fft)
        x_patch_fft = fft_to_patch(x_patch_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)
        x_patch = jnp.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = patch_to_x(x_patch, patch1=self.patch_size, patch2=self.patch_size)
        x1, x2 = nn.Conv(hidden_features *2 , kernel_size=(3,3), padding='SAME', feature_group_count=2)(x).split(2, axis=-1)
        x = nn.activation.gelu(x1) * x2
        x = Conv1x1(self.dim)(x)
        return x
    
class FSAS(nn.Module):
    dim :int 
    heads: int = 8
    bias :bool = False
    patch_size :int = 8
    dropout_rate :float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = False):
        hidden = Conv1x1(self.dim * 6 * self.heads, use_bias=self.bias)(x)
        q, k, v = Conv3x3(self.dim * 6 * self.heads, use_bias=self.bias)(hidden).split(3, axis=-1)
        
        q_patch = x_to_patch(q, patch1=self.patch_size, patch2=self.patch_size, heads=self.heads)
        k_patch = x_to_patch(k, patch1=self.patch_size, patch2=self.patch_size, heads=self.heads)
        
        q_fft = jnp.fft.rfft2(q_patch)
        q_fft = patch_to_fft(q_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)
        
        k_fft = jnp.fft.rfft2(k_patch)
        k_fft = patch_to_fft(k_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)
        
        out = q_fft * k_fft
        out = fft_to_patch(out, patch1=self.patch_size, patch2=self.patch_size//2+1)
        out = jnp.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = patch_to_x(out)
        out = nn.LayerNorm()(out)
        
        output = v * out
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=not train)
        
        output = Conv1x1(self.dim, use_bias=self.bias)(output)
        
        return output
    
class FrequencyTransformer(nn.Module):
    dim: int
    n_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    dtype: Dtype = jnp.float32
    patch_size: int = 8
    encoder: bool = True

    @nn.compact
    def __call__(self, inputs, train: bool = False):
        # Attention block
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = FSAS(
            dim=self.dim,
            heads=self.n_heads,
            dropout_rate=self.attention_dropout_rate,
            patch_size=self.patch_size
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        x = x + inputs

        # if using as a decoder
        if not self.encoder:
            x_ff = nn.LayerNorm(dtype=self.dtype)(x)
            x_ff = DFFN(dim=self.dim)(x)
            x_ff = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
            x = x + x_ff

        # MLP block
        y = block_images_einops(x, patch_size =  [self.patch_size, self.patch_size])
        y = nn.LayerNorm(dtype=self.dtype)(y)
        y = MlpBlock(
            self.dim,
            dtype=self.dtype,   
            dropout_rate=self.dropout_rate
        )(y, train)
        y = unblock_images_einops(y, [self.patch_size, self.patch_size], image_height=x.shape[-3], image_width=x.shape[-2])
        return x + y
    
class FrequencyFusionBlock(nn.Module):
    hidden_dim: int
    bias: bool = False
    heads: int = 0
    patch_size: int = 8

    @nn.compact
    def __call__(self, image1, image2, train=False):
        hidden_features1 = Conv1x1(self.hidden_dim * 6, use_bias=self.bias)(image1)
        hidden_features2 = Conv1x1(self.hidden_dim * 6, use_bias=self.bias)(image2)

        q1, k1, v1 = Conv3x3(self.hidden_dim * 6, use_bias=self.bias)(hidden_features1).split(3, axis=-1)
        q2, k2, v2 = Conv3x3(self.hidden_dim * 6, use_bias=self.bias)(hidden_features2).split(3, axis=-1)

        q1_patch = x_to_patch(q1, patch1=self.patch_size, patch2=self.patch_size, heads=self.heads)
        k1_patch = x_to_patch(k1, patch1=self.patch_size, patch2=self.patch_size, heads=self.heads)

        q2_patch = x_to_patch(q2, patch1=self.patch_size, patch2=self.patch_size, heads=self.heads)
        k2_patch = x_to_patch(k2, patch1=self.patch_size, patch2=self.patch_size, heads=self.heads)

        q1_fft = jnp.fft.rfft2(q1_patch)
        q1_fft = patch_to_fft(q1_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        k1_fft = jnp.fft.rfft2(k1_patch)
        k1_fft = patch_to_fft(k1_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        q2_fft = jnp.fft.rfft2(q2_patch)
        q2_fft = patch_to_fft(q2_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        k2_fft = jnp.fft.rfft2(k2_patch)
        k2_fft = patch_to_fft(k2_fft, patch1=self.patch_size, patch2=self.patch_size//2+1)

        # frequency filter takes the necessary freqnecies before corelation
        q1_fft = nn.Dense(self.hidden_dim * 2)(q1_fft)
        k1_fft = nn.Dense(self.hidden_dim * 2)(k1_fft)
        q2_fft = nn.Dense(self.hidden_dim * 2)(q2_fft)
        k2_fft = nn.Dense(self.hidden_dim * 2)(k2_fft)

        out1 = q1_fft * k2_fft
        out2 = q2_fft * k1_fft
        
        out1 = fft_to_patch(out1, patch1=self.patch_size, patch2=self.patch_size//2+1)
        out1 = jnp.fft.irfft2(out1, s=(self.patch_size, self.patch_size))
        out1 = patch_to_x(out1)

        out2 = fft_to_patch(out2, patch1=self.patch_size, patch2=self.patch_size//2+1)
        out2 = jnp.fft.irfft2(out2, s=(self.patch_size, self.patch_size))
        out2 = patch_to_x(out2)

        out1 = nn.LayerNorm()(out1)
        out2 = nn.LayerNorm()(out2)

        out1 = Conv1x1(1)(out1)
        out2 = Conv1x1(1)(out2)

        out = jnp.stack([out1, out2], axis=-1)
        out = nn.activation.softmax(out, axis=-1)

        fused_features = v1 * out[:,:,:,0] + v2 * out[:,:,:,1]

        return fused_features


dffn = FrequencyTransformer(3, 16)
rng = jax.random.PRNGKey(0)
state = dffn.init(rng, jnp.ones((1, 512, 512, 3)))    
#state = create_train_state(rng, config, DFFN(3), learning_rate_fn=None)
#%%
tabulate_fn = nn.tabulate(
    Encoder(3, 8, patch_size=8), rng)
print(tabulate_fn(x))
      #%%
class ConvBlock(nn.Module):
    dimention : int

    @nn.compact
    def __call__(self, image, train=False):
        feature = Conv3x3(self.dimention)(image)
        feature = nn.activation.gelu(feature)
        feature = nn.BatchNorm(use_running_average = not train)(feature)
        return feature

class FusionModel(nn.Module):
    @nn.compact
    def __call__(self, image1, image2, train=False):
        feature1 = ConvBlock(16)(image1, train)
        feature2 = ConvBlock(16)(image2, train)
        return feature1 + feature2
    
workdir = '/home/anirudhan/project/fusion/results'
# checkpoint_manager = create_checkpoints_manager(config, '/content')
config = get_default_configs()

rng = jax.random.PRNGKey(0)

writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
local_batch_size = config.batch_size // jax.process_count()
print(rng)
#%%
state = create_train_state(rng, config, FusionModel(), learning_rate_fn=None)