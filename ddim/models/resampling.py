import jax
import jax.numpy as jnp
import flax.linen as nn

from .convolutional import ConvND, AvgND
from .basic import Identity

class Scale2X(nn.Module):
    """
    nD 2x nearest neighbour scaling
    """
    dims: int

    @nn.compact
    def __call__(self, x):
        for i in range(1, self.dims + 1):
            x = jnp.repeat(x, 2, axis = i)
        return x

class Upsample(nn.Module):
    """
    Upsampling layer, factor 2

    Nearest neighour (element repeating), optionally with a convolution afterwards.
    """
    dims: int
    out_channels: int = 0
    use_conv: bool = False
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        self.scale = Scale2X(self.dims)
        if self.use_conv:
            assert self.out_channels != 0
            self.conv = ConvND(self.dims, self.out_channels, 3, dtype=self.dtype)
        else:
            self.conv = Identity()

    def __call__(self, x):
        x = self.scale(x)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    Downsampling layer, factor 2

    Uses either average pooling (default) or a strided convolution.
    """
    dims: int
    out_channels: int = 0
    use_conv: bool = False
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        if self.use_conv:
            assert self.out_channels != 0
            self.downsample = ConvND(self.dims, self.out_channels, 3, stride=2, padding=1, dtype=self.dtype)
        else:
            self.downsample = AvgND(self.dims)

    def __call__(self, x):
        x = self.downsample(x)
        return x
    
if __name__ == "__main__":
    # Basic function tests
    conv_in = jnp.zeros((10, 32, 32, 3))

    up_test = Upsample(2)
    params = up_test.init(prng, conv_in)
    out = up_test.apply(*precision_policy.cast_to_compute([params, conv_in]))
    print(out.shape, out.dtype)
    
    down_test = Downsample(2)
    params = down_test.init(prng, conv_in)
    out = down_test.apply(*precision_policy.cast_to_compute([params, conv_in]))
    print(out.shape, out.dtype)
