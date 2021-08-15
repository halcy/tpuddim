import jax
import jax.numpy as jnp
import flax.linen as nn

class ConvND(nn.Module):
    """
    n-D convolution with square kernel
    """
    dims: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 'SAME'
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        self.kernel = tuple([self.kernel_size] * self.dims)
        self.strides = tuple([self.stride] * self.dims)
        if self.padding in ['SAME', 'VALID']:
            self.paddings = self.padding
        else:
            self.paddings = tuple([(self.padding, self.padding)] * self.dims)
        self.conv = nn.Conv(self.out_channels, self.kernel, self.strides, self.paddings, dtype=self.dtype)

    def __call__(self, x):
        return self.conv(x)

class AvgND(nn.Module):
    """
    n-D average pooling with square window

    Numeric padding is NOT supported, specify either SAME or VALID padding.
    """
    dims: int
    window_size: int = 2
    stride: int = 2
    padding: int = 'VALID'

    def setup(self):
        self.window = [1] + [self.window_size] * self.dims
        self.strides = [1] + [self.stride] * self.dims
        if self.padding in ['SAME', 'VALID']:
            self.paddings = self.padding

    def __call__(self, x):
        full_window = tuple(list(self.window) + list([1] * (len(x.shape) - len(self.window) - 1)))
        full_strides = tuple(list(self.strides) + list([1] * (len(x.shape) - len(self.window) - 1)))
        return nn.avg_pool(x, full_window, full_strides, self.paddings)


if __name__ == "__main__":
    # Basic function tests
    conv_in = jnp.zeros((10, 32, 32, 3))
    conv_test = ConvND(2, 64, 3)
    params = conv_test.init(prng, conv_in)
    out = conv_test.apply(*precision_policy.cast_to_compute([params, conv_in]))
    print(out.shape, out.dtype)
    
    avg_test = AvgND(2, 2, 2)
    params = avg_test.init(prng, conv_in)
    out = avg_test.apply(*precision_policy.cast_to_compute([params, conv_in]))
    print(out.shape, out.dtype)
