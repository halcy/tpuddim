import jax
import jax.numpy as jnp
import flax.linen as nn

from convolutional import ConvND

class SpatialSelfAttentionBlock(nn.Module):
    """
    Dot-product attention block for spatial dimensions
    """
    head_channels: int = 64
    dtype: jnp.dtype = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x):
        # Flatten out spatial channels and norm
        batches = x.shape[0]
        channels = x.shape[-1]
        x_in = x.reshape((batches, -1, channels))
        qkv = nn.GroupNorm(epsilon=1e-05, dtype=self.dtype)(x_in)

        # Convolve to three times the amount of channels and split into query, key and value
        qkv = ConvND(1, channels * 3, 1, dtype=self.dtype)(qkv)

        # Calculate number of heads and split into heads
        heads = int(channels / self.head_channels)
        qkv = qkv.reshape(batches, -1, heads, self.head_channels * 3)

        # Split into query/key/value
        query = qkv[:, :, :, 0:self.head_channels] 
        key = qkv[:, :, :, self.head_channels:self.head_channels*2]
        value = qkv[:, :, :, self.head_channels*2:]

        # Calculate dot product attention and flatten out
        x_out = nn.dot_product_attention(query, key, value, deterministic=True, dtype=self.dtype)
        x_out = x_out.reshape((batches, -1, channels))

        # Project
        x_out = ConvND(1, channels, 1, dtype=self.dtype)(x_out)

        # Resicual and reshape back to original shape
        return (x_in + x_out).reshape(x.shape)
    
if __name__ == "__main__":
    # Basic function tests
    attention = SpatialSelfAttentionBlock(64)
    attention_in = jnp.zeros((10, 20, 30, 256))
    params = attention.init(prng, attention_in)
    out = attention.apply(*precision_policy.cast_to_compute([params, attention_in]))
    print(out.shape, out.dtype)
