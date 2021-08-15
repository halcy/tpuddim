import jax
import jax.numpy as jnp
import flax.linen as nn

from time_embed import TimestepBlock, TimestepEmbedSequential
from resampling import Upsample, Downsample
from basic import Identity

class ResBlock(TimestepBlock):
    """
    Residual block with timesteps
    """
    dims: int
    in_channels: int
    out_channels: int
    dropout: float
    use_conv: bool = False
    up: bool = False
    down: bool = False
    use_scale_shift_norm: bool = False
    dtype: jnp.dtype = jnp.bfloat16
    
    def setup(self):
        # Initial normalization block
        self.in_block = TimestepEmbedSequential((
            nn.GroupNorm(epsilon=1e-05, dtype=self.dtype),
            nn.silu,
        ))

        # Up/Downsampling block
        if self.up:
            self.h_upd = Upsample(self.dims, dtype=self.dtype)
            self.x_upd = Upsample(self.dims, dtype=self.dtype)
        elif self.down:
            self.h_upd = Downsample(self.dims, dtype=self.dtype)
            self.x_upd = Downsample(self.dims, dtype=self.dtype)
        else:
            self.h_upd = self.x_upd = Identity()

        # Input convolution
        self.in_conv = ConvND(self.dims, self.out_channels, 3, dtype=self.dtype)

        # Embedding projection block
        self.embed_project = TimestepEmbedSequential((
            nn.silu,
            nn.Dense(2 * self.out_channels if self.use_scale_shift_norm else self.out_channels, dtype=self.dtype)
        ))

        # Actual layer stack
        self.out_norm = nn.GroupNorm(epsilon=1e-05, dtype=self.dtype)
        self.out_layers = TimestepEmbedSequential((
            nn.silu,
            nn.Dropout(self.dropout, deterministic=True), # TODO: This should be dynamic, for training. Right now it's a noop.
            ConvND(self.dims, self.out_channels, 3, dtype=self.dtype) # There was a zero initializer (?) here, it's gone now, sorry. Maybe not important.
        ))

        # Channel change for skip connection
        if self.out_channels == self.in_channels:
            self.skip_connection = Identity()
        elif self.use_conv:
            self.skip_connection = ConvND(self.dims, self.out_channels, 3, dtype=self.dtype)
        else:
            self.skip_connection = ConvND(self.dims, self.out_channels, 1, dtype=self.dtype)

    def __call__(self, x, emb):
        # For residual: Resample x
        x_res = self.x_upd(x)

        # Run x through input & up/downsample block
        x = self.in_block(x)
        x = self.h_upd(x)
        x = self.in_conv(x)
            
        # Project embedding and unsqueeze up to match shape of x
        emb_out = self.embed_project(emb.reshape(emb.shape[0], -1))
        emb_out = emb_out.reshape((emb_out.shape[0],) + tuple([1] * (len(x.shape) - len(emb_out.shape))) + (emb_out.shape[-1],))
        
        # Apply actual convolution
        if self.use_scale_shift_norm:
            scale = emb_out[..., :self.out_channels]
            shift = emb_out[..., self.out_channels:]
            x = self.out_norm(x) * (1 + scale) + shift
            x = self.out_layers(x)
        else:
            x = x + emb_out
            x = self.out_norm(x)
            x = self.out_layers(x)

        # Return residual
        return self.skip_connection(x_res) + x
    
    
if __name__ == "__main__":
    # Basic function tests
    conv = ResBlock(2, 64, 128, .5, up=True, use_conv=False, use_scale_shift_norm=True)
    conv_in = jnp.zeros((10, 20, 20, 64))
    params = conv.init(prng, conv_in, conv_in)
    out = conv.apply(*precision_policy.cast_to_compute([params, conv_in, conv_in]))
    print(out.shape, out.dtype)
