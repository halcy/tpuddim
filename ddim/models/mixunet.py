from typing import Any, Sequence

import einops
import flax.linen as nn
import jax.numpy as jnp
import math

from .time_embed import TimeEmbed, TimestepEmbedSequential, TimestepBlock
from .convolutional import ConvND
from .resblock import ResBlock
from .mixer import MlpMixerBlock
from .attention import SpatialSelfAttentionBlock

class MixUNet(nn.Module):
    """
    Unet style model with mixer layers
    """
    model_channels: int
    channel_mult: int
    use_scale_shift_norm: bool
    num_head_channels: int
    num_res_blocks: int
    attention_resolutions: Sequence[int]
    out_channels: int
    mixer_config: dict
    dims: int = 2
    dropout: float = 0.0
    
    dtype: jnp.dtype = jnp.bfloat16
    dtype_out: jnp.dtype = jnp.float32
    
    def setup(self):
        # Timestep embedding
        time_embed_dim = self.model_channels * 4
        self.time_embed = TimeEmbed(self.model_channels, 10000, time_embed_dim, embed_dtype=self.dtype)

        # Initial block for input stack
        input_block_out_channels = int(self.channel_mult[0] * self.model_channels)
        input_blocks = [TimestepEmbedSequential([ConvND(self.dims, input_block_out_channels, 3, padding = 1, dtype=self.dtype)])]

        # Loop to create rest of input stack
        current_channels = input_ch = int(self.channel_mult[0] * self.model_channels)
        input_block_chans = [current_channels]
        downsample_fact = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                # One res block
                layers = [ResBlock(self.dims, current_channels, int(mult * self.model_channels), dropout = self.dropout, use_scale_shift_norm = self.use_scale_shift_norm, dtype=self.dtype)]
                needs_train = [True]
                current_channels = int(mult * self.model_channels)

                # One mixer block, if requested
                if downsample_fact in self.attention_resolutions:
                    layers.append(MlpMixerBlock(**self.mixer_config))
                    needs_train.append(False)
                # Put those in sequence
                input_blocks.append(TimestepEmbedSequential(layers, needs_train))
                input_block_chans.append(current_channels)

            # Downsample if not the final block
            if level != len(self.channel_mult) - 1:
                input_blocks.append(ResBlock(self.dims, current_channels, current_channels, dropout = self.dropout, use_scale_shift_norm = self.use_scale_shift_norm, down = True, dtype=self.dtype))
                input_block_chans.append(current_channels)
                downsample_fact *= 2
                
        self.input_blocks = input_blocks

        # Middle block
        self.middle_block = TimestepEmbedSequential([
            ResBlock(self.dims, current_channels, current_channels, self.dropout, use_scale_shift_norm = self.use_scale_shift_norm, dtype=self.dtype),
            #MlpMixerBlock(**self.mixer_config),
            SpatialSelfAttentionBlock(self.num_head_channels, dtype=self.dtype),
            ResBlock(self.dims, current_channels, current_channels, self.dropout, use_scale_shift_norm = self.use_scale_shift_norm, dtype=self.dtype),
        ], [True, False, True])

        # Output blocks
        output_blocks = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                # One res block, with skip input from same unet level
                skip_channels = input_block_chans.pop()
                in_channels = current_channels + skip_channels
                layers = [ResBlock(self.dims, in_channels, int(self.model_channels * mult), dropout = self.dropout, use_scale_shift_norm = self.use_scale_shift_norm, dtype=self.dtype)]
                needs_train = [True]
                current_channels = int(self.model_channels * mult)

                # One mixer block, if requested
                if downsample_fact in self.attention_resolutions:
                    layers.append(MlpMixerBlock(**self.mixer_config))
                    needs_train.append(False)
                    
                # Upsample, if not the final block
                if level != 0 and i == self.num_res_blocks:
                    out_ch = current_channels
                    layers.append(ResBlock(self.dims, current_channels, current_channels, dropout = self.dropout, use_scale_shift_norm = self.use_scale_shift_norm, up = True, dtype=self.dtype))
                    needs_train.append(True)
                    downsample_fact //= 2
                output_blocks.append(TimestepEmbedSequential(layers, needs_train))
        self.output_blocks = output_blocks

        # Final output block
        self.out = TimestepEmbedSequential([
            nn.GroupNorm(epsilon=1e-05, dtype=self.dtype),
            nn.silu,
            ConvND(self.dims, self.out_channels, 3, dtype=self.dtype),
        ])

    # Left / input side of fotward pass
    def forward_in(self, x, t):
        emb = self.time_embed(t)

        h = x
        hs = []
        for block in self.input_blocks:
            h = block(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)
        return h, emb, hs
        
    # Right / output side of forward pass
    def forward_out(self, h, emb, hs):
        for i in range(len(self.output_blocks)):
            h = jnp.concatenate([h, hs[len(self.output_blocks) - i - 1]], axis = -1)
            h = self.output_blocks[i](h, emb)
        h = self.out(h)
        return h

    # Full forward pass
    def __call__(self, x, t, train = False):
        emb = self.time_embed(t)

        h = x
        hs = []
        for block in self.input_blocks:
            h = block(h, emb, train)
            hs.append(h)

        h = self.middle_block(h, emb)

        for block in self.output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis = -1)
            h = block(h, emb, train)

        h = self.out(h)
        return h.astype(self.dtype_out)

if __name__ == "__main__":
    from model_utils import get_default_channel_mult
    
    # Lets see if it works, by instantiating parameters for a 256x256 model that matches OpenAIs    
    unet = UNet(
        dims = 2,
        model_channels = 256,
        channel_mult = get_default_channel_mult(256),
        use_scale_shift_norm = True,
        dropout = 0.0,
        num_head_channels = 64,
        num_res_blocks = 2,
        attention_resolutions = (32, 16, 8),
        out_channels = 6
    )
    image_in = jnp.zeros((1, 256, 256, 3))
    embed_in = jnp.zeros((1,))

    # To instantiate a new model (for training, or loading pytorch params, or renaming parameters after jit-ing parts of the model)
    params = unet.init(prng, image_in, embed_in)

    # Whole network function test - commented out because slow
    out = unet.apply(*precision_policy.cast_to_compute([params, image_in]), embed_in)
    print(out.shape, out.dtype)
    
