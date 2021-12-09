# Largely based on code by Google LLC, licensed under the apache license
# https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

from typing import Any, Sequence

import einops
import flax.linen as nn
import jax.numpy as jnp
import math

from .time_embed import TimeEmbed, TimestepEmbedSequential, TimestepBlock
from .convolutional import ConvND
from .resblock import ResBlock

from typing import Sequence

class MlpBlock(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    """Mixer block layer."""
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)

class MlpMixer(nn.Module):
    """
    Mixer architecture.
    Modified to be a denoising net in a DDIM model. Injects position at start and timestep at regular intervals.
    """
    patches: Any
    num_blocks: int
    timestep_inject_to: int
    timestep_inject_every: int
    hidden_per_channel: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    out_stage_dim_per_channel: int
    out_stage_kernel_dim: int
    out_channels: int = 3
    
    # TODO: These are ignored
    dtype: Any = jnp.bfloat16
    dtype_out: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, timesteps):
        b, h, w, c = x.shape
        
        # Hidden dim is in terms of channels
        hidden_dim = c * self.hidden_per_channel + 2
        
        # Prepare timestep embedding
        time_embed = nn.silu(TimeEmbed(hidden_dim * 4, 10000, hidden_dim)(timesteps))
        time_embed = jnp.expand_dims(time_embed, 1)
        time_embed = jnp.expand_dims(time_embed, 1)
        
        # Simple positional embedding
        pos_embed_h = jnp.repeat(jnp.expand_dims(jnp.sin(jnp.linspace(-math.pi / 2.0, math.pi / 2.0, h)), -1), w, axis = -1)
        pos_embed_w = jnp.repeat(jnp.expand_dims(jnp.sin(jnp.linspace(-math.pi / 2.0, math.pi / 2.0, w)), 0), h, axis = 0)
        pos_embed_h = jnp.expand_dims(jnp.repeat(jnp.expand_dims(pos_embed_h, 0), b, axis = 0), -1)
        pos_embed_w = jnp.expand_dims(jnp.repeat(jnp.expand_dims(pos_embed_w, 0), b, axis = 0), -1)
        pos_embed = jnp.concatenate((pos_embed_h, pos_embed_w), axis = -1)
        x = jnp.concatenate((x, pos_embed), axis = -1)
        
        # Actual MLP Mixer
        x = nn.Conv(hidden_dim, self.patches, strides=self.patches, name='stem')(x)
        _, patch_h, patch_w, _ = x.shape
        x = einops.rearrange(x, 'n h w c -> n (h w) c')
        for i in range(self.num_blocks):
            if i <= self.timestep_inject_to and i % self.timestep_inject_every == 0:
                x = x + nn.gelu(nn.Dense(hidden_dim)(time_embed))
            print(self.tokens_mlp_dim, self.channels_mlp_dim, x.shape)
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
        
        # Output block is convolutional, sorry
        x = x.reshape(b, patch_h, patch_w, -1)
        x = nn.gelu(nn.ConvTranspose(c * self.out_stage_dim_per_channel, self.patches, strides=self.patches)(x))
        x = nn.gelu(nn.Conv(c * self.out_stage_dim_per_channel, (self.out_stage_kernel_dim, self.out_stage_kernel_dim), strides = (1, 1))(x))
        x = nn.Conv(self.out_channels, (self.out_stage_kernel_dim, self.out_stage_kernel_dim), strides = (1, 1))(x)
        
        return x
    
class PositionEncoder(nn.Module):
    # Simple position embedding
    
    @nn.compact    
    def __call__(self, x):
        b, h, w, c = x.shape
        pos_embed_h = jnp.repeat(jnp.expand_dims(jnp.sin(jnp.linspace(-math.pi / 2.0, math.pi / 2.0, h)), -1), w, axis = -1)
        pos_embed_w = jnp.repeat(jnp.expand_dims(jnp.sin(jnp.linspace(-math.pi / 2.0, math.pi / 2.0, w)), 0), h, axis = 0)
        pos_embed_h = jnp.expand_dims(jnp.repeat(jnp.expand_dims(pos_embed_h, 0), b, axis = 0), -1)
        pos_embed_w = jnp.expand_dims(jnp.repeat(jnp.expand_dims(pos_embed_w, 0), b, axis = 0), -1)
        pos_embed = jnp.concatenate((pos_embed_h, pos_embed_w), axis = -1)
        x = jnp.concatenate((x, pos_embed), axis = -1)
        return x    
    
class PositionAwareMixerBlock(nn.Module):
    """Mixer block layer, but now it knows what patch it is working with"""
    tokens_mlp_dim: int
    channels_mlp_dim: int
    patch_h: int
    patch_w: int
    
    @nn.compact
    def __call__(self, x):
        y = x.reshape(x.shape[0], self.patch_h, self.patch_w, -1)
        y = PositionEncoder()(y)
        y = y.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        
        y = nn.LayerNorm()(y)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
        y = jnp.swapaxes(y, 1, 2)
        #x = x + y # hack, bad, make the pos embedding saner instead
        y = nn.LayerNorm()(x) 
        return x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)    

class UpdownMixerBlockWithTime(nn.Module):
    """Pos embed -> Patches -> Mixer block -> unpatch"""
    hidden_dim: int
    patches: Sequence[int]
    tokens_mlp_dim: int
    channels_mlp_dim: int
    
    @nn.compact
    def __call__(self, x, t):
        b, h, w, c = x.shape
        
        # Add position channels
        #x = PositionEncoder(name="pos embed")(x)
        #print("in", x.shape)
        
        # To patches
        x = nn.Conv(self.hidden_dim, self.patches, strides=self.patches, name='patch')(x)
        _, patch_h, patch_w, _ = x.shape
        x = einops.rearrange(x, 'n h w c -> n (h w) c')
        #print("before", x.shape)
        
        # Add time embedding
        x = x + t
        
        # Mix
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        
        y = nn.LayerNorm()(x) 
        x = x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)
        
        #print("after", x.shape)
        
        # Back up to original shape
        x = x.reshape(b, patch_h, patch_w, -1)
        #print("conv", x.shape)
        
        x = nn.ConvTranspose(c, self.patches, strides=self.patches, name='unpatch')(x)
        
        return x    
    
class UpdownMlpMixer(nn.Module):
    """
    Mixer architecture, modified to be a denoising net in a DDIM model.
    This version is supposed to inject position into every block, but instead just doesn't work whatsoever.
    """
    patches: Sequence[int]
    num_blocks: int
    timestep_inject_to: int
    timestep_inject_every: int
    hidden_per_channel: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    out_channels: int = 3
    
    # TODO: These are ignored
    dtype: Any = jnp.bfloat16
    dtype_out: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, timesteps):
        b, h, w, c = x.shape
        
        # Hidden dim is in terms of channels
        hidden_dim = c * self.hidden_per_channel
        
        # Prepare timestep embedding
        time_embed = nn.silu(TimeEmbed(hidden_dim * 4, 10000, hidden_dim)(timesteps))
        time_embed = jnp.expand_dims(time_embed, 1)
        time_embed = jnp.expand_dims(time_embed, 1)
        
        for i in range(self.num_blocks):
            if i <= self.timestep_inject_to and i % self.timestep_inject_every == 0:
                t = time_embed
            else:
                t = jnp.zeros(time_embed.shape)
            #print("pre", x.shape)
            x = UpdownMixerBlockWithTime(hidden_dim, self.patches, self.tokens_mlp_dim, self.channels_mlp_dim, name=f"block{i}")(x, t)
            #print("post", x.shape)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
            
        # Convolutional output block - just get down to channel count
        x = nn.Conv(self.out_channels, (1, 1), strides = (1, 1))(x)
        
        return x
    
    
    import jax

mixer_hyper = {
    "patches": (8, 8),
    "num_blocks": 3,
    "timestep_inject_every": 1,
    "timestep_inject_to": 2,
    "hidden_per_channel": 1,
    "tokens_mlp_dim": 128,
    "channels_mlp_dim": 512,
    "out_stage_dim_per_channel": 1,
}    


class MlpMixerBlock(TimestepBlock):
    """
    Mixer architecture.
    Modified to be a denoising net in a DDIM model. Injects position at start and timestep at regular intervals.
    """
    patches: Any
    num_blocks: int
    timestep_inject_to: int
    timestep_inject_every: int
    hidden_per_channel: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    out_stage_dim_per_channel: int
    
    # TODO: These are ignored
    dtype: Any = jnp.bfloat16
    dtype_out: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x, time_embed):
        #print("in", x.shape)
        b, h, w, c = x.shape
        
        # Hidden dim is in terms of channels
        hidden_dim = c * self.hidden_per_channel + 2
        
        # Simple positional embedding
        pos_embed_h = jnp.repeat(jnp.expand_dims(jnp.sin(jnp.linspace(-math.pi / 2.0, math.pi / 2.0, h)), -1), w, axis = -1)
        pos_embed_w = jnp.repeat(jnp.expand_dims(jnp.sin(jnp.linspace(-math.pi / 2.0, math.pi / 2.0, w)), 0), h, axis = 0)
        pos_embed_h = jnp.expand_dims(jnp.repeat(jnp.expand_dims(pos_embed_h, 0), b, axis = 0), -1)
        pos_embed_w = jnp.expand_dims(jnp.repeat(jnp.expand_dims(pos_embed_w, 0), b, axis = 0), -1)
        pos_embed = jnp.concatenate((pos_embed_h, pos_embed_w), axis = -1)
        x = jnp.concatenate((x, pos_embed), axis = -1)
        
        # Actual MLP Mixer
        #print(x.shape)
        x = nn.Conv(hidden_dim, self.patches, strides=self.patches, name='stem')(x)
        _, patch_h, patch_w, _ = x.shape
        x = einops.rearrange(x, 'n h w c -> n (h w) c')
        for i in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
        
        # Back to proper shape
        x = x.reshape(b, patch_h, patch_w, -1)
        x = nn.gelu(nn.ConvTranspose(c * self.out_stage_dim_per_channel, self.patches, strides=self.patches)(x))
        x = nn.Conv(c, (1, 1), strides = (1, 1))(x)
        #print("out", x.shape)
        return x

class MixNet(nn.Module):
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
                #print(downsample_fact)
                if downsample_fact in self.attention_resolutions:
                    layers.append(MlpMixerBlock(**mixer_hyper))
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
            MlpMixerBlock(**mixer_hyper),
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
                    layers.append(MlpMixerBlock(**mixer_hyper))
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
    
