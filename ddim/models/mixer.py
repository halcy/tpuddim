# Largely based on code by Google LLC, licensed under the apache license
# https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

from typing import Any, Sequence

import einops
import flax.linen as nn
import jax.numpy as jnp
from ddim.models.time_embed import TimeEmbed
import math

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
        x = PositionEncoder(name="pos embed")(x)
        
        # To patches
        x = nn.Conv(self.hidden_dim, self.patches, strides=self.patches, name='patch')(x)
        _, patch_h, patch_w, _ = x.shape
        x = einops.rearrange(x, 'n h w c -> n (h w) c')
        
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
        
        # Back up to original shape
        x = x.reshape(b, patch_h, patch_w, -1)
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
            x = UpdownMixerBlockWithTime(hidden_dim, self.patches, self.tokens_mlp_dim, self.channels_mlp_dim, name=f"block{i}")(x, t)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
            
        # Convolutional output block - just get down to channel count
        x = nn.Conv(self.out_channels, (1, 1), strides = (1, 1))(x)
        
        return x
    