import jax
import jax.numpy as jnp
import flax.linen as nn

class TimeEmbed(nn.Module):
    """
    Timestep embedding module
    """
    time_embed_dim: int
    max_period: int
    project_embed_dim: int
    embed_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, timesteps):
        # Calculate sinusodial embedding
        half = self.time_embed_dim // 2
        freqs = jnp.exp(-math.log(self.max_period) * jnp.arange(0, half) / half)
        args = timesteps[:, None].astype(self.embed_dtype) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis = -1)
        
        if self.time_embed_dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros(embedding[:, :1].shape)], axis = -1)
        
        # Some dense layers to properly embed for real
        embedding = nn.Dense(self.project_embed_dim, dtype = self.embed_dtype)(embedding)
        embedding = nn.silu(embedding)
        embedding = nn.Dense(self.project_embed_dim, dtype = self.embed_dtype)(embedding)
        
        return embedding

class TimestepBlock(nn.Module):
    """
    Interface for modules that can take timestep embeddings as input
    in addition to regular embeddings.
    """
    def __call__(self, x, emb):
        pass

class TimestepEmbedSequential(TimestepBlock):
    """
    Block that passes timestep embeddings to all submodules that
    want them.

    Also works as a regular sequential type module
    """
    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x, emb = None):
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class TimestepIdentity(TimestepBlock):
    """
    Identity block that is also a timestep block
    """
    @nn.compact
    def __call__(self, x, emb):
        return x
    
if __name__ == "__main__":
    # Basic function tests
    embed_test = TimeEmbed(16, 10000, 16 * 4)
    embed_in = jnp.zeros((10, 1))
    params = embed_test.init(prng, embed_in)
    out = embed_test.apply(*precision_policy.cast_to_compute([params, embed_in]))
    print(out.shape, out.dtype)
    
