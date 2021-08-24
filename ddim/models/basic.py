import jax
import jax.numpy as jnp
import flax.linen as nn

class Identity(nn.Module):
    """
    For model building convenience, a class that does nothing
    """
    @nn.compact
    def __call__(self, x):
        return x
    
