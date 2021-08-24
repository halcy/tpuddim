import jax
import jax.numpy as jnp

class EMA():
    """
    Class that will apply an EMA to parameters.
    
    Doesn't actually keep state, keep that in your train loop and pass it as a parameter.
    """
    def __init__(self, mu = 0.9999):
        self.mu = mu

    def apply(self, params, params_prev = None):
        if params_prev is None:
            return params
        else:
            params_prev, _ = jax.tree_util.tree_flatten(params_prev)
            
        params, treedef = jax.tree_util.tree_flatten(params)
        
        params_new = []
        for i in range(len(params)):
            params[i] = (1. - self.mu) * params_prev[i] + self.mu * jax.lax.stop_gradient(params[i])
            
        params = jax.tree_util.tree_unflatten(treedef, params)
        return params
