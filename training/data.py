import jax
import jax.numpy as jnp

class TimestepSampler():
    """
    Sampler for timesteps
    """
    def __init__(self, timesteps, batch_size):
        self.timesteps = timesteps
        self.batch_size = batch_size
        
    def sample(self, prng):
        prng, key = jax.random.split(prng)
        
        weights = jnp.ones([self.timesteps])
        probs = weights / np.sum(weights)
        indices_sample = jax.random.choice(key, len(weights), (self.batch_size,), p = probs)
        weights_sample = 1.0 / (len(probs) * probs[indices_sample])
        
        return prng, indices_sample, weights_sample
    
class DataSampler():
    """
    Very basic data sampler that returns randomly picked batches from a base samples-first array
    
    Hope your dataset fits all in memory
    """
    def __init__(self, base_data, batch_size):
        self.base_data = base_data
        self.batch_size = batch_size
        pass
    
    def sample(self, prng):
        prng, key = jax.random.split(prng)
        indices_sample = jax.random.choice(key, len(self.base_data), (self.batch_size,))
        sample = self.base_data[indices_sample]
        return prng, sample
