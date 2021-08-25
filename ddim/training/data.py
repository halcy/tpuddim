import jax
import jax.numpy as jnp
import numpy as np
import glob
import PIL

from functools import partial

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
        probs = weights / jnp.sum(weights)
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
    
    def sample(self, prng):
        prng, key = jax.random.split(prng)
        indices_sample = jax.random.choice(key, len(self.base_data), (self.batch_size,))
        sample = self.base_data[indices_sample]
        return prng, sample

class ImageDataset():
    """
    Basic image dataset. TODO: Augmentations, caching
    """
    def __init__(self, file_glob, resize = None):
        self.file_list = list(sorted(list(glob.glob(file_glob))))
        self.resize = resize
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):
        image = PIL.Image.open(self.file_list[i])
        if not self.resize is None:
            image.thumbnail((self.resize, self.resize), PIL.Image.ANTIALIAS)
        return jnp.array(image)    

class DatasetSampler():
    """
    Very basic data sampler that returns randomly picked batches from a Dataset class
    """
    def __init__(self, dataset, batch_size, norm_fact = 127.5, norm_sub = 1.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shape = tuple([batch_size] + list(dataset[0].shape))
        self.norm_fact = norm_fact
        self.norm_sub = norm_sub
        self.sample_arr = np.zeros(self.shape)
    
    def sample(self, prng):
        prng, key = jax.random.split(prng)
        indices_sample = jax.random.choice(key, len(self.dataset), (self.batch_size,))
        for i, index in enumerate(indices_sample):
            self.sample_arr[i] = self.dataset[index]
        return prng, (jnp.array(self.sample_arr) / self.norm_fact) - self.norm_sub
