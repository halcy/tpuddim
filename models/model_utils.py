import pickle

from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit, PartitionSpec

def get_default_channel_mult(image_size):
    """
    Helper function for the channel multipliers from the OpenAI guided diffusion model.
    """
    channel_mult = None
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    return channel_mult

def load_params(path, devices, precision_policy = None):
    """
    Load and shard/replicate parameters
    """
    with open(path, "rb") as f:
        params_load = pickle.load(f)
    
    if precision_policy != None:
        # Convert to desired precision
        params_load = precision_policy.cast_to_compute(params_load)
        
    # Replicate loaded parameters
    with mesh(devices, ('batch', 'x', 'y')):
        params = pjit(lambda x: x,  PartitionSpec(None),  PartitionSpec(None))(params_load)        
        
    return params
