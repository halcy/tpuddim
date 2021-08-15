import jax
import jax.numpy as jnp

def p_xstart(model, params, diff_params, images_in, timesteps_in, rescale_fact = 1, learned_variance = True):
    """
    Get predicted previous diffusion step from model
    """
    # Skip over the whole variance and model mean part for a lean and flexible, 
    # probably-could-be-more-ideal-but-whatever DDIM only model
    timesteps_scaled = timesteps_in // rescale_fact
    model_output = model.apply(params, images_in, timesteps_in)
    
    if learned_variance:
        image_output = model_output[:, :, :, :3]
        #var_values = model_output[:, :, :, 3:]
    else:
        image_output = model_output
        
    #min_log = diff_params["posterior_log_variance_clipped"][timesteps_scaled]
    #max_log = diff_params["betas"][timesteps_scaled]
    sqrt_recip_alphas_cumprod_step =  diff_params["sqrt_recip_alphas_cumprod"][timesteps_scaled]
    sqrt_recipm1_alphas_cumprod_step = diff_params["sqrt_recipm1_alphas_cumprod"][timesteps_scaled]
    
    #frac = (var_values + 1) / 2
    #model_log_variance = frac * max_log + (1 - frac) * min_log
    #model_variance = jnp.exp(model_log_variance)
    
    pred_xstart = sqrt_recip_alphas_cumprod_step * images_in - sqrt_recipm1_alphas_cumprod_step * image_output
    pred_xstart = jax.lax.clamp(-1.0, pred_xstart, 1.0)
    
    return pred_xstart

def ddim_sample(model, params, diff_params, images_in, timesteps_in, rescale_fact = 1, dtype_out = jnp.float32):
    """
    Perform one ddim sampling step
    """
    timesteps_scaled = timesteps_in // rescale_fact    
    out = p_xstart(model, params, diff_params, images_in, timesteps_in, rescale_fact = rescale_fact)
    
    sqrt_recip_alphas_cumprod_step = diff_params["sqrt_recip_alphas_cumprod"][timesteps_scaled]
    sqrt_recipm1_alphas_cumprod_step = diff_params["sqrt_recipm1_alphas_cumprod"][timesteps_scaled]
    eps = (sqrt_recip_alphas_cumprod_step * images_in - out) / sqrt_recipm1_alphas_cumprod_step

    alpha_bar = diff_params["alphas_cumprod"][timesteps_scaled]
    alpha_bar_prev = diff_params["alphas_cumprod_prev"][timesteps_scaled]
    mean_pred = (out * jnp.sqrt(alpha_bar_prev) + jnp.sqrt(1 - alpha_bar_prev) * eps)

    return mean_pred.astype(dtype_out)

def denoising_loop(model, params, diff_params, image_in_0, dtype = jnp.float32):
    """
    Full on device ddim sample loop, non-respaced version
    """
    def one_step(i, image_in_step):
        embed_in = jnp.array([999 - i])
        return ddim_sample(model, params, diff_params, image_in_step, embed_in, dtype_out = dtype)
    return jax.lax.fori_loop(0, 1000, one_step, image_in_0)

def denoising_loop_respaced(params, image_in_0):
    """
    Full on device ddim sample loop, respaced version
    """
    def one_step(i, image_in_step, respacing_factor):
        embed_in = jnp.array([999 - i * respacing_factor])
        return ddim_sample(params, image_in_step, embed_in, rescale_fact = respacing_factor)
    image_in_0 = jax.lax.stop_gradient(image_in_0)
    return jax.lax.fori_loop(0, 25, one_step, image_in_0)

def sample_to_img(sample):
    """
    Converrt -1 -> 1 float to 0 -> 255 uint8
    """
    sample_img = jax.lax.clamp(jnp.array(0.0).astype(sample.dtype), ((sample + 1) * 127.5), jnp.array(255.0).astype(sample.dtype)).astype(jnp.uint8)
    return sample_img 

def img_to_sample(sample):
    """
    Converrt 0 -> 255 uint8 to -1 -> 1 float
    """
    sample = (jnp.array(mnist_data) / 127.5 - 1.0).astype(jnp.float32)
    return sample 
