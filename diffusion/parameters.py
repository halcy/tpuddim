import jax
import jax.numpy as jnp

def linear_beta_schedule(steps):
    """
    Get a linear spaced beta schedule
    """
    scale = 1000 / steps
      = scale * 0.0001
    beta_end = scale * 0.02
    return jnp.linspace(beta_start, beta_end, steps)

def diff_params_from_betas(betas, param_scale = 1.0):
    """
    Calculate diffusion parameters from betas
    """
    diff_params = {}
    diff_params["betas"] = betas
    
    diff_params["alphas"] = 1.0 - diff_params["betas"]
    diff_params["alphas_cumprod"] = jnp.cumprod(diff_params["alphas"], axis=0)
    diff_params["alphas_cumprod_prev"] = jnp.append(1.0, diff_params["alphas_cumprod"][:-1])
    diff_params["alphas_cumprod_next"] = jnp.append(diff_params["alphas_cumprod"][1:], 0.0)

    diff_params["sqrt_alphas_cumprod"] = jnp.sqrt(diff_params["alphas_cumprod"])
    diff_params["sqrt_one_minus_alphas_cumprod"] = jnp.sqrt(1.0 - diff_params["alphas_cumprod"])
    diff_params["log_one_minus_alphas_cumprod"] = jnp.log(1.0 - diff_params["alphas_cumprod"])
    diff_params["sqrt_recip_alphas_cumprod"] = jnp.sqrt(1.0 / diff_params["alphas_cumprod"])
    diff_params["sqrt_recipm1_alphas_cumprod"] = jnp.sqrt(1.0 / diff_params["alphas_cumprod"] - 1)

    diff_params["posterior_variance"] = (diff_params["betas"] * (1.0 - diff_params["alphas_cumprod_prev"]) / (1.0 - diff_params["alphas_cumprod"]))
    diff_params["posterior_log_variance_clipped"] = jnp.log(jnp.append(diff_params["posterior_variance"][1], diff_params["posterior_variance"][1:]))

    for key in diff_params:
        diff_params[key] *= param_scale
        
    return diff_params

def respace_betas(betas, steps = 25):
    """
    Respace a beta schedule to a given amount of steps
    """
    use_timesteps = list(np.arange(len(betas) - 1, 0, -(len(betas) // steps)))
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    
    new_betas = []
    last_alpha_cumprod = 1.0
    for i, alpha_cumprod in enumerate(alphas_cumprod):
        if i in use_timesteps:    
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
    
    return jnp.array(new_betas)
