import jax
import jax.numpy as jnp

def loss_fn(model, params, diff_params, x, t, noise):
    """
    DDIM training loss function
    """
    # Loss (ddim variant, _no_ learned variance).
    alphas_cumprod_t = diff_params["sqrt_alphas_cumprod"][t].reshape(*tuple([-1] +  [1] * (len(x.shape) - 1)))
    sqrt_one_minus_alphas_cumprod_t = diff_params["sqrt_one_minus_alphas_cumprod"][t].reshape(*tuple([-1] +  [1] * (len(x.shape) - 1)))
    x_diffused = alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    model_output = model.apply(params, x_diffused, t)
    #model_output, model_var_values = jnp.split(model_output, 2, -1) # in case of actually wanting to learn variance after all, do something with this
    loss = jnp.mean((noise - model_output) ** 2, axis = tuple(range(1, len(model_output.shape)))) # mean over all but batch dim
    return loss

def loss_fn_mean(model, params, diff_params, x, t, noise):
    """
    Whole batch mean calculating wrapper for loss_fn
    """
    return jnp.mean(loss_fn(model, params, diff_params, x, t, noise))
