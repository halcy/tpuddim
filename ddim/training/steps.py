import optax
import jax
from jax.experimental.pjit import pjit, PartitionSpec

from ..diffusion.loss import loss_fn_mean

def train_step_update(opt, model, diff_params, params, opt_params, batch, timesteps, noise):
    """
    Take one parameter update step
    """
    def loss_curried(params, diff_params, batch, timesteps, noise):
        return loss_fn_mean(model, params, diff_params, batch, timesteps, noise)
    loss, grad = jax.value_and_grad(loss_curried)(params, diff_params, batch, timesteps, noise)
    updates, opt_params = opt.update(grad, opt_params, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_params

def get_pjit_train_step_update(opt, model, diff_params):
    """
    Return a version of train_step_update with opt, model and diff_params curried out and pjit applied
    """
    def step_func(params, opt_params, batch, timesteps, noise):
        return train_step_update(
            opt, 
            model, 
            diff_params,
            params,
            opt_params,
            batch,
            timesteps,
            noise
        )

    step_func = pjit(
        step_func, 
        [None, None, PartitionSpec("batch", "x", "y"), PartitionSpec("batch"), PartitionSpec("batch", "x", "y")], 
        [None, None, None]
    )
    
    return step_func

def train_step(update_func, data_sampler, timestep_sampler, ema, prng, params, opt_params):
    """
    Take one full training step:
    * Sample timesteps
    * Sample data batch
    * Calculate loss and update paramaters
    * Perform EMA update, if desired
    """
    prng, batch = data_sampler.sample(prng)
    prng, timesteps, weights = timestep_sampler.sample(prng)
    
    prng, subkey = jax.random.split(prng)
    noise = jax.random.normal(subkey, batch.shape)
    
    loss, params_new, opt_params_new = update_func(params, opt_params, batch, timesteps, noise)
        
    if not ema is None:
        params_new = ema.apply(params_new, params)
    return prng, params_new, opt_params_new, loss

def get_pjit_train_step(update_func, data_sampler, timestep_sampler, ema = None):
    """
    Version of train_step with constant arguments curried out and pjit applied
    """
    def step_func(prng, params, opt_params):
        return train_step(
            update_func,
            data_sampler,
            timestep_sampler,
            ema,
            prng, 
            params, 
            opt_params
        )
    return pjit(step_func, None, None)

def get_train_loop(opt, model, diff_params, data_sampler, timestep_sampler, ema, how_many = 1000):
    """
    Get a function that trains for multiple steps (default: 1000) without going back to the host
    """
    update_func = get_pjit_train_step_update(opt, model, diff_params)
    step_func = get_pjit_train_step(update_func, data_sampler, timestep_sampler, ema)
    
    def train_iterations(prng, params, opt_params):
        def one_step(idx, args):
            (prng, params, opt_params, loss) = args
            prng, params, opt_params, loss = step_func(prng, params, opt_params)
            return (prng, params, opt_params, loss)

        loss = 0.0
        args_0 = (prng, params, opt_params, loss)
        return jax.lax.fori_loop(0, how_many, one_step, args_0)

    return pjit(train_iterations, None, None)
    
