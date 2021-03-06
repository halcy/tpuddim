import optax
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit, PartitionSpec
from jax.ops import index_update

from ..diffusion.loss import loss_fn_mean

def train_step_update(opt, model, diff_params, params, opt_params, batch, timesteps, noise, prng, precision_policy = None):
    """
    Take one parameter update step
    """
    def loss_curried(params, diff_params, batch, timesteps, noise, prng):
        return loss_fn_mean(model, params, diff_params, batch, timesteps, noise, prng)
    
    if not precision_policy is None:
        params_compute = precision_policy.cast_to_compute(params)
        batch = precision_policy.cast_to_compute(batch)
    else:
        params_compute = params
        
    loss, grad = jax.value_and_grad(loss_curried)(params_compute, diff_params, batch, timesteps, noise, prng)
    
    if not precision_policy is None:
        loss = precision_policy.cast_to_param(loss)
        grad = precision_policy.cast_to_param(grad)
    
    updates, opt_params = opt.update(grad, opt_params, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_params

def get_pjit_train_step_update(opt, model, diff_params, use_pjit = True, donate = True, precision_policy = None):
    """
    Return a version of train_step_update with opt, model and diff_params curried out and pjit applied
    """
    def step_func(params, opt_params, batch, timesteps, noise, prng):
        return train_step_update(
            opt, 
            model, 
            diff_params,
            params,
            opt_params,
            batch,
            timesteps,
            noise,
            prng,
            precision_policy
        )
    
    if use_pjit:
        donate_argnums = tuple()
        if donate:
            donate_argnums = (1, 2, 3, 4, 5)
            
        step_func = pjit(
            step_func, 
            [None, None, PartitionSpec("batch", "x", "y"), PartitionSpec("batch"), PartitionSpec("batch", "x", "y"), None], 
            [None, None, None],
            donate_argnums = donate_argnums
        )
    
    return step_func

def get_train_loop(opt, model, diff_params, data_sampler, timestep_sampler, ema, how_many = 1000, pjit_loop = True, pjit_update = True, donate = True, precision_policy = None):
    """
    Get a function that trains for multiple steps (default: 1000) without going back to the host
    """
    update_func = get_pjit_train_step_update(opt, model, diff_params, pjit_update, donate, precision_policy = precision_policy)
    
    def train_iterations(prng, params, params_ema, opt_params, batches):
        def one_step(idx, args):
            (prng, params, params_ema, opt_params, batches, loss) = args
            
            # Get data sample
            batch = batches[idx]
            
            # Sample steps, noise
            prng, timesteps, weights = timestep_sampler.sample(prng)
            prng, subkey = jax.random.split(prng)
            noise = jax.random.normal(subkey, batch.shape)
            
            # Run update
            prng, subkey = jax.random.split(prng)
            loss_v, params_new, opt_params_new = update_func(params, opt_params, batch, timesteps, noise, subkey)
            loss = index_update(loss, idx, loss_v)
            if not ema is None:
                params_ema = ema.apply(params_new, params_ema)
            else:
                params_ema
            return (prng, params_new, params_ema, opt_params_new, batches, loss)

        loss = jnp.zeros((how_many,))
        args_0 = (prng, params, params_ema, opt_params, batches, loss)
        prng, params, params_ema, opt_params, _, loss = jax.lax.fori_loop(0, how_many, one_step, args_0)
        return (prng, params, params_ema, opt_params, loss)
            
    if pjit_loop:
        donate_argnums = tuple()
        if donate:
            donate_argnums = (0, 1, 2, 3, 4)
        train_loop = pjit(train_iterations, None, None, donate_argnums = donate_argnums)
    else:
        train_loop = train_iterations
        
    def get_data_and_train(prng, params, params_ema, opt_params):
        samples = []
        for i in range(how_many):
            prng, batch = data_sampler.sample(prng)
            samples.append(batch)
        samples = jnp.array(samples)
        if not precision_policy is None:
            samples = precision_policy.cast_to_compute(samples)
        return train_loop(prng, params, params_ema, opt_params, samples)
    
    return get_data_and_train
    