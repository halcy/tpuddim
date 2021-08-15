def train_step_update(prng, opt, model, params, diff_params, opt_params, batch, timesteps):
    """
    Take one parameter update step
    """
    prng, subkey = jax.random.split(prng)
    noise = jax.random.normal(subkey, batch.shape)
    loss, grad = jax.value_and_grad(loss_fn_mean)(params, batch, timesteps, noise)
    updates, opt_params = opt.update(grad, opt_params, params)
    params = optax.apply_updates(params, updates)
    return prng, loss, params

def train_step(prng, opt, model, params, diff_params, opt_params, data_sampler, timestep_sampler, ema = None):
    """
    Take one full training step:
    * Sample timesteps
    * Sample data batch
    * Calculate loss and update paramaters
    * Perform EMA update, if desired
    """
    prng, batch = data_sampler.sample(prng)
    prng, timesteps, weights = timestep_sampler.sample(prng)
    prng, loss, params_new = train_step_update(prng, opt, model, params, diff_params, opt_params, batch, timesteps)
    if not ema is None:
        params = ema.apply(params_new, params)
    return prng, params, opt_params, loss
