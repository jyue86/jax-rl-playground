import jax
import jax.numpy as jnp
import gymnax
from gymnax.visualize import Visualizer

if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)
    env, env_params = gymnax.make("CartPole-v1")
    obs, state = env.reset(key_reset, env_params)

    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = env.action_space(env_params).sample(rng_act)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
            obs = next_obs
        env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    print(len(state_seq), len(cum_rewards))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate("recordings/anim.gif")