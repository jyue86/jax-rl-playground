import gymnax
import jax
from numpngw import write_apng
from PIL import Image
import numpy as np
import jax.numpy as jnp
from gymnax.visualize import Visualizer


if __name__ == "__main__":
    classic_envs = [
        "Acrobot-v1",
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
    ]

    env_name = "CartPole-v1"
    env, env_params = gymnax.make(env_name) 
    key = jax.random.PRNGKey(0)
    key, reset_key = jax.random.split(key)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    obs, env_state = jit_reset(reset_key, env_params)
    frames = []
    reward_seq = []

    for _ in range(200):
        if env_name in classic_envs:
            frames.append(env_state)
        key, action_key, step_key = jax.random.split(key, 3)
        action = env.action_space(env_params).sample(action_key)
        obs, env_state, reward, done, info = jit_step(step_key, env_state, action, env_params)
        reward_seq.append(reward)

        if env_name in classic_envs:
            continue
        fig, ax = env.render(env_state, env_state)
        fig.canvas.draw()
        frame = Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb()
        )
        frames.append(np.array(frame))

        if done:
            break

    if env_name in classic_envs:
        cum_rewards = jnp.cumsum(jnp.array(reward_seq))
        vis = Visualizer(env, env_params, frames, cum_rewards)
        vis.animate("gymnax.gif")
    else:
        write_apng("gymnax.gif", frames)

