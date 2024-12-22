import jax
from brax import envs
from brax.io import image
from PIL import Image
from tqdm import tqdm
from numpngw import write_apng

if __name__ == "__main__":
    # Initialize the environment
    env_name = "ant"
    backend = "mjx"  # Choose backend
    env = envs.get_environment(env_name=env_name, backend=backend)

    # Initialize random key
    key = jax.random.PRNGKey(0)
    state = env.reset(rng=key)
    jit_step = jax.jit(env.step)

    # Simulate random actions and collect frames
    frames = []
    for _ in tqdm(range(50)):  # Number of steps
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, shape=env.action_size, minval=-1.0, maxval=1.0)
        state = jit_step(state, action)
        frames.append(image.render_array(env.sys, state.pipeline_state))

    write_apng("random_episode.png", frames)
    print("GIF saved as 'random_episode.gif'.")
