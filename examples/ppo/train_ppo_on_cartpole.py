from src.training import PPO, PPOConfig
from src.training.networks import ActorCritic

from argparse import ArgumentParser
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import nnx
from mujoco import mjx
from mujoco_playground import registry
import optax

def parse_args():
    parser = ArgumentParser(description="Train PPO on CartPole-v1")
    # TODO: add the arguments or switch to Tyro???
    return parser.parse_args()

def create_env():
    env = registry.load("CartpoleBalance")
    return env

def create_model(n_obs, n_actions):
    rng = nnx.Rngs(0)
    hidden_layers = [512, 256, 128]
    model = ActorCritic(obs_dim=n_obs, action_dim=n_actions, hidden_layers=hidden_layers, rng=rng)
    return model

if __name__ == "__main__":
    args = parse_args()
    env = create_env()
    n_obs = env.observation_size
    n_actions = env.action_size

    config = PPOConfig(hidden_layers=(512, 256, 128))
    model = create_model(n_obs, n_actions)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=config.lr))
    ppo_trainer = PPO(env, config)
    train = ppo_trainer.make_train()
    jit_train = nnx.jit(train)
    jit_train()
    # train()