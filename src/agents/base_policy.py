import abc
import gymnax
from flax.struct import dataclass
import jax.numpy as jnp

@dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class BasePolicy(abc):
    def __init__(self, env: gymnax.envs.Env):
        self.env = env

    @abc.abstractmethod
    def make_train(self):
        pass