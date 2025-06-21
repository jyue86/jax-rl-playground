from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from distrax import Normal, Distribution

class SimpleMLP(nnx.Module):
    def __init__(self, layers: List[int], rng: nnx.Rngs, activation: str = "relu",):
        super().__init__()
        model_layers = []
        activation = getattr(nnx, activation)
        for i in range(len(layers) - 1):
            model_layers.append(nnx.Linear(layers[i], layers[i + 1], rngs=rng))
            model_layers.append(activation)
        self.layers = nnx.Sequential(*model_layers) 

    def __call__(self, x):
        return self.layers(x)
    
class ActorCritic(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_layers: Tuple[int, ...], rng: nnx.Rngs):
        super().__init__()
        hidden_layers_list = list(hidden_layers)
        actor_layers = [obs_dim] + hidden_layers_list + [action_dim]
        critic_layers = [obs_dim] + hidden_layers_list + [1]
        self.actor = SimpleMLP(actor_layers, rng=rng)
        self.critic = SimpleMLP(critic_layers, rng=rng)
        self.std = nnx.Param(jnp.ones(action_dim) * 0.5, name="std")

    def __call__(self, x: jax.Array, rng: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        mean = self.actor(x)
        value = self.critic(x) 
        action_dist = Normal(loc=mean, scale=self.std.value)
        action = action_dist.sample(seed=rng)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action, value, log_prob, entropy

    def take_deterministic_action(self, x) -> jax.Array:
        action = self.actor(x)
        return action
    
    def get_updated_dist(self, x) -> Distribution:
        mean = self.actor(x)
        action_dist = Normal(loc=mean, scale=self.std.value)
        return action_dist