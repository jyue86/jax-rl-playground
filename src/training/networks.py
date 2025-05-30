from typing import List

from flax import nnx
from distrax import Normal

class SimpleMLP(nnx.Module):
    def __init__(self, layers: List[int], rng: nnx.Rngs, activation: str = "relu",):
        super().__init__()
        model_layers = []
        activation = getattr(nnx, activation)
        for i in range(len(layers) - 1):
            model_layers.append(nnx.Dense(layers[i], layers[i + 1], rngs=rng))
            model_layers.append(activation)
        self.layers = nnx.Sequential(model_layers) 

    def __call__(self, x):
        return self.layers(x)
    
class ActorCric(nnx.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_layers: List[int], rng: nnx.Rngs):
        super().__init__()
        self.actor = SimpleMLP([obs_dim] + hidden_layers + [action_dim], rng=rng)
        self.critic = SimpleMLP([obs_dim] + hidden_layers + [1], rng=rng)
        self.std = nnx.Param(nnx.ones(action_dim) * 0.5, name="std")

    def __call__(self, x):
        mean = self.actor(x)
        value = self.critic(x) 
        action_dist = Normal(loc=mean, scale=self.std)
        action = action_dist.sample(seed=x.rng)
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action, value, log_prob, entropy

    def take_deterministic_action(self, x):
        mean = self.actor(x)
        action = mean
    
    def get_updated_dist(self, x):
        mean = self.actor(x)
        action_dist = Normal(loc=mean, scale=self.std)
        return action_dist