from .trainer import Trainer
from .networks import SimpleMLP

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass

@dataclass
class PPOConfig:
    seed: int = 0
    training_steps: int = 1_000_000
    n_envs: int = 8
    eps: float = 0.2
    minibatch_size: int = 64
    episode_length: int = 2048
    epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_lr = 1e-4
    value_lr = 1e-4
    entropy_weight: float = 0.01

@dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    value: jax.Array
    entropy: jax.Array
    next_obs: jax.Array
    done: jax.Array
    log_prob: jax.Array

class PPO(Trainer):
    def __init__(self, env: "Env", model: nnx.Module, optimizer: "optax.Optim", config: dataclass):
        super().__init__(env, model, config)

    def make_train(self):
        def train():
            key = jax.random.PRNGKey(self.config.seed)
            jit_reset = jax.jit(self.env.reset)
            jit_step = jax.jit(self.env.step)

            def update_step(key):
                key, _key = jax.random.split(key, 2)
                reset_keys = jax.random.split(_key, self.config.n_envs)
                obs = jax.vmap(jit_reset)(reset_keys)

                def env_step(prev_obs: jax.Array, _):
                    action, value, log_prob, entropy = self.model(prev_obs)
                    next_obs = nnx.vmap(jit_step)(prev_obs, action)
                    return next_obs, Transition(
                        obs=prev_obs,
                        action=action,
                        reward=next_obs.reward,
                        value=value,
                        entropy=entropy,
                        next_obs=next_obs.obs,
                        done=next_obs.done,
                        log_prob=log_prob
                    ) 

                _, rollouts = jax.lax.scan(env_step, obs, length=self.config.episode_length)
                vmap_calculate_gae_and_targets = jax.vmap(self.calculate_gae_and_targets)
                gaes, targets = vmap_calculate_gae_and_targets(
                    rollouts.reward, 
                    rollouts.value, 
                    rollouts.done
                )

                # Shapes
                # obs shape = S, action shape = A, episode lenght =T
                # obs = (n_envs, T, S)
                # action = (n_envs, T, A)
                # rewards = (n_envs, T, 1)
                # values = (n_envs, T, 1)
                # entropy = (n_envs, T, 1)
                # next_obs = (n_envs, T, S)
                # dones = (n_envs, T, 1)
                # log_prob = (n_envs, T, 1)

                rollouts = jax.tree.map(lambda x: x.reshape((-1, x.shape[-1])), rollouts)
                minibatch_rollouts = jax.tree.map(self.reshape_transitions_to_minibatches, rollouts)

                def update_policy(rollouts: Transition):
                    pass

        return train
    
    def calculate_gae_and_targets(self, rewards: jax.Array, values: jax.Array, dones: jax.Array):
        T = self.config.episode_length
        init_gae = 0.0
        init_target = values[-1]
        last_value = values[-1]

        def calculate_gae_and_targets_helper(carry, i):
            reward = rewards[i]
            done = dones[i]
            value = values[i]
            prev_gae, prev_target, last_value = carry
            
            delta = reward + self.config.gamma * last_value * (1 - done) - value 
            gae = delta + self.config.gamma * self.config.gae_lambda * prev_gae * (1 - done)
            target = reward + self.config.gamma * prev_target 
            return (gae, target, value), (gae, target) 
        # TODO: check if this unpacking is still correct
        _, (gaes, targets) = jax.lax.scan(
            calculate_gae_and_targets_helper, 
            (init_gae, init_target, last_value), 
            jnp.arange(T - 1, -1, -1))
        return gaes[::-1], targets[::-1]

    def reshape_transitions_to_minibatches(self, x, batch_size: int):
        shape = x.shape
        assert shape[0] % batch_size == 0, "Batch size must divide the number of transitions"
        n_batches = shape[0] // batch_size
        x = x.reshape((n_batches, batch_size) + shape[1:])
        return x