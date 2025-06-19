from .trainer import Trainer
from .networks import SimpleMLP, ActorCritic

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass
import optax

from typing import Tuple, List

@dataclass
class PPOConfig:
    seed: int = 0
    training_steps: int = 1_000_000
    n_envs: int = 8
    eps: float = 0.2
    n_minibatches: int = 128
    episode_length: int = 512 
    epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    entropy_weight: float = 0.01
    hidden_layers: Tuple[int,...] = (64, 64)

@dataclass
class EnvState:
    key: jax.Array
    prev_obs: jax.Array

@dataclass
class UpdateState:
    key: jax.Array
    loss: jax.Array

@dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    value: jax.Array
    next_obs: jax.Array
    done: jax.Array
    log_prob: jax.Array

class PPO(Trainer):
    # TODO: consider making a model a string as well?
    # Basically, make it consistent with optimizer
    def __init__(self, env: "Env", config: dataclass):
        super().__init__(env, config)

    def make_train(self):
        def train():
            rng = nnx.Rngs(0)
            n_obs = self.env.observation_size
            n_actions = self.env.action_size
            model = ActorCritic(obs_dim=n_obs, action_dim=n_actions, hidden_layers=self.config.hidden_layers, rng=rng)
            optimizer = nnx.Optimizer(model, optax.adam(learning_rate=self.config.lr))

            key = jax.random.PRNGKey(self.config.seed)
            vmap_reset = jax.vmap(self.env.reset)
            vmap_step = jax.vmap(self.env.step)
            n_updates = self.config.training_steps // (self.config.n_envs * self.config.episode_length)
            minibatch_size = self.config.n_envs * self.config.episode_length // self.config.n_minibatches

            def update_step(key: jax.Array, _):
                key, _key = jax.random.split(key, 2)
                reset_keys = jax.random.split(_key, self.config.n_envs)
                obs = vmap_reset(reset_keys)

                def env_step(env_state: EnvState, _):
                    key = env_state.key
                    prev_obs = env_state.prev_obs
                    key, _key = jax.random.split(key, 2)

                    action, value, log_prob, _ = model(prev_obs.obs, rng=_key)
                    next_obs = vmap_step(prev_obs, action)
                    next_env_state = EnvState(key=key, prev_obs=next_obs)

                    return next_env_state, Transition(
                        obs=prev_obs.obs,
                        action=action,
                        reward=next_obs.reward,
                        value=value,
                        next_obs=next_obs.obs,
                        done=next_obs.done,
                        log_prob=log_prob
                    ) 

                final_env_state, rollouts = jax.lax.scan(env_step, EnvState(key, obs), length=self.config.episode_length)
                key = final_env_state.key
                rollouts = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rollouts)
                vmap_calculate_gae_and_targets = jax.vmap(self.calculate_gae_and_targets)
                gaes, targets = vmap_calculate_gae_and_targets(
                    rollouts.reward, 
                    rollouts.value, 
                    rollouts.done
                )

                # Shapes
                # obs shape = S, action shape = A, episode lenght =T
                # obs = (T, n_envs, S)
                # action = (T, n_envs, A)
                # rewards = (T, n_envs, 1)
                # values = (T, n_envs, 1)
                # entropy = (T, n_envs, 1)
                # next_obs = (T, n_envs, S)
                # dones = (T, n_envs, 1)
                # log_prob = (T, n_envs, 1)

                batch = (rollouts, gaes, targets)
                # rollouts = jax.tree.map(lambda x: x.reshape((-1, x.shape[-1])), batch)
                minibatches = jax.tree.map(lambda x: self.reshape_transitions_to_minibatches(x, minibatch_size), batch)
                key, _key = jax.random.split(key, 2)
                permutation = jax.random.permutation(_key, jnp.arange(self.config.n_minibatches))
                shuffled_minibatches = jax.tree.map(lambda x: x[permutation], minibatches)

                def update_policy(update_state_and_minibatches: Tuple[UpdateState, Tuple], current_epoch: int):
                    update_state, minibatches = update_state_and_minibatches
                    start_key = update_state.key
                    update_loss = update_state.loss
                    def minibatch_update(minibatch_update_state: UpdateState, minibatch: jax.Array):
                        current_loss = minibatch_update_state.loss
                        key = minibatch_update_state.key
                        key, _key = jax.random.split(key, 2)
                        loss, grads = nnx.value_and_grad(self.loss_fn)(model, minibatch, _key)
                        optimizer.update(grads)
                        # TODO: remove second return value, make it None?
                        return UpdateState(key=key, loss=current_loss + loss), loss 
                    end_of_epoch_state, _ = jax.lax.scan(minibatch_update, UpdateState(key=start_key, loss=jnp.zeros((1,))), minibatches)
                    total_loss = end_of_epoch_state.update_loss
                    end_key = end_of_epoch_state.key
                    avg_epoch_loss = total_loss / self.config.n_minibatches
                    return UpdateState(key=end_key, loss=update_loss + total_loss), avg_epoch_loss

                update_state = UpdateState(key=key, loss=jnp.zeros((1,)))
                end_of_update_state, avg_epoch_losses = jax.lax.scan(
                    update_policy,
                    (update_state, shuffled_minibatches),
                    length=self.config.epochs
                )
                total_update_loss = update_state.loss 
                key = end_of_update_state.key
                avg_update_loss = total_update_loss / self.config.epochs
                return key, avg_update_loss 

            key, loss = jax.lax.scan(update_step, key, length=n_updates)
        return train

    def evaluate(self, n_episodes: int):
        pass
    
    def calculate_gae_and_targets(self, rewards: jax.Array, values: jax.Array, dones: jax.Array):
        T = self.config.episode_length
        init_gae = jnp.zeros((1,))
        init_target = values[-1]
        last_value = jnp.zeros((1,)) 

        def calculate_gae_and_targets_helper(carry, i):
            reward = rewards[i]
            done = dones[i]
            value = values[i]
            prev_gae, prev_target, last_value = carry
            
            delta = reward + self.config.gamma * last_value * (1 - done) - value 
            gae = delta + self.config.gamma * self.config.gae_lambda * prev_gae * (1 - done)
            target = reward + self.config.gamma * prev_target 
            return (gae, target, value), (gae, target) 

        _, (gaes, targets) = jax.lax.scan(
            calculate_gae_and_targets_helper, 
            (init_gae, init_target, last_value), 
            jnp.arange(T - 1, -1, -1))
        return gaes[::-1], targets[::-1]

    def reshape_transitions_to_minibatches(self, x, batch_size: int):
        # x = x.reshape((-1, x.shape[-1]))
        x = x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        shape = x.shape
        assert shape[0] % batch_size == 0, f"Batch size {batch_size} must divide the number of transitions {shape[0]}"
        n_batches = shape[0] // batch_size
        x = x.reshape((n_batches, batch_size) + shape[1:])
        return x

    def loss_fn(self, model: ActorCritic, minibatch: Tuple[Transition, jax.Array, jax.Array], rng: jax.random.PRNGKey):
        rollouts, gaes, targets = minibatch

        # normalize advantages
        gaes = (gaes - jnp.mean(gaes)) / (jnp.std(gaes) + 1e-8)
        _, values, _, entropy = model(rollouts.obs, rng=rng)
        
        # Calculate policy loss
        dist = model.get_updated_dist(rollouts.obs)
        minibatch_log_probs = dist.log_prob(rollouts.action)
        ratio = jnp.exp(minibatch_log_probs - rollouts.log_prob)
        clipped_ratio = jnp.clip(ratio, 1 - self.config.eps, 1 + self.config.eps)
        # jax.debug.print("Ratio shape: {x}", x=(ratio * gaes).shape)
        # jax.debug.print("Clipped ratio shape: {x}", x=(clipped_ratio * gaes).shape)
        ratios = jnp.concat([ratio * gaes, clipped_ratio * gaes], axis=1)
        # assert ratio.shape == gaes.shape, "Ratio and GAEs must have the same shape"
        # assert clipped_ratio.shape == gaes.shape, "Clipped ratio and GAEs must have the same shape"
        policy_loss = -jnp.mean(jnp.min(ratios, axis=1)) - self.config.entropy_weight * jnp.mean(entropy)

        # calculate value loss
        value_loss = jnp.mean((values - targets) ** 2)
        jax.debug.breakpoint()
        loss = policy_loss + value_loss

        return loss
