from .trainer import Trainer
from .networks import ActorCritic

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import train_state
from flax.struct import dataclass
import optax

from typing import Tuple, Any

@dataclass
class PPOConfig:
    seed: int = 0
    training_steps: int = 1_000_000
    n_envs: int = 8
    eps: float = 0.2
    n_minibatches: int = 32 
    episode_length: int = 512 
    epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    entropy_weight: float = 0.01
    hidden_layers: Tuple[int,...] = (64, 64)

class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef

@dataclass
class RolloutState:
    key: jax.Array
    prev_obs: jax.Array

class UpdateState(train_state.TrainState):
    key: jax.Array
    graphdef: nnx.GraphDef
    loss: jax.Array

@dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    value: jax.Array
    done: jax.Array
    log_prob: jax.Array

@dataclass
class EvalResult:
    reward: jax.Array
    done: jax.Array

class PPO(Trainer):
    # TODO: consider making a model a string as well?
    # Basically, make it consistent with optimizer
    def __init__(self, env: "Env", config: PPOConfig):
        super().__init__(env, config)

    def make_train(self):
        def train():
            rng = nnx.Rngs(0)
            n_obs = self.env.observation_size
            n_actions = self.env.action_size
            model = ActorCritic(obs_dim=n_obs, action_dim=n_actions, hidden_layers=self.config.hidden_layers, rng=rng)
            graphdef, params = nnx.split(model, nnx.Param)
            optimizer = optax.adam(learning_rate=self.config.lr)
            train_state = TrainState.create(
                apply_fn=None,
                graphdef=graphdef,
                params=params,
                tx=optimizer
            )
            del params
            # optimizer = nnx.Optimizer(model, optax.adam(learning_rate=self.config.lr))

            key = jax.random.PRNGKey(self.config.seed)
            vmap_reset = jax.vmap(self.env.reset)
            vmap_step = jax.vmap(self.env.step)
            n_updates = self.config.training_steps // (self.config.n_envs * self.config.episode_length)
            minibatch_size = self.config.n_envs * self.config.episode_length // self.config.n_minibatches

            def update_step(runner_state: Tuple[jax.Array, TrainState, jax.Array], _):
                key, train_state, avg_loss = runner_state
                key, _key = jax.random.split(key, 2)
                reset_keys = jax.random.split(_key, self.config.n_envs)
                obs = vmap_reset(reset_keys)

                rollout_model = nnx.merge(train_state.graphdef, train_state.params)
                def env_step(rollout_state: RolloutState, _):
                    key = rollout_state.key
                    prev_obs = rollout_state.prev_obs
                    key, _key = jax.random.split(key, 2)

                    action, value, log_prob, _ = rollout_model(prev_obs.obs, rng=_key)
                    # jax.debug.breakpoint()
                    next_obs = vmap_step(prev_obs, action)
                    next_env_state = RolloutState(key, next_obs)

                    return next_env_state, Transition(
                        obs=prev_obs.obs,
                        action=action,
                        reward=next_obs.reward,
                        value=value,
                        done=next_obs.done,
                        log_prob=log_prob
                    ) 

                final_rollout_state, rollouts = jax.lax.scan(env_step, RolloutState(key, obs), length=self.config.episode_length)
                key = final_rollout_state.key
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
                # dones = (T, n_envs, 1)
                # log_prob = (T, n_envs, 1)
                assert rollouts.obs.shape == (self.config.n_envs, self.config.episode_length, n_obs), f"Observation shape mismatch, actual: {rollouts.obs.shape}, expected: {(self.config.n_envs, self.config.episode_length, n_obs)}"
                assert rollouts.action.shape == (self.config.n_envs, self.config.episode_length, n_actions), f"Action shape mismatch, actual: {rollouts.action.shape}, expected: {(self.config.n_envs, self.config.episode_length, n_actions)}"
                assert rollouts.reward.shape == (self.config.n_envs, self.config.episode_length), f"Reward shape mismatch, actual: {rollouts.reward.shape}, expected: {(self.config.n_envs, self.config.episode_length)}"
                assert rollouts.value.shape == (self.config.n_envs, self.config.episode_length, 1), f"Value shape mismatch, actual: {rollouts.value.shape}, expected: {(self.config.n_envs, self.config.episode_length, 1)}"
                assert rollouts.done.shape == (self.config.n_envs, self.config.episode_length), f"Done shape mismatch, actual: {rollouts.done.shape}, expected: {(self.config.n_envs, self.config.episode_length)}"
                assert rollouts.log_prob.shape == (self.config.n_envs, self.config.episode_length, 1), f"Log probability shape mismatch, actual: {rollouts.log_prob.shape}, expected: {(self.config.n_envs, self.config.episode_length, 1)}"
                assert gaes.shape == (self.config.n_envs, self.config.episode_length, 1), f"GAE shape mismatch, actual: {gaes.shape}, expected: {(self.config.n_envs, self.config.episode_length, 1)}"
                assert targets.shape == (self.config.n_envs, self.config.episode_length, 1), f"Targets shape mismatch, actual: {targets.shape}, expected: {(self.config.n_envs, self.config.episode_length, 1)}"

                batch = (rollouts, gaes, targets)
                # rollouts = jax.tree.map(lambda x: x.reshape((-1, x.shape[-1])), batch)
                minibatches = jax.tree.map(lambda x: self.reshape_transitions_to_minibatches(x, minibatch_size), batch)
                key, _key = jax.random.split(key, 2)
                permutation = jax.random.permutation(_key, jnp.arange(self.config.n_minibatches))
                shuffled_minibatches = jax.tree.map(lambda x: x[permutation], minibatches)

                def update_policy(update_state: Tuple[jax.Array, TrainState, Tuple, jax.Array], current_epoch: int):
                    key, train_state, epoch_minibatches, update_loss = update_state 
                    def minibatch_update(minibatch_update_state: Tuple[jax.Array, TrainState, jax.Array], minibatch: jax.Array):
                        key, train_state, current_epoch_loss = minibatch_update_state
                        graphdef = train_state.graphdef
                        model = nnx.merge(graphdef, train_state.params)

                        key, _key = jax.random.split(key, 2)
                        loss, grads = nnx.value_and_grad(self.loss_fn)(model, minibatch, _key)
                        train_state = train_state.apply_gradients(grads=grads)
                        return (key, train_state, current_epoch_loss + loss), loss 

                    epoch_loss = jnp.zeros((1,))
                    next_update_state, _ = jax.lax.scan(minibatch_update, (key, train_state, epoch_loss), epoch_minibatches)
                    key, train_state, total_loss = next_update_state
                    avg_epoch_loss = total_loss / self.config.n_minibatches
                    next_update_state = (key, train_state, shuffled_minibatches, update_loss + avg_epoch_loss)
                    return next_update_state, avg_epoch_loss

                loss = jnp.zeros((1,))
                end_of_update_state, avg_epoch_losses = jax.lax.scan(
                    update_policy,
                    (key, train_state, shuffled_minibatches, loss),
                    length=self.config.epochs
                )
                key, train_state, _, total_update_loss = end_of_update_state
                avg_update_loss = total_update_loss / self.config.epochs
                return (key, train_state, avg_update_loss), None

            avg_update_loss = jnp.zeros((1,))
            train_state, loss = jax.lax.scan(update_step, (key, train_state, avg_update_loss), length=n_updates)
            return train_state
        return train

    def evaluate(self, key: jax.Array, n_episodes: int, train_state: TrainState):
        assert n_episodes % self.config.n_envs == 0, "Number of episodes must be divisible by the number of environments"
        eval_steps = n_episodes // self.config.n_envs
        vmap_reset = jax.vmap(self.env.reset)
        vmap_step = jax.vmap(self.env.step)
        model = nnx.merge(train_state.graphdef, train_state.params)

        def evaluate_loop(key: jax.Array, _):
            key, reset_keys = jax.random.split(key, self.config.n_envs + 1)
            obs = vmap_reset(reset_keys)
            def evaluate_episode(prev_obs, _):
                action = model.take_deterministic_action(prev_obs.obs)
                obs = vmap_step(action)
                return obs, EvalResult(reward=obs.reward, done=obs.done)
            _, eval_results = jax.lax.scan(evaluate_episode, obs, length = self.config.episode_length)
            return key, eval_results
        key, eval_results = jax.lax.scan(evaluate_loop, key, length=eval_steps)
        eval_results = jax.tree.map(lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), eval_results)
        valid_indices = jnp.where(eval_results.done)[0]
        valid_rewards = eval_results.reward[valid_indices]
        return jnp.mean(valid_rewards), jnp.std(valid_rewards)
    
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
        ratios = jnp.concat([ratio * gaes, clipped_ratio * gaes], axis=1)
        assert ratio.shape == gaes.shape, "Ratio and GAEs must have the same shape"
        assert clipped_ratio.shape == gaes.shape, "Clipped ratio and GAEs must have the same shape"
        policy_loss = -jnp.mean(jnp.min(ratios, axis=1)) - self.config.entropy_weight * jnp.mean(entropy)

        # calculate value loss
        value_loss = jnp.mean((values - targets) ** 2)
        entropy_loss = self.config.entropy_weight * jnp.mean(entropy)
        loss = policy_loss + value_loss - entropy_loss

        return loss
