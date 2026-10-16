"""Add an adversarial motion reward and discriminator update to stock RSL-RL PPO."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch

from locomotion.rl.ppo import CappedRatePPO, CommandBootstrapPPO

from . import discriminator as disc
from .features import extract_features, feature_contract
from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import ppo_config
from locomotion.rl.wrapper import OmniVecEnv, scale_observation, clock_features, smoothed_action
from . import networks as nets

ROOT = Path(__file__).resolve().parents[2]
from .dataset import DATASET, load as load_dataset, sha
OBSERVATION_WIDTH = nets.HISTORY_WIDTH + nets.COMMAND_WIDTH + nets.ACTION_WIDTH
PRIVILEGED_LAYOUT = (
    ('base_linear_velocity_body', 3, 'telemetry linear_velocity_body, m/s'),
    ('base_height', 1, 'telemetry root_pose_xyzw z, m above the flat floor'),
    ('ground_friction', 1, 'declared constant 1.0 until Table II randomization (Step 7)'),
    ('foot_contact_force', 18, 'telemetry tibia_floor_force_world_n, six XYZ normal forces, N'),
    ('external_perturbation', 6, 'zeros until Table II perturbations (Step 7)'),
    ('collision_state', 13, 'max floor-force norm > 1 N on body, six coxae and six femurs over the control'),
)
PRIVILEGED_WIDTH = sum(width for _, width, _ in PRIVILEGED_LAYOUT)
COLLISION_FORCE_N = 1.
# The default run seed 20260917 maps to 20260925, the constant every run used before this binding.
DISCRIMINATOR_SEED_OFFSET = 8


@dataclass(frozen=True)
class AMPConfig:
    dataset: str = 'locomotion/priors/datasets/amp_demonstrations_001'
    style_weight: float = 1.
    gradient_penalty: float = 10.
    discriminator_hidden: tuple = (1024, 512)
    discriminator_learning_rate: float = 1e-4
    discriminator_batch: int = 256
    discriminator_updates: int = 20
    estimator_learning_rate: float = 1e-3
    seed: int | None = None

    def resolve(self, run_seed):
        """This configuration with the discriminator seed bound to the run seed, unless it names one."""
        return self if self.seed is not None else replace(self, seed=run_seed + DISCRIMINATOR_SEED_OFFSET)

    def validate(self):
        if not all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in (
                self.style_weight, self.gradient_penalty, self.discriminator_learning_rate,
                self.estimator_learning_rate)):
            raise ValueError('AMP coefficients must be finite and nonnegative')
        if (type(self.discriminator_batch) is not int or self.discriminator_batch < 1
                or type(self.discriminator_updates) is not int or self.discriminator_updates < 0
                or (self.seed is not None and (type(self.seed) is not int or self.seed < 0))
                or not self.discriminator_hidden):
            raise ValueError('Invalid AMP discriminator schedule')
        path = Path(self.dataset)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Dataset path must be relative to the repository root')


def json_shape(value):
    """The value as JSON stores it, so a saved record compares equal to a fresh configuration."""
    return json.loads(json.dumps(value, allow_nan=False))


def discriminator_loss(discriminator, prior, policy, gradient_penalty=10.):
    """Eq. (1) with the gradient penalty taken on the standardized prior transition.

    ``amp_discriminator.discriminator_loss`` differentiates the raw transition;
    the network divides each feature by its spread, so that gradient scales by
    1/std. This learner differentiates the tensor the network consumes, as the
    cited AMP method does, so feature scale cannot change the penalty.
    """
    standardized = ((prior - discriminator.mean) / discriminator.std).clone().requires_grad_(True)
    prior_scores = discriminator.net(standardized).squeeze(-1)
    gradient, = torch.autograd.grad(prior_scores.sum(), standardized, create_graph=True)
    terms = {"prior": (prior_scores - 1).square().mean(), "policy": (discriminator(policy) + 1).square().mean(),
             "gradient_penalty": .5 * gradient_penalty * gradient.square().sum(-1).mean()}
    return sum(terms.values()), {key: float(value.detach()) for key, value in terms.items()}


def load_demonstrations(directory=DATASET):
    """Load the admitted transition bank and bind its manifest identity."""
    directory = Path(directory)
    try:
        arrays, manifest = load_dataset(directory)
    except ValueError as error:
        raise ValueError('Demonstration transitions differ from their manifest or feature contract') from error
    states, next_states = arrays['states'], arrays['next_states']
    file_sha = manifest['file_sha256']
    identity = {'path': str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
                'manifest_sha256': sha(directory/'manifest.json'), 'transitions_sha256': file_sha,
                'transitions': int(len(states)), 'model_sha256': manifest.get('model_sha256'),
                'reviewer': manifest.get('reviewer')}
    return disc.transitions(states, next_states), identity


class CollisionCapture:
    """Record per-control floor-force maxima on the 13 non-tibia bodies without a training trajectory."""

    def __init__(self, env, inner=None):
        self.inner = inner
        self.bodies = [i for i, name in enumerate(env.native_body_names) if not name.endswith('_tibia')]
        if len(self.bodies) != 13:
            raise ValueError('Expected the body, six coxae and six femurs as non-tibia bodies')
        self.maximum = torch.zeros(env.num_envs, 13, device=env.device)

    def __call__(self, env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre):
        norms = torch.linalg.vector_norm(forces[:, self.bodies], dim=-1)
        if substep == 0:
            self.maximum = norms.clone()
        else:
            self.maximum = torch.maximum(self.maximum, norms)
        if self.inner is not None:
            self.inner(env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre)

    def __getattr__(self, name):
        if self.__dict__.get('inner') is None:
            raise AttributeError(name)
        return getattr(self.inner, name)


class AMPVecEnv(OmniVecEnv):
    """The omni task adapter plus AMP states, named actor groups and the critic's privileged state."""

    def __init__(self, task, tensor_dict=None, *, collision=None, amp_config=None, **wrapper_options):
        self.collision = collision
        self.amp_config = AMPConfig() if amp_config is None else amp_config
        self.amp_config.validate()
        self.reset_rows = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        super().__init__(task, tensor_dict, **wrapper_options)
        if not task.cfg.record_motion_features:
            raise ValueError('AMP training needs record_motion_features in the environment configuration')
        self.cfg.update(motion_prior=True, amp=json_shape(asdict(self.amp_config)), privileged_layout=[
            {'name': name, 'width': width, 'source': source} for name, width, source in PRIVILEGED_LAYOUT])
        self.reset_rows[:] = True

    def privileged_state(self, output):
        env, n = self.task.env, self.num_envs
        telemetry = getattr(env, 'telemetry', None) or {}
        velocity = output['critic'][:, OBSERVATION_WIDTH:OBSERVATION_WIDTH + 3]
        # The critic observation carries navigation-frame velocity; the body frame is forward=-Y, left=+X.
        body_velocity = torch.stack((velocity[:, 1], -velocity[:, 0], velocity[:, 2]), dim=-1)
        height = torch.full((n, 1), float(env.reset_height), device=self.device)
        forces = torch.zeros(n, 18, device=self.device)
        collision = torch.zeros(n, 13, device=self.device)
        if telemetry:
            height = telemetry['root_pose_xyzw'][:, 2:3].clone()
            forces = telemetry['tibia_floor_force_world_n'].reshape(n, 18).clone()
            if self.collision is not None:
                collision = (self.collision.maximum > COLLISION_FORCE_N).float()
        fresh = self.reset_rows[:, None]
        height = torch.where(fresh, torch.full_like(height, float(env.reset_height)), height)
        forces = torch.where(fresh, torch.zeros_like(forces), forces)
        collision = torch.where(fresh, torch.zeros_like(collision), collision)
        state = torch.cat((body_velocity, height, torch.ones(n, 1, device=self.device), forces,
                           torch.zeros(n, 6, device=self.device), collision), dim=-1)
        if state.shape != (n, PRIVILEGED_WIDTH) or not bool(torch.isfinite(state).all()):
            raise ValueError('Privileged state shape/finite check failed')
        return state

    def observations(self, output):
        """Policy groups for ``output``, which the task returned for the environment's current state."""
        obs = output['obs']
        groups = dict(super().observations(output).items())
        groups.update({'amp': extract_features(self.task.env.current),
                  'velocity_label': output['critic'][:, OBSERVATION_WIDTH:OBSERVATION_WIDTH + 3],
                  'privileged': self.privileged_state(output)})
        groups.update(nets.actor_observation(obs).items())
        return self.tensor_dict(groups, batch_size=[self.num_envs])

    def step(self, action):
        held = self.task.commands.clone() if self.command_segments == 'bootstrap' else None
        if self.action_smoothing == 'mean2':
            # PPO keeps the raw sample; the environment receives the two-control mean.
            raw = action.detach().clone()
            action = smoothed_action(action, self.previous_action.to(action), self.task.episode_steps)
            self.previous_action = raw
        output = self.task.step(action)
        rewards = output['reward'].clone()
        done = output['terminated'] | output['truncated']
        # A physical termination overrides a coincident episode timeout.
        extras = {'time_outs': output['truncated'] & ~output['terminated'],
                  'amp_next': extract_features(self.task.env.current)}
        if held is not None:
            switched = (output['obs'][:, 210:213] != held).any(-1) & ~done
            if bool(switched.any()):
                # Capture the post-action state under the held command before any selected reset.
                critic = output['critic'].clone()
                critic[:, 210:213] = held
                critic = scale_observation(critic, self.observation_scaling, critic=True)
                if self.gait_clock:
                    clock = clock_features(self.task.episode_steps, self.gait_clock, held).to(critic)
                    critic = torch.cat((critic, clock), -1)
                extras['command_time_outs'] = switched
                held_obs = output['obs'].clone()
                held_obs[:, 210:213] = held
                groups = {'critic': critic, 'privileged': self.privileged_state(output), **nets.actor_observation(held_obs)}
                extras['command_bootstrap_observation'] = self.tensor_dict(groups, batch_size=[self.num_envs])
            extras['time_outs'] = extras['time_outs'] | switched
        selected = done.nonzero(as_tuple=False).flatten()
        self.reset_rows[:] = False
        self.current = self.task.reset(selected) if len(selected) else output
        self.reset_rows[selected] = True
        if held is not None:
            done = done | switched
        return self.get_observations(), rewards, done.long(), extras

def amp_ppo_config(seed, *, networks='mlp', amp_config=None, action_mean='unbounded', **learner_options):
    """Stock PPO settings plus the AMP algorithm and the chosen Table III or MLP networks."""
    config = ppo_config(seed, action_mean=action_mean, **learner_options)
    amp_config = (AMPConfig() if amp_config is None else amp_config).resolve(seed)
    amp_config.validate()
    # Checkpoint records pass through JSON; a tuple here would never equal its saved list.
    config['algorithm'].update(class_name='locomotion.amp.amp:AMPPPO', amp_cfg=json_shape(asdict(amp_config)))
    if networks == 'paper':
        if learner_options.get('gait_clock', 0) or learner_options.get('observation_scaling', 'none') != 'none' or learner_options.get('velocity_noise', 0.):
            raise ValueError('Paper networks require their recorded raw observation groups without gait-clock inputs')
        config['obs_groups'] = {'actor': list(nets.ACTOR_GROUPS), 'critic': list(nets.CRITIC_GROUPS)}
        config['actor'].update(class_name='locomotion.amp.networks:PaperActor', hidden_dims=list(nets.ACTOR_HIDDEN))
        config['critic'].update(class_name='locomotion.amp.networks:PaperCritic', hidden_dims=list(nets.CRITIC_HIDDEN))
    elif networks != 'mlp':
        raise ValueError("networks must be 'mlp' or 'paper'")
    config['networks'] = networks
    return config


class AMPPPO(CappedRatePPO):
    """PPO with an online least-squares discriminator, style reward and supervised velocity estimator."""

    def __init__(self, actor, critic, storage, *, amp_cfg, **kwargs):
        super().__init__(actor, critic, storage, learning_rate_max=kwargs.pop("learning_rate_max", 1e-2), **kwargs)
        if self.rnd is not None or self.symmetry is not None or self.is_multi_gpu:
            raise ValueError('AMP PPO supports neither RND, symmetry nor multi-GPU training')
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError('AMP PPO supports feed-forward models only')
        self.amp_config = AMPConfig(**{**amp_cfg, 'discriminator_hidden': tuple(amp_cfg['discriminator_hidden'])})
        self.amp_config.validate()
        if self.amp_config.seed is None:
            raise ValueError('AMP discriminator seed is unresolved; build the configuration with amp_ppo_config')
        self.demonstrations, self.dataset_identity = load_demonstrations(ROOT/self.amp_config.dataset)
        self.demonstrations = self.demonstrations.to(self.device)
        std = self.demonstrations.std(0).clamp_min(1e-3)
        # Initialize on a forked generator: the discriminator must not advance the global draws
        # that stock PPO's action sampling and mini-batch order consume.
        with torch.random.fork_rng(devices=[]):
            # Seed only the CPU generator: torch.manual_seed also reseeds every CUDA device,
            # and the fork restores CPU state alone.
            torch.random.default_generator.manual_seed(self.amp_config.seed)
            self.discriminator = disc.Discriminator(self.demonstrations.mean(0).cpu(), std.cpu(),
                                                   self.amp_config.discriminator_hidden)
        self.discriminator = self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.amp_config.discriminator_learning_rate)
        self.estimator_optimizer = None
        if hasattr(self.actor, 'estimator_parameters'):
            self.estimator_optimizer = torch.optim.Adam(self.actor.estimator_parameters(),
                                                        lr=self.amp_config.estimator_learning_rate)
        self.generator = torch.Generator(device='cpu').manual_seed(self.amp_config.seed)
        self.policy_transitions = torch.zeros(storage.num_transitions_per_env, storage.num_envs,
                                              2 * disc.AMP_WIDTH, device=self.device)
        self.style_rewards = torch.zeros(storage.num_envs, device=self.device)
        self.style_sum, self.style_count = 0., 0
        self._amp_before = None

    def act(self, obs):
        self._amp_before = obs['amp'].clone()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        if self._amp_before is None or 'amp_next' not in extras:
            raise ValueError('AMP transitions need the pre-step state and the pre-reset next state')
        pair = disc.transitions(self._amp_before, extras['amp_next'].to(self.device))
        with torch.no_grad():
            self.style_rewards = disc.style_reward(self.discriminator(pair))
        self.policy_transitions[self.storage.step] = pair
        self.style_sum += float(self.style_rewards.sum())
        self.style_count += int(self.style_rewards.numel())
        super().process_env_step(obs, rewards + self.amp_config.style_weight * self.style_rewards, dones, extras)

    def _update_discriminator(self):
        policy = self.policy_transitions[:self.storage.num_transitions_per_env].reshape(-1, 2 * disc.AMP_WIDTH)
        totals = {'prior': 0., 'policy': 0., 'gradient_penalty': 0.}
        batch = self.amp_config.discriminator_batch
        self.discriminator.train()
        for _ in range(self.amp_config.discriminator_updates):
            prior_rows = torch.randint(len(self.demonstrations), (batch,), generator=self.generator).to(self.device)
            policy_rows = torch.randint(len(policy), (batch,), generator=self.generator).to(self.device)
            loss, terms = discriminator_loss(self.discriminator, self.demonstrations[prior_rows],
                                                 policy[policy_rows], self.amp_config.gradient_penalty)
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            self.discriminator_optimizer.step()
            for key, value in terms.items():
                totals[key] += value
        self.discriminator.eval()
        count = max(1, self.amp_config.discriminator_updates)
        return {'discriminator_' + key: value / count for key, value in totals.items()}

    def _update_estimator(self):
        if self.estimator_optimizer is None:
            return {}
        total, count = 0., 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, 1):
            loss = self.actor.estimator_loss(batch.observations)
            self.estimator_optimizer.zero_grad()
            loss.backward()
            self.estimator_optimizer.step()
            total += float(loss.detach())
            count += 1
        return {'velocity_estimator': total / max(1, count)}

    def update(self):
        discriminator = self._update_discriminator()
        loss_dict = super().update()
        estimator = self._update_estimator()
        style = {'style_reward_mean': self.style_sum / max(1, self.style_count)}
        self.style_sum, self.style_count = 0., 0
        return {**loss_dict, **discriminator, **estimator, **style}

    def train_mode(self):
        super().train_mode()
        self.discriminator.eval()

    def eval_mode(self):
        super().eval_mode()
        self.discriminator.eval()

    def save(self):
        saved = super().save()
        saved['discriminator_state_dict'] = self.discriminator.state_dict()
        saved['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        saved['amp_dataset_identity'] = dict(self.dataset_identity)
        if self.estimator_optimizer is not None:
            saved['estimator_optimizer_state_dict'] = self.estimator_optimizer.state_dict()
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        result = super().load(loaded_dict, load_cfg, strict)
        if loaded_dict.get('amp_dataset_identity') != self.dataset_identity:
            raise ValueError('Checkpoint demonstration bank differs')
        # As stock PPO does, restore only what the caller names; no load_cfg restores everything.
        if load_cfg is None or load_cfg.get('discriminator'):
            self.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=strict)
            self.discriminator_optimizer.load_state_dict(loaded_dict['discriminator_optimizer_state_dict'])
        if self.estimator_optimizer is not None and (load_cfg is None or load_cfg.get('optimizer')):
            self.estimator_optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        return result

    def declaration(self):
        return {'schema': 'hexapod_amp_learner_v1', 'config': json_shape(asdict(self.amp_config)),
                'dataset': dict(self.dataset_identity), 'feature_contract': feature_contract(),
                'stage2_complete': False}
