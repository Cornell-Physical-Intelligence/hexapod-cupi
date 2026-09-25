"""Online adversarial motion prior for RSL-RL PPO (docs/TRAINING.md Step 5).

``AMPVecEnv`` extends the vanilla adapter with the 61-value AMP state, the
pre-reset next state of every transition, the actor's named observation groups
and a 42-value privileged state for the critic. ``AMPPPO`` subclasses stock
PPO: it adds the Eq. (2) style reward to every task reward, trains the Eq. (1)
discriminator against the admitted demonstration bank on each update, and
trains the paper actor's velocity estimator by supervised regression. Stock PPO
losses, storage and checkpoint fields are unchanged; ``test_amp_ppo`` checks
parity with plain PPO when the style weight and discriminator updates are zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rsl_rl.algorithms import PPO

from . import amp
from .ppo import VanillaVecEnv, ppo_config
from . import paper_networks as nets

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT/'locomotion/priors/datasets/amp_demonstrations_001'
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
DECISIONS = (
    "Style reward weight 1 follows Table I; the total reward is task plus style plus penalties.",
    "The discriminator trains on this rollout's policy transitions against uniform dataset samples; "
    "no replay buffer of older policy transitions.",
    "Discriminator updates per PPO iteration equal num_learning_epochs * num_mini_batches; batch 256 "
    "prior and 256 policy transitions; Adam 1e-4; these are not stated in the paper.",
    "A terminal transition pairs the last state with the terminal state before reset; reset-to-first "
    "pairs are never formed because the AMP state after a reset starts the next pair.",
    "The velocity estimator trains after each PPO update on the stored rollout with Adam 1e-3, one "
    "pass over the same mini-batches, against measured body-frame base velocity.",
    "Privileged friction and perturbation slots hold declared constants until Step 7 supplies them.",
)


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
    seed: int = 20260925

    def validate(self):
        if not all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in (
                self.style_weight, self.gradient_penalty, self.discriminator_learning_rate,
                self.estimator_learning_rate)):
            raise ValueError('AMP coefficients must be finite and nonnegative')
        if (type(self.discriminator_batch) is not int or self.discriminator_batch < 1
                or type(self.discriminator_updates) is not int or self.discriminator_updates < 0
                or type(self.seed) is not int or self.seed < 0 or not self.discriminator_hidden):
            raise ValueError('Invalid AMP discriminator schedule')
        path = Path(self.dataset)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Dataset path must be relative to the repository root')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_demonstrations(directory=DATASET):
    """Load the admitted transition bank and bind its manifest identity."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema') != 'hexapod_amp_dataset_v1' or manifest.get('admitted') is not True
            or manifest.get('feature_contract') != amp.feature_contract()):
        raise ValueError('Demonstration bank is not an admitted bank under the current feature contract')
    file_sha = sha(directory/'transitions.npz')
    if file_sha != manifest.get('file_sha256'):
        raise ValueError('Demonstration transitions differ from their manifest')
    with np.load(directory/'transitions.npz', allow_pickle=False) as archive:
        states, next_states = archive['states'], archive['next_states']
    if (states.shape != next_states.shape or states.ndim != 2 or states.shape[1] != amp.AMP_WIDTH
            or len(states) != manifest.get('transitions') or not np.isfinite(states).all()
            or not np.isfinite(next_states).all()):
        raise ValueError('Demonstration transitions differ from the 61-value contract')
    identity = {'path': str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
                'manifest_sha256': sha(directory/'manifest.json'), 'transitions_sha256': file_sha,
                'transitions': int(len(states)), 'model_sha256': manifest.get('model_sha256'),
                'reviewer': manifest.get('reviewer')}
    return amp.transitions(states, next_states), identity


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


class AMPVecEnv(VanillaVecEnv):
    """Vanilla adapter plus AMP states, named actor groups and the critic's privileged state."""

    def __init__(self, task, tensor_dict=None, *, collision=None, amp_config=None):
        self.collision = collision
        self.amp_config = AMPConfig() if amp_config is None else amp_config
        self.amp_config.validate()
        self.reset_rows = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        super().__init__(task, tensor_dict)
        if 'amp' not in self.current:
            raise ValueError('AMP training needs record_motion_features in the environment configuration')
        self.cfg.update(motion_prior=True, amp=asdict(self.amp_config), privileged_layout=[
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
        obs = output['obs']
        groups = {'policy': obs, 'critic': output['critic'], 'amp': output['amp'],
                  'velocity_label': output['critic'][:, OBSERVATION_WIDTH:OBSERVATION_WIDTH + 3],
                  'privileged': self.privileged_state(output)}
        groups.update(nets.actor_observation(obs).items())
        return self.tensor_dict(groups, batch_size=[self.num_envs])

    def step(self, action):
        output = self.task.step(action)
        rewards = output['reward'].clone()
        done = output['terminated'] | output['truncated']
        extras = {'time_outs': output['truncated'] & ~output['terminated'], 'amp_next': output['amp'].clone()}
        selected = done.nonzero(as_tuple=False).flatten()
        self.reset_rows[:] = False
        if len(selected):
            self.current = self.task.reset(selected)
            self.reset_rows[selected] = True
        else:
            self.current = output
        return self.get_observations(), rewards, done.long(), extras


def amp_ppo_config(seed, *, networks='mlp', amp_config=None):
    """Stock PPO settings plus the AMP algorithm and the chosen Table III or MLP networks."""
    config = ppo_config(seed)
    amp_config = AMPConfig() if amp_config is None else amp_config
    amp_config.validate()
    config['algorithm'].update(class_name='locomotion.amp_ppo:AMPPPO', amp_cfg=asdict(amp_config))
    if networks == 'paper':
        config['obs_groups'] = {'actor': list(nets.ACTOR_GROUPS), 'critic': list(nets.CRITIC_GROUPS)}
        config['actor'].update(class_name='locomotion.paper_networks:PaperActor', hidden_dims=list(nets.ACTOR_HIDDEN))
        config['critic'].update(class_name='locomotion.paper_networks:PaperCritic', hidden_dims=list(nets.CRITIC_HIDDEN))
    elif networks != 'mlp':
        raise ValueError("networks must be 'mlp' or 'paper'")
    config['networks'] = networks
    return config


class AMPPPO(PPO):
    """PPO with an online least-squares discriminator, style reward and supervised velocity estimator."""

    def __init__(self, actor, critic, storage, *, amp_cfg, **kwargs):
        super().__init__(actor, critic, storage, **kwargs)
        if self.rnd is not None or self.symmetry is not None or self.is_multi_gpu:
            raise ValueError('AMP PPO supports neither RND, symmetry nor multi-GPU training')
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError('AMP PPO supports feed-forward models only')
        self.amp_config = AMPConfig(**{**amp_cfg, 'discriminator_hidden': tuple(amp_cfg['discriminator_hidden'])})
        self.amp_config.validate()
        self.demonstrations, self.dataset_identity = load_demonstrations(ROOT/self.amp_config.dataset)
        self.demonstrations = self.demonstrations.to(self.device)
        std = self.demonstrations.std(0).clamp_min(1e-3)
        self.discriminator = amp.Discriminator(self.demonstrations.mean(0), std,
                                               self.amp_config.discriminator_hidden).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                                        lr=self.amp_config.discriminator_learning_rate)
        self.estimator_optimizer = None
        if hasattr(self.actor, 'estimator_parameters'):
            self.estimator_optimizer = torch.optim.Adam(self.actor.estimator_parameters(),
                                                        lr=self.amp_config.estimator_learning_rate)
        self.generator = torch.Generator(device='cpu').manual_seed(self.amp_config.seed)
        self.policy_transitions = torch.zeros(storage.num_transitions_per_env, storage.num_envs,
                                              2 * amp.AMP_WIDTH, device=self.device)
        self.style_rewards = torch.zeros(storage.num_envs, device=self.device)
        self.style_sum, self.style_count = 0., 0
        self._amp_before = None

    def act(self, obs):
        self._amp_before = obs['amp'].clone()
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        if self._amp_before is None or 'amp_next' not in extras:
            raise ValueError('AMP transitions need the pre-step state and the pre-reset next state')
        pair = amp.transitions(self._amp_before, extras['amp_next'].to(self.device))
        with torch.no_grad():
            self.style_rewards = amp.style_reward(self.discriminator(pair))
        self.policy_transitions[self.storage.step] = pair
        self.style_sum += float(self.style_rewards.sum())
        self.style_count += int(self.style_rewards.numel())
        super().process_env_step(obs, rewards + self.amp_config.style_weight * self.style_rewards, dones, extras)

    def _update_discriminator(self):
        policy = self.policy_transitions[:self.storage.num_transitions_per_env].reshape(-1, 2 * amp.AMP_WIDTH)
        totals = {'prior': 0., 'policy': 0., 'gradient_penalty': 0.}
        batch = self.amp_config.discriminator_batch
        self.discriminator.train()
        for _ in range(self.amp_config.discriminator_updates):
            prior_rows = torch.randint(len(self.demonstrations), (batch,), generator=self.generator).to(self.device)
            policy_rows = torch.randint(len(policy), (batch,), generator=self.generator).to(self.device)
            loss, terms = amp.discriminator_loss(self.discriminator, self.demonstrations[prior_rows],
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
        if load_cfg is None or load_cfg.get('discriminator', True):
            self.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=strict)
            self.discriminator_optimizer.load_state_dict(loaded_dict['discriminator_optimizer_state_dict'])
        if self.estimator_optimizer is not None and (load_cfg is None or load_cfg.get('optimizer', True)):
            self.estimator_optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        return result

    def declaration(self):
        return {'schema': 'hexapod_amp_learner_v1', 'config': asdict(self.amp_config),
                'dataset': dict(self.dataset_identity), 'feature_contract': amp.feature_contract(),
                'discriminator_decisions': list(amp.DECISIONS), 'learner_decisions': list(DECISIONS),
                'network_decisions': list(nets.DECISIONS), 'stage2_complete': False}
