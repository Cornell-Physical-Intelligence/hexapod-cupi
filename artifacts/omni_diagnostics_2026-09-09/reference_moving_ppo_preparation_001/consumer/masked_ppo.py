"""RSL 5.0.1 PPO with explicit per-row learning and final-observation masks.

The optimization loss is the installed PPO loss. Only collection, normalization,
return construction and minibatch selection change. Recovery controls never
enter these four paths. Finite terminal transitions DO remain learning data.
This is a new training lineage, not an evaluation-gate change.
"""
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage


def boolean(name, value, n, device):
    if not isinstance(value, torch.Tensor) or value.shape != (n,) or value.dtype != torch.bool or value.device != device:
        raise ValueError('Expected device-local bool[N]: ' + name)
    return value


class MaskedStorage(RolloutStorage):
    def __init__(self, old):
        if old.step != 0 or old.training_type != 'rl':
            raise ValueError('Only a fresh RL rollout storage may be replaced')
        super().__init__('rl', old.num_envs, old.num_transitions_per_env,
                         old.observations[0], old.actions_shape, old.device)
        shape = (self.num_transitions_per_env, self.num_envs)
        self.learnable = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.terminated = torch.zeros_like(self.learnable)
        self.truncated = torch.zeros_like(self.learnable)
        self.episode = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.next_value = torch.zeros(shape, device=self.device)
        self.last_batch_indices = []

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        ids = torch.nonzero(self.learnable.flatten()).flatten()
        if len(ids) < 2 * num_mini_batches:
            raise RuntimeError('Insufficient valid transitions for declared PPO minibatches')
        flat = {name: getattr(self, name).flatten(0, 1) for name in
                ('observations', 'actions', 'values', 'advantages', 'returns', 'actions_log_prob')}
        params = tuple(p.flatten(0, 1) for p in self.distribution_params)
        self.last_batch_indices = []
        for _ in range(num_epochs):
            order = ids[torch.randperm(len(ids), device=self.device)]
            for idx in torch.tensor_split(order, num_mini_batches):
                self.last_batch_indices.append(idx.detach().clone())
                yield self.Batch(observations=flat['observations'][idx], actions=flat['actions'][idx],
                    values=flat['values'][idx], advantages=flat['advantages'][idx], returns=flat['returns'][idx],
                    old_actions_log_prob=flat['actions_log_prob'][idx],
                    old_distribution_params=tuple(p[idx] for p in params))


class MaskedPPO(PPO):
    def __init__(self, actor, critic, storage, **kwargs):
        super().__init__(actor, critic, MaskedStorage(storage), **kwargs)
        if self.actor.is_recurrent or self.critic.is_recurrent or self.rnd or self.symmetry or self.is_multi_gpu:
            raise ValueError('First masked lineage supports single-device feedforward PPO only')
        self._pending_mask = None
        self.normalizer_rows = 0
        self.last_return_audit = None

    def act(self, obs):
        if self._pending_mask is not None:
            raise RuntimeError('Previous action has no recorded outcome')
        n = self.storage.num_envs
        valid = boolean('learning_valid', obs['learning_valid'].squeeze(-1), n, obs.device)
        episode = obs['episode_id'].squeeze(-1)
        if episode.shape != (n,) or episode.dtype != torch.int64 or (episode < 0).any():
            raise ValueError('Exact nonnegative episode IDs required')
        if any(not torch.isfinite(v).all() for v in obs.values()):
            raise ValueError('Nonfinite actor packet, including disabled placeholders')
        # Updating only the current valid packet avoids both reset observations
        # and final-terminal observations entering normalization statistics.
        if valid.any():
            self.actor.update_normalization(obs[valid])
            self.critic.update_normalization(obs[valid])
            self.normalizer_rows += int(valid.sum())
        actions = super().act(obs)
        self._pending_mask = valid.clone()
        self._pending_episode = episode.clone()
        # Sampling on disabled finite placeholders is harmless; the executed
        # residual must be exact zero and its transition is never learned.
        return torch.where(valid[:, None], actions, torch.zeros_like(actions))

    def process_env_step(self, obs, rewards, dones, extras):
        if self._pending_mask is None:
            raise RuntimeError('No matching action')
        st = self.storage; n = st.num_envs; device = st.learnable.device
        required = {'learnable', 'terminated', 'truncated', 'fatal', 'episode_id',
                    'final_observation', 'final_observation_valid', 'final_episode_id'}
        data = extras.get('masked_transition')
        if not isinstance(data, dict) or set(data) != required:
            raise ValueError('Exact masked transition contract required')
        masks = {k: boolean(k, data[k], n, device) for k in
                 ('learnable', 'terminated', 'truncated', 'fatal', 'final_observation_valid')}
        learn = masks['learnable']; term = masks['terminated']; trunc = masks['truncated']
        if masks['fatal'].any():
            raise RuntimeError('Fatal physics/data failure: reject entire unoptimized rollout')
        if not torch.equal(learn, self._pending_mask) or not torch.equal(data['episode_id'], self._pending_episode):
            raise ValueError('Outcome mask/episode differs from action packet')
        if (term & trunc).any() or ((term | trunc) & ~learn).any():
            raise ValueError('Invalid terminal or failed recovery outcome')
        if not torch.equal(dones.bool(), term | trunc):
            raise ValueError('RSL dones differs from explicit outcome')
        need = learn & ~term
        if (need & ~masks['final_observation_valid']).any() or (need & (data['final_episode_id'] != self._pending_episode)).any():
            raise ValueError('Bootstrap must use valid final observation from the same episode')
        final = data['final_observation']
        if final.batch_size != obs.batch_size or any(not torch.isfinite(v).all() for v in final.values()):
            raise ValueError('Finite final observation batch required, even for zero-bootstrap rows')
        if rewards.shape != (n,) or not torch.isfinite(rewards).all():
            raise ValueError('Finite per-row reward required')
        next_value = torch.zeros(n, device=device)
        if need.any():
            next_value[need] = self.critic(final[need]).detach().squeeze(-1)
        if not torch.isfinite(next_value).all():
            raise RuntimeError('Nonfinite final critic value')
        i = st.step
        st.learnable[i].copy_(learn); st.terminated[i].copy_(term); st.truncated[i].copy_(trunc)
        st.episode[i].copy_(self._pending_episode); st.next_value[i].copy_(next_value)
        self.transition.rewards = torch.where(learn, rewards, 0.).clone()
        self.transition.dones = term | trunc
        st.add_transition(self.transition)
        self.transition.clear(); self._pending_mask = None
        self.actor.reset(dones); self.critic.reset(dones)

    def compute_returns(self, obs):
        st = self.storage
        if st.step != st.num_transitions_per_env or self._pending_mask is not None:
            raise ValueError('Only a complete recorded rollout may be optimized')
        carry = torch.zeros(st.num_envs, device=st.learnable.device)
        raw_adv = torch.zeros_like(st.next_value)
        for i in reversed(range(st.num_transitions_per_env)):
            chain = st.learnable[i] & ~st.terminated[i] & ~st.truncated[i]
            if i + 1 == st.num_transitions_per_env:
                chain = torch.zeros_like(chain)
            else:
                chain &= st.learnable[i + 1] & (st.episode[i] == st.episode[i + 1])
            delta = st.rewards[i, :, 0] + self.gamma * st.next_value[i] - st.values[i, :, 0]
            carry = torch.where(st.learnable[i], delta + self.gamma * self.lam * chain * carry, 0.)
            raw_adv[i].copy_(carry)
        st.returns.copy_(torch.where(st.learnable[..., None], raw_adv[..., None] + st.values, 0.))
        advantage = raw_adv.clone()
        selected = advantage[st.learnable]
        if selected.numel() < 2:
            raise RuntimeError('Too few valid PPO transitions')
        if not self.normalize_advantage_per_mini_batch:
            advantage[st.learnable] = (selected - selected.mean()) / (selected.std() + 1e-8)
        st.advantages.copy_(advantage[..., None])
        self.last_return_audit = {k: v.detach().clone() for k, v in
            {'learnable': st.learnable, 'terminated': st.terminated, 'truncated': st.truncated,
             'episode': st.episode, 'next_value': st.next_value, 'returns': st.returns,
             'raw_advantages': raw_adv}.items()}
