"""Bounded collector around actual RSL PPO; no implicit checkpoint overwrite."""
import time
import torch
from rsl_rl.runners import OnPolicyRunner


class MaskedRunner(OnPolicyRunner):
    def collect_updates(self, count, *, maximum_updates, after_update=None):
        if type(count) is not int or count < 1 or maximum_updates not in (2, 10, 25):
            raise ValueError('Explicit bounded smoke or decision-chunk allocation required')
        if self.current_learning_iteration + count > maximum_updates:
            raise ValueError('Requested updates exceed current allocation')
        if self.cfg['num_steps_per_env'] != 256:
            raise ValueError('Moving collector requires exactly256 controls per update')
        if self.is_distributed:
            raise ValueError('First moving collector is single-device only')
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        reports = []
        for _ in range(count):
            start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(256):
                    action = self.alg.act(obs)
                    obs, reward, done, extras = self.env.step(action.to(self.env.device))
                    if any(not torch.isfinite(x).all() for x in obs.values()) or not torch.isfinite(reward).all():
                        raise RuntimeError('Nonfinite rollout: no optimization or checkpoint')
                    obs, reward, done = obs.to(self.device), reward.to(self.device), done.to(self.device)
                    self.alg.process_env_step(obs, reward, done, extras)
                self.alg.compute_returns(obs)
            collected = time.perf_counter()
            valid = int(self.alg.storage.learnable.sum())
            terminal = int(self.alg.storage.terminated.sum())
            timeout = int(self.alg.storage.truncated.sum())
            losses = self.alg.update()
            if any(not __import__('math').isfinite(v) for v in losses.values()):
                raise RuntimeError('Nonfinite PPO loss: no decision checkpoint')
            self.current_learning_iteration += 1
            report = {'updates_completed': self.current_learning_iteration, 'controls_per_replica': 256,
                'valid_learning_transitions': valid, 'finite_terminal_transitions': terminal,
                'time_limit_transitions': timeout, 'recovery_transitions_excluded': 256*self.env.num_envs-valid,
                'collect_s': collected-start, 'learn_s': time.perf_counter()-collected, 'losses': losses}
            reports.append(report)
            if after_update is not None:
                after_update(report)
        return reports
