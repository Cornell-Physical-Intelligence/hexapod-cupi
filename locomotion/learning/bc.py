"""Clone Shaurya's scripted forward/backward tripod with the shared PPO actor.

The teacher adapts hexapod/tools/bc_clone.py to navigation forward=-bodyY.
This experiment supplies no omnidirectional walking or native acceptance evidence.
"""
from __future__ import annotations

import math
import torch

from locomotion.backends.isaac.config import JOINT_NAMES, LEGS


class ScriptedTripodTeacher:
    """Label the current command and next control phase in raw policy action units."""

    def __init__(self, joint_names, gait_clock):
        if len(joint_names) != 18 or set(joint_names) != set(JOINT_NAMES):
            raise ValueError('BC requires the 18 approved joint names')
        if type(gait_clock) is not int or not 10 <= gait_clock <= 250:
            raise ValueError('BC requires the shared gait clock from 10 to 250 controls')
        self.period = gait_clock
        self.indices = [[joint_names.index(f'{leg}_{joint}') for leg in LEGS]
                        for joint in ('coxa_yaw', 'femur_pitch', 'tibia_pitch')]

    def actions(self, commands, episode_steps):
        if commands.ndim != 2 or commands.shape[1] != 3 or episode_steps.shape != commands.shape[:1]:
            raise ValueError('BC requires command rows and matching episode controls')
        if not bool(torch.isfinite(commands).all()) or bool((commands[:, 1:] != 0).any()):
            raise ValueError('The scripted BC teacher supports forward/backward and zero commands alone')
        phase = (episode_steps.to(commands) + 1)[:, None] / self.period
        offsets = commands.new_tensor([0., .5, 0., .5, 0., .5])
        cycle = (phase + offsets).remainder(1.)
        stance = cycle < .5
        sweep = torch.where(stance, 1 - 4 * cycle, 4 * cycle - 3)
        lift = torch.where(stance, 0., torch.sin(math.pi * (2 * cycle - 1).clamp(0, 1)))
        forward = commands[:, 0]
        moving = (forward != 0).to(commands)
        hip = (forward.abs() / .45).clamp(max=.35) * forward.sign()
        side = commands.new_tensor([-1., -1., -1., 1., 1., 1.])
        result = commands.new_zeros((len(commands), 18))
        result[:, self.indices[0]] = hip[:, None] * sweep * side / .35
        result[:, self.indices[1]] = moving[:, None] * lift
        result[:, self.indices[2]] = moving[:, None] * (-.30 / .35) * lift
        return result


def collect(env, actor, teacher, steps, noise, *, drive_actor=False, check_deadline=lambda: None):
    """Label each visited observation after the task updates commands and reset rows."""
    env.current = env.task.reset()
    env.previous_action.zero_()
    obs = env.get_observations()
    samples, labels = [], []
    for _ in range(steps):
        check_deadline()
        label = teacher.actions(env.task.commands, env.task.episode_steps)
        samples.append(obs.detach().clone())
        labels.append(label.detach().clone())
        with torch.no_grad():
            action = actor(obs) if drive_actor else label
            obs, _, _, _ = env.step(action + noise * torch.randn_like(action))
    return torch.cat(samples), torch.cat(labels)


def fit(actor, observations, labels, epochs, learning_rate, check_deadline=lambda: None):
    """Fit the configured deterministic mean; keep its normalization and action transform."""
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    actor.train()
    actor.update_normalization(observations)
    actor.eval()
    for _ in range(epochs):
        order = torch.randperm(len(labels), device=labels.device)
        for indices in order.split(4096):
            check_deadline()
            loss = (actor(observations[indices]) - labels[indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        loss = (actor(observations) - labels).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('BC produced a nonfinite fit loss')
    return float(loss)


def reset_ppo_optimizer(runner):
    """Discard moments from the prior actor and start the PPO iteration count at zero."""
    runner.alg.optimizer.state.clear()
    runner.alg.optimizer.zero_grad(set_to_none=True)
    runner.current_learning_iteration = 0


def run_bc(runner, env, *, steps=600, epochs=30, dagger_rounds=2, noise=.05, learning_rate=.001,
           check_deadline=lambda: None):
    """Run teacher collection and DAgger on the guarded Direct candidate's shared actor."""
    if any(type(v) is not int or v < 1 for v in (steps, epochs)) or type(dagger_rounds) is not int or dagger_rounds < 0:
        raise ValueError('BC needs positive steps/epochs and nonnegative DAgger rounds')
    if not math.isfinite(noise) or noise < 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('BC noise and learning rate exceed their domains')
    teacher = ScriptedTripodTeacher(env.task.joint_names, env.gait_clock)
    actor = runner.alg.actor
    initial_training = actor.training
    original_bank = env.task.bank
    env.task.bank = original_bank.new_tensor([[.05, 0., 0.], [-.05, 0., 0.], [0., 0., 0.]])
    try:
        actor.eval()
        observations, labels = collect(env, actor, teacher, steps, noise, check_deadline=check_deadline)
        losses = [fit(actor, observations, labels, epochs, learning_rate, check_deadline)]
        for _ in range(dagger_rounds):
            new_obs, new_labels = collect(env, actor, teacher, steps, noise, drive_actor=True,
                                          check_deadline=check_deadline)
            observations, labels = torch.cat((observations, new_obs)), torch.cat((labels, new_labels))
            losses.append(fit(actor, observations, labels, epochs, learning_rate, check_deadline))
    finally:
        actor.train(initial_training)
        env.task.bank = original_bank
        env.current = env.task.reset()
        env.previous_action.zero_()
    reset_ppo_optimizer(runner)
    return {'schema': 'hexapod_direct_bc_v1', 'teacher': 'Shaurya hexapod/tools/bc_clone.py scripted tripod',
            'teacher_source_revision': '969ea5bc',
            'command_scope': 'navigation forward/backward and zero; lateral/yaw unsupported',
            'samples': len(labels), 'fit_mse': losses, 'steps': steps, 'epochs': epochs,
            'dagger_rounds': dagger_rounds, 'noise': noise, 'learning_rate': learning_rate,
            'gait_clock': env.gait_clock, 'optimizer_reset': True, 'native_acceptance': False}
