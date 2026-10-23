"""Wrap a manager-based walking environment for RSL-RL with the input recipe and command bootstraps of ``OmniVecEnv``."""

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from locomotion.rl.wrapper import check_options, command_switch, policy_inputs, smoothed_action


class ManagerVecEnv(RslRlVecEnvWrapper):
    """Stock RSL-RL wrapper that masks timeouts on physical terminations and applies the recorded options.

    The environment resets finished rows inside its step, so the returned observation is the post-reset one,
    as ``OmniVecEnv`` returns after its selected reset.
    """

    def __init__(
        self,
        env,
        *,
        observation_scaling="none",
        command_segments="continuous",
        gait_clock=0,
        action_smoothing="none",
        velocity_noise=0.0,
        command_name="velocity",
    ):
        check_options(observation_scaling, command_segments, action_smoothing)
        self.observation_scaling, self.command_segments = observation_scaling, command_segments
        self.gait_clock, self.action_smoothing, self.velocity_noise = (
            gait_clock,
            action_smoothing,
            velocity_noise,
        )
        self.command_name = command_name
        super().__init__(env)
        self.previous_action = torch.zeros(self.num_envs, 18, device=self.device)

    def observations(self, obs):
        policy, critic = policy_inputs(
            obs["policy"],
            obs["critic"],
            self.unwrapped.episode_length_buf,
            scaling=self.observation_scaling,
            velocity_noise=self.velocity_noise,
            gait_clock=self.gait_clock,
        )
        return TensorDict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def get_observations(self):
        return self.observations(self.unwrapped.observation_manager.compute())

    def reset(self):
        obs, extras = self.env.reset()
        return self.observations(obs), extras

    def step(self, actions):
        env = self.unwrapped
        held = (
            env.command_manager.get_command(self.command_name).clone()
            if self.command_segments == "bootstrap"
            else None
        )
        if self.action_smoothing == "mean2":
            # PPO keeps the raw sample; the environment receives the two-control mean.
            raw = actions.detach().clone()
            actions = smoothed_action(
                actions, self.previous_action.to(actions), env.episode_length_buf
            )
            self.previous_action = raw
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        done = terminated | truncated
        # The environment returns one extras dictionary on every step, so copy it before adding this step's keys.
        # A physical termination overrides a coincident episode timeout.
        extras = {**extras, "time_outs": truncated & ~terminated}
        if held is not None:
            switched, bootstrap = command_switch(
                obs["policy"],
                obs["critic"],
                held,
                done,
                env.episode_length_buf,
                scaling=self.observation_scaling,
                gait_clock=self.gait_clock,
                tensor_dict=TensorDict,
            )
            extras.update(bootstrap)
            extras["time_outs"] = extras["time_outs"] | switched
            done = done | switched
        return self.observations(obs), rewards, done.long(), extras
