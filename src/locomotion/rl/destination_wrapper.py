"""Give the actor and the critic the gap to each command's destination (experiment ``destination_v1``).

``DestinationVecEnv`` is ``OmniVecEnv`` with five inputs after its own: the planar gap to the pose the
command would have reached, in the body's navigation axes over 0.1 m, the sine and cosine of the heading
gap, and the controls since the command began over 250 (``omniwalk/mdp/destination.py``). A command that
has not run a control reads the destination at the body. It needs a ``DestinationTrainingTask``;
``ppo_config`` records ``destination_input="ghost"``, so evaluation rebuilds the same inputs.
"""

import torch

from locomotion.omniwalk.mdp.destination import NEUTRAL_FEATURES, destination_features
from locomotion.rl.wrapper import OmniVecEnv


class DestinationVecEnv(OmniVecEnv):
    """``OmniVecEnv`` with the five destination inputs appended to the actor and critic inputs."""

    def __init__(self, task, tensor_dict=None, *, destination_input="ghost", **options):
        if destination_input != "ghost":
            raise ValueError("DestinationVecEnv applies the ghost destination input")
        if not hasattr(task, "destination"):
            raise ValueError("The destination inputs need a DestinationTrainingTask")
        super().__init__(task, tensor_dict, **options)

    def destination_inputs(self, pending):
        """The task's five inputs; rows in ``pending`` read the neutral values."""
        task = self.task
        current = destination_features(task.current["root"], task.destination, task.command_elapsed)
        neutral = torch.tensor(NEUTRAL_FEATURES, device=current.device).expand_as(current)
        return torch.where(pending[:, None].to(current.device), neutral, current)

    def observations(self, output):
        base = super().observations(output)
        extra = self.destination_inputs(self.task.command_started)
        policy = torch.cat((base["policy"], extra.to(base["policy"])), -1)
        critic = torch.cat((base["critic"], extra.to(base["critic"])), -1)
        return self.tensor_dict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def step(self, action):
        observation, rewards, done, extras = super().step(action)
        bootstrap = extras.get("command_bootstrap_observation")
        if bootstrap is not None:
            # A switched row has not reset: its pose is post-action and the task moves its destination
            # and time to the new command only at the next control, so these are the held command's.
            held = self.destination_inputs(torch.zeros_like(self.task.command_started))
            critic = torch.cat((bootstrap["critic"], held.to(bootstrap["critic"])), -1)
            extras["command_bootstrap_observation"] = self.tensor_dict(
                {"critic": critic}, batch_size=[self.num_envs]
            )
        return observation, rewards, done, extras
