"""The destination inputs beyond the task's five: ghost_v10/v11, the terrain blocks and leg odometry.

``DestinationVecEnv`` (``rl/destination_wrapper.py``) shows these to the actor and the critic. The
locomotion.rl package imports nothing from locomotion.omniwalk, so the task builds this object
(``DestinationTrainingTask.destination_input_source``) and the wrapper only joins tensors.

``ghost`` is the task's own five inputs. ``ghost_v10`` and ``ghost_v11`` are the sideways gap,
heading gap, age and velocity relative to the ghost (v11 adds the along-track gap; nine values).
``terrain_v1`` appends the 30 footfall-lookahead values of ``mdp/footfall.py`` after the nine
ghost_v11 values (zeros without a terrain, so flat suites and the native robot stay valid);
``terrain_v2`` appends its 6-point forward height profile after those. ``terrain_v2_critic`` gives
both terrain blocks to the critic alone, ``terrain_v2_ghost_critic`` gives the nine ghost values
to the critic alone, and ``terrain_v2_odom`` shows the actor the same nine features computed from
leg odometry (``mdp/odometry.py``) while the critic keeps the simulator's ghost.
"""

import torch

from locomotion.omniwalk.mdp.destination import GHOST_V11_FEATURES, ghost_v10_features, planar_pose

INPUTS = (
    "ghost",
    "ghost_v10",
    "ghost_v11",
    "terrain_v1",
    "terrain_v2",
    "terrain_v2_critic",
    "terrain_v2_ghost_critic",
    "terrain_v2_odom",
)
TERRAIN_INPUTS = INPUTS[3:]


class DestinationInputs:
    """One task's destination inputs for one ``destination_input`` name."""

    def __init__(self, task, destination_input, gait_clock_scaling=None):
        if destination_input not in INPUTS:
            raise ValueError(
                "DestinationVecEnv applies the ghost, ghost_v10, ghost_v11, terrain_v1, "
                "terrain_v2, terrain_v2_critic, terrain_v2_ghost_critic or terrain_v2_odom input"
            )
        self.task = task
        self.destination_input = destination_input
        self.footfall = None
        # terrain_v2_critic gives the terrain values to the critic alone: the actor keeps the nine
        # ghost_v11 inputs it could run on the robot with, the ablation docs/TRAINING.md asks for.
        self.critic_only = destination_input == "terrain_v2_critic"
        # terrain_v2_ghost_critic is the inverse split: the nine ghost values come from the
        # simulator's world pose, which the robot cannot supply, so the actor keeps only the
        # terrain blocks and the ghost stays the critic's privileged signal.
        self.ghost_critic = destination_input == "terrain_v2_ghost_critic"
        # terrain_v2_odom replaces the actor's nine world-pose ghost values with the same nine
        # features computed from leg odometry (mdp/odometry.py) — quantities the robot has — while
        # the critic keeps the true privileged ghost. Same widths, so terrain_v2 policies warm-start.
        self.odometry = None
        self._odometry_fresh = True
        if destination_input in TERRAIN_INPUTS:
            from locomotion.omniwalk.mdp.footfall import FootfallInputs

            # The recorded option names the clock; a task may carry none (the flat suites
            # evaluate every checkpoint under the v9 task, whose reward predates the clock).
            self.footfall = FootfallInputs(
                task, gait_clock_scaling, profile=destination_input != "terrain_v1"
            )
        if destination_input == "terrain_v2_odom":
            from locomotion.omniwalk.mdp.odometry import OdometryGhost

            self.odometry = OdometryGhost(task.num_envs, task.device)

    def features(self, pending, command=None):
        """The critic's inputs; rows in ``pending`` read a command that has not run a control yet: the
        neutral values for ``ghost``, and a destination at the body for ``ghost_v10`` and ``ghost_v11``.
        ``command`` defaults to the task's current commands."""
        task = self.task
        if self.destination_input == "ghost":
            return task.destination_inputs(pending)
        pending = pending[:, None].to(task.destination.device)
        here = planar_pose(task.current["root"]).to(task.destination)
        ghost = ghost_v10_features(
            task.current["root"],
            task.control_start_pose,
            torch.where(pending, here, task.destination),
            torch.where(
                pending[:, 0], torch.zeros_like(task.command_elapsed), task.command_elapsed
            ),
            task.commands if command is None else command,
            along_track=self.destination_input != "ghost_v10",
        )
        if self.footfall is None:
            return ghost
        # terrain_v1: the nine ghost_v11 values, then the 30 footfall-lookahead values.
        return torch.cat((ghost, self.footfall.features(command).to(ghost)), -1)

    def actor_view(self, extra):
        """The part of the critic's inputs ``extra`` that the actor sees."""
        if self.critic_only:
            return extra[..., :GHOST_V11_FEATURES]
        if self.ghost_critic:
            return extra[..., GHOST_V11_FEATURES:]
        if self.odometry is not None:
            task = self.task
            odom = self.odometry.features(task.command_started, task.command_elapsed, task.commands)
            return torch.cat((odom.to(extra), extra[..., GHOST_V11_FEATURES:]), -1)
        return extra

    def advance(self, held_command, pre_started, done):
        """After a control: advance the leg odometry; True when the observations must be rebuilt."""
        if self.odometry is None:
            return False
        from locomotion.omniwalk.mdp.odometry import observed

        task = self.task
        root, toes, contact = observed(task.env)
        every = torch.arange(task.num_envs, device=task.device)
        if self._odometry_fresh:
            self.odometry.reset(every, root, toes, contact)
            self._odometry_fresh = False
        # The executed command and the pre-step start flags mirror _advance_destination's
        # timing; rows the learner reset re-align their odometry frame with the fresh pose.
        self.odometry.advance(root, toes, contact, held_command, pre_started)
        reset_rows = every[done.bool().to(task.device)]
        if len(reset_rows):
            self.odometry.reset(reset_rows, root, toes, contact)
        return True
