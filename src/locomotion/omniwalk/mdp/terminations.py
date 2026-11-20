"""End an episode on a fall, a tilt or a joint-limit violation, and time it out after its declared length."""

import math

FALL_HEIGHT_M = 0.045
TILT_LIMIT_RAD = 0.85
JOINT_LIMIT_SLACK_RAD = 2e-6


def physical_termination(state, lower, upper):
    """True where the root falls below 4.5 cm, tilts past 0.85 rad or a joint passes its limit by 2e-6 rad."""
    joint_violation = (
        (state["q"] < lower - JOINT_LIMIT_SLACK_RAD) | (state["q"] > upper + JOINT_LIMIT_SLACK_RAD)
    ).any(-1)
    return (
        (state["root"][:, 2] < FALL_HEIGHT_M)
        | (state["gravity"][:, 2] > -math.cos(TILT_LIMIT_RAD))
        | joint_violation
    )


def episode_timeout(episode_steps, episode_seconds, control_dt):
    """True where an episode has run its declared number of controls."""
    return episode_steps >= round(episode_seconds / control_dt)
