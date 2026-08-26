"""Pure contract helpers for continuous Stage2 joystick-transition screens."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable


SCHEMA_ID = "hexapod.stage2_command_transitions.v1"
POLICY_STEP_SECONDS = 0.02
COPIES = 4
EVALUATION_SEED = 71
TRANSIENT_STEPS = 50
STEADY_STEPS = 100
SEGMENT_STEPS = TRANSIENT_STEPS + STEADY_STEPS
SETTLING_DWELL_STEPS = 10
MAXIMUM_SETTLING_TIME_SECONDS = 1.0

# A sample counts as settled only if it is both close to the target and shows
# a real signed response on every commanded axis.  This prevents a stationary
# robot from "settling" to the small pure-lateral command by absolute error
# alone.
SETTLING_PLANAR_ERROR_MPS = 0.10
SETTLING_YAW_ERROR_RADPS = 0.12
SETTLING_MINIMUM_ACTIVE_TRANSLATION_FRACTION = 0.40
SETTLING_MINIMUM_ACTIVE_YAW_FRACTION = 0.35
SETTLING_MAXIMUM_INACTIVE_TRANSLATION_MPS = 0.08
SETTLING_MAXIMUM_INACTIVE_YAW_RADPS = 0.10
COMMAND_EPSILON = 1.0e-9


@dataclass(frozen=True)
class TransitionSegment:
    """One fixed command in the uninterrupted joystick schedule."""

    key: str
    label: str
    command: tuple[float, float, float]


@dataclass(frozen=True)
class ScheduledTransitionSegment:
    """One command with exact global and local policy-step boundaries."""

    segment: TransitionSegment
    index: int
    start_step: int
    stop_step: int

    @property
    def transient_stop_step(self) -> int:
        return self.start_step + TRANSIENT_STEPS

    @property
    def previous_command(self) -> tuple[float, float, float] | None:
        if self.index == 0:
            return None
        return TRANSITION_SEGMENTS[self.index - 1].command


TRANSITION_SEGMENTS: tuple[TransitionSegment, ...] = (
    TransitionSegment("stand_initial", "INITIAL STAND", (0.00, 0.00, 0.00)),
    TransitionSegment("forward", "FORWARD", (0.25, 0.00, 0.00)),
    TransitionSegment("reverse", "REVERSE", (-0.15, 0.00, 0.00)),
    TransitionSegment("strafe_left", "STRAFE LEFT", (0.00, 0.09, 0.00)),
    TransitionSegment("strafe_right", "STRAFE RIGHT", (0.00, -0.09, 0.00)),
    TransitionSegment("yaw_left", "YAW LEFT", (0.00, 0.00, 0.24)),
    TransitionSegment("yaw_right", "YAW RIGHT", (0.00, 0.00, -0.24)),
    TransitionSegment(
        "diagonal_forward_left",
        "DIAGONAL FORWARD LEFT",
        (0.16, 0.07, 0.00),
    ),
    TransitionSegment(
        "combined_forward_right_yaw_left",
        "FORWARD + RIGHT + YAW LEFT",
        (0.16, -0.07, 0.20),
    ),
    TransitionSegment(
        "combined_reverse_left_yaw_right",
        "REVERSE + LEFT + YAW RIGHT",
        (-0.12, 0.07, -0.20),
    ),
    TransitionSegment("stand_final", "FINAL STAND", (0.00, 0.00, 0.00)),
)


def scheduled_segments() -> tuple[ScheduledTransitionSegment, ...]:
    """Return the immutable, gap-free 33-second transition schedule."""

    return tuple(
        ScheduledTransitionSegment(
            segment=segment,
            index=index,
            start_step=index * SEGMENT_STEPS,
            stop_step=(index + 1) * SEGMENT_STEPS,
        )
        for index, segment in enumerate(TRANSITION_SEGMENTS)
    )


def schedule_payload() -> list[dict[str, object]]:
    """Return the exact JSON schedule used for hashing and report validation."""

    return [
        {
            "index": scheduled.index,
            "key": scheduled.segment.key,
            "label": scheduled.segment.label,
            "command": list(scheduled.segment.command),
            "previous_command": (
                None
                if scheduled.previous_command is None
                else list(scheduled.previous_command)
            ),
            "start_step": scheduled.start_step,
            "transient_stop_step": scheduled.transient_stop_step,
            "stop_step": scheduled.stop_step,
            "transient_steps": TRANSIENT_STEPS,
            "steady_steps": STEADY_STEPS,
        }
        for scheduled in scheduled_segments()
    ]


def schedule_sha256() -> str:
    """Return a stable digest of the complete step/command schedule."""

    canonical = json.dumps(
        schedule_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def tracking_within_settling_band(
    command: tuple[float, float, float],
    achieved_linear: tuple[float, float, float],
    achieved_angular: tuple[float, float, float],
) -> bool:
    """Return whether one post-switch sample meets the settling-band contract."""

    values = (*command, *achieved_linear, *achieved_angular)
    if not all(math.isfinite(value) for value in values):
        return False
    vx, vy, yaw = command
    achieved_vx, achieved_vy = achieved_linear[:2]
    achieved_yaw = achieved_angular[2]
    if math.hypot(achieved_vx - vx, achieved_vy - vy) > SETTLING_PLANAR_ERROR_MPS:
        return False
    if abs(achieved_yaw - yaw) > SETTLING_YAW_ERROR_RADPS:
        return False

    for commanded, achieved in ((vx, achieved_vx), (vy, achieved_vy)):
        if abs(commanded) > COMMAND_EPSILON:
            if achieved * commanded <= 0.0:
                return False
            if abs(achieved) < (
                abs(commanded) * SETTLING_MINIMUM_ACTIVE_TRANSLATION_FRACTION
            ):
                return False
        elif abs(achieved) > SETTLING_MAXIMUM_INACTIVE_TRANSLATION_MPS:
            return False
    if abs(yaw) > COMMAND_EPSILON:
        if achieved_yaw * yaw <= 0.0:
            return False
        if abs(achieved_yaw) < (
            abs(yaw) * SETTLING_MINIMUM_ACTIVE_YAW_FRACTION
        ):
            return False
    elif abs(achieved_yaw) > SETTLING_MAXIMUM_INACTIVE_YAW_RADPS:
        return False
    return True


def settling_completion_step(
    settled_samples: Iterable[bool],
    *,
    dwell_steps: int = SETTLING_DWELL_STEPS,
) -> int | None:
    """Return the 1-based end step of the first consecutive settled dwell."""

    if isinstance(dwell_steps, bool) or not isinstance(dwell_steps, int) or dwell_steps < 1:
        raise ValueError("dwell_steps must be a positive integer")
    consecutive = 0
    for index, settled in enumerate(settled_samples):
        consecutive = consecutive + 1 if bool(settled) else 0
        if consecutive >= dwell_steps:
            return index + 1
    return None


TOTAL_STEPS = len(TRANSITION_SEGMENTS) * SEGMENT_STEPS
TOTAL_DURATION_SECONDS = TOTAL_STEPS * POLICY_STEP_SECONDS
SCHEDULE_SHA256 = schedule_sha256()


__all__ = [
    "COMMAND_EPSILON",
    "COPIES",
    "EVALUATION_SEED",
    "MAXIMUM_SETTLING_TIME_SECONDS",
    "POLICY_STEP_SECONDS",
    "SCHEMA_ID",
    "SCHEDULE_SHA256",
    "SEGMENT_STEPS",
    "SETTLING_DWELL_STEPS",
    "SETTLING_MAXIMUM_INACTIVE_TRANSLATION_MPS",
    "SETTLING_MAXIMUM_INACTIVE_YAW_RADPS",
    "SETTLING_MINIMUM_ACTIVE_TRANSLATION_FRACTION",
    "SETTLING_MINIMUM_ACTIVE_YAW_FRACTION",
    "SETTLING_PLANAR_ERROR_MPS",
    "SETTLING_YAW_ERROR_RADPS",
    "STEADY_STEPS",
    "TOTAL_DURATION_SECONDS",
    "TOTAL_STEPS",
    "TRANSIENT_STEPS",
    "TRANSITION_SEGMENTS",
    "ScheduledTransitionSegment",
    "TransitionSegment",
    "schedule_payload",
    "schedule_sha256",
    "scheduled_segments",
    "settling_completion_step",
    "tracking_within_settling_band",
]
