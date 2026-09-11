"""Pure NumPy observation/action adapter candidate for the detailed hexapod PPO.

Proposal only: no simulator imports, optimizer, rewards, gains, chosen scales,
navigation, or task registration. Every physical scale and limit is supplied
by the caller.

Timing contract per control tick k (dt = 0.02 s):
  1. Measure the state at the end of interval k-1. Build ActorFrameInputs using
     pipeline.previous_clipped_action and pipeline.executed_target, which both
     describe interval k-1 (already held). Call ObservationHistory.push.
  2. Read actor_observation, run the actor, call TargetPipeline.step. The
     returned executed_target is what the joint controller holds for interval k.
  3. Advance physics by dt.
An observation therefore never contains a target that has not been executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

NUM_JOINTS = 18
HISTORY_LENGTH = 5
FRAME_SIZE = 81
ACTOR_OBS_SIZE = HISTORY_LENGTH * FRAME_SIZE                 # 405
CRITIC_PRIVILEGED_SIZE = 3                                   # body linear velocity
CRITIC_OBS_SIZE = ACTOR_OBS_SIZE + CRITIC_PRIVILEGED_SIZE    # 408

FRAME_LAYOUT = {
    "angular_velocity": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "command": slice(6, 9),
    "joint_position_error": slice(9, 27),
    "joint_rate": slice(27, 45),
    "previous_action": slice(45, 63),
    "target_error": slice(63, 81),
}
assert FRAME_LAYOUT["target_error"].stop == FRAME_SIZE

# ---------------------------------------------------------------------------
# Fixed frame map
# ---------------------------------------------------------------------------
# Native robot body axes: +X left, -Y forward, +Z up.
# Policy command axes:    +X forward, +Y left, +Z up.
# command = NATIVE_TO_COMMAND @ native = (-native_y, native_x, native_z).
# Proper rotation (det +1), so angular velocities map the same way as vectors.
# Applies ONLY to vectors already expressed in the native body frame. Rotating
# world quantities into the body frame and choosing the measurement origin are
# caller responsibilities and are not part of this map.
NATIVE_TO_COMMAND = np.array([[0.0, -1.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0]])
COMMAND_TO_NATIVE = NATIVE_TO_COMMAND.T.copy()
NATIVE_TO_COMMAND.flags.writeable = False
COMMAND_TO_NATIVE.flags.writeable = False


def _vectors3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 3:
        raise ValueError(f"{name}: trailing dimension must be 3, got shape {arr.shape}")
    return arr


def native_body_to_command_frame(native_vectors) -> np.ndarray:
    """Rotate body-frame vectors (last axis xyz) from native axes to command axes."""
    return _vectors3(native_vectors, "native_vectors") @ NATIVE_TO_COMMAND.T


def command_to_native_body_frame(command_vectors) -> np.ndarray:
    """Inverse of native_body_to_command_frame."""
    return _vectors3(command_vectors, "command_vectors") @ COMMAND_TO_NATIVE.T


# ---------------------------------------------------------------------------
# Validation helpers (every accepted array is an owned float64 copy)
# ---------------------------------------------------------------------------
def _finite_array(value, shape: tuple, name: str) -> np.ndarray:
    arr = np.array(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: contains non-finite values")
    return arr


def _positive_array(value, shape: tuple, name: str) -> np.ndarray:
    arr = _finite_array(value, shape, name)
    if not np.all(arr > 0.0):
        raise ValueError(f"{name}: all entries must be positive")
    arr.flags.writeable = False
    return arr


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name}: must be a positive integer")
    return int(value)


def _positive_float(value, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name}: must be a positive finite number")
    return v


def _row_indices(rows, num_rows: int) -> np.ndarray:
    idx = np.asarray(rows)
    if idx.ndim != 1:
        raise ValueError("rows: must be a 1-D integer index array")
    if idx.size == 0:
        return np.zeros(0, dtype=np.intp)
    if not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("rows: must be a 1-D integer index array")
    if idx.min() < 0 or idx.max() >= num_rows:
        raise ValueError(f"rows: indices must lie in [0, {num_rows})")
    if np.unique(idx).size != idx.size:
        raise ValueError("rows: indices must be unique")
    return idx.astype(np.intp)


# ---------------------------------------------------------------------------
# Joint configuration and target pipeline
# ---------------------------------------------------------------------------
class JointConfig:
    """Explicit per-joint configuration in the order the controller consumes targets.

    Nothing here is chosen by the adapter; action_scale is the caller's decision.
    """

    def __init__(self, names: Sequence[str], lower, upper, neutral, action_scale):
        names = tuple(names)
        if len(names) != NUM_JOINTS:
            raise ValueError(f"names: expected {NUM_JOINTS} joint names, got {len(names)}")
        if any(not isinstance(n, str) or not n for n in names):
            raise ValueError("names: every joint name must be a non-empty string")
        if len(set(names)) != NUM_JOINTS:
            raise ValueError("names: joint names must be unique")
        shape = (NUM_JOINTS,)
        self.names = names
        self.lower = _finite_array(lower, shape, "lower")
        self.upper = _finite_array(upper, shape, "upper")
        self.neutral = _finite_array(neutral, shape, "neutral")
        if not np.all(self.lower < self.upper):
            raise ValueError("limits: lower must be strictly below upper for every joint")
        if np.any(self.neutral < self.lower) or np.any(self.neutral > self.upper):
            raise ValueError("neutral: must lie within [lower, upper] for every joint")
        self.action_scale = _positive_array(action_scale, shape, "action_scale")
        for arr in (self.lower, self.upper, self.neutral):
            arr.flags.writeable = False


class TargetSnapshot(NamedTuple):
    """Owned copies describing one step. Mutating them cannot affect the pipeline."""
    raw_action: np.ndarray        # as received, before clipping
    clipped_action: np.ndarray    # clip(raw_action, -1, 1)
    requested_target: np.ndarray  # neutral + action_scale * clipped, limited to [lower, upper]
    executed_target: np.ndarray   # requested, slewed to within max_target_step of the previous executed target


class TargetPipeline:
    """Batched, stateful normalized-action to joint-target pipeline.

    State per row: executed target (rad) and previous clipped action.
    Construction sets the executed target to neutral and the previous action to
    zero. Call reset_rows with measured positions before the first step so the
    first slew is relative to the real pose.

    No deadband, hold arbitration, filtering, acceleration limit, or gating.
    """

    def __init__(self, joints: JointConfig, num_envs: int,
                 dt: float = 0.02, max_target_step: float = 0.040):
        if not isinstance(joints, JointConfig):
            raise TypeError("joints must be a JointConfig")
        self._joints = joints
        self._num_envs = _positive_int(num_envs, "num_envs")
        self._dt = _positive_float(dt, "dt")
        self._max_step = _positive_float(max_target_step, "max_target_step")
        self._executed = np.tile(joints.neutral, (self._num_envs, 1))
        self._prev_action = np.zeros((self._num_envs, NUM_JOINTS))

    @property
    def joints(self) -> JointConfig:
        return self._joints

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def max_target_step(self) -> float:
        return self._max_step

    @property
    def max_target_rate(self) -> float:
        """Implied target rate limit in rad/s (max_target_step / dt)."""
        return self._max_step / self._dt

    @property
    def executed_target(self) -> np.ndarray:
        """Target currently held (copy), shape (num_envs, 18)."""
        return self._executed.copy()

    @property
    def previous_clipped_action(self) -> np.ndarray:
        """Clipped action that produced executed_target (copy), shape (num_envs, 18)."""
        return self._prev_action.copy()

    def step(self, action) -> TargetSnapshot:
        """Consume normalized actions (num_envs, 18); commit and return the executed target.

        All validation happens before any state change; a rejected call leaves state untouched.
        """
        raw = _finite_array(action, (self._num_envs, NUM_JOINTS), "action")
        j = self._joints
        clipped = np.clip(raw, -1.0, 1.0)
        requested = np.clip(j.neutral + j.action_scale * clipped, j.lower, j.upper)
        executed = np.clip(requested, self._executed - self._max_step,
                           self._executed + self._max_step)
        self._executed = executed
        self._prev_action = clipped
        return TargetSnapshot(raw, clipped.copy(), requested, executed.copy())

    def reset_rows(self, rows, measured_positions) -> np.ndarray:
        """Reset selected rows to measured joint positions.

        rows: unique 1-D integer indices. measured_positions: (len(rows), 18), finite.
        Out-of-limit measurements are clamped into [lower, upper] and the clamped
        value becomes the held target. Non-finite values reject without mutation.
        The previous clipped action is cleared to zero for the selected rows only.
        Returns an owned copy of the held targets for the selected rows.
        """
        idx = _row_indices(rows, self._num_envs)
        measured = _finite_array(measured_positions, (idx.size, NUM_JOINTS), "measured_positions")
        held = np.clip(measured, self._joints.lower, self._joints.upper)
        self._executed[idx] = held
        self._prev_action[idx] = 0.0
        return held


# ---------------------------------------------------------------------------
# Observation history
# ---------------------------------------------------------------------------
class ObservationScales:
    """Explicit positive normalization scales; each block is divided by its scale.

    No defaults: physical limits are not guessed here. linear_velocity is used
    only by the critic's privileged block.
    """

    def __init__(self, angular_velocity, projected_gravity, command,
                 joint_position_error, joint_rate, previous_action, target_error,
                 linear_velocity):
        j = (NUM_JOINTS,)
        self.angular_velocity = _positive_array(angular_velocity, (3,), "angular_velocity")
        self.projected_gravity = _positive_array(projected_gravity, (3,), "projected_gravity")
        self.command = _positive_array(command, (3,), "command")
        self.joint_position_error = _positive_array(joint_position_error, j, "joint_position_error")
        self.joint_rate = _positive_array(joint_rate, j, "joint_rate")
        self.previous_action = _positive_array(previous_action, j, "previous_action")
        self.target_error = _positive_array(target_error, j, "target_error")
        self.linear_velocity = _positive_array(linear_velocity, (3,), "linear_velocity")


@dataclass(frozen=True, eq=False)
class ActorFrameInputs:
    """Raw inputs for one actor frame over k rows. All arrays must be finite.

    native_angular_velocity  (k, 3)  body angular velocity in NATIVE body axes, rad/s
    native_projected_gravity (k, 3)  gravity direction in NATIVE body axes (caller rotates world to body)
    command                  (k, 3)  (vx, vy, wz) already in POLICY command axes; never remapped
    joint_position           (k, 18) measured joint positions, rad, configured joint order
    joint_rate               (k, 18) measured joint velocities, rad/s
    previous_clipped_action  (k, 18) clipped action behind the target held during the completed interval
    executed_target          (k, 18) target actually held during the completed interval
    Nothing here may be simulated contact, terrain, or other unobservable truth.
    """
    native_angular_velocity: np.ndarray
    native_projected_gravity: np.ndarray
    command: np.ndarray
    joint_position: np.ndarray
    joint_rate: np.ndarray
    previous_clipped_action: np.ndarray
    executed_target: np.ndarray


class ObservationHistory:
    """Five-frame actor history (num_envs, 5, 81), oldest to newest.

    Actor observation: flattened history (num_envs, 405).
    Critic observation: actor observation plus 3 privileged body linear velocity
    values (num_envs, 408). The privileged block is never stored in the history.
    Rows must be reset (first valid frame repeated five times) before push.
    """

    def __init__(self, neutral, scales: ObservationScales, num_envs: int):
        if not isinstance(scales, ObservationScales):
            raise TypeError("scales must be an ObservationScales")
        self._num_envs = _positive_int(num_envs, "num_envs")
        self._neutral = _finite_array(neutral, (NUM_JOINTS,), "neutral")
        self._neutral.flags.writeable = False
        self._scales = scales
        self._history = np.zeros((self._num_envs, HISTORY_LENGTH, FRAME_SIZE))
        self._initialized = np.zeros(self._num_envs, dtype=bool)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def history(self) -> np.ndarray:
        """Copy of the raw history, shape (num_envs, 5, 81), oldest first."""
        return self._history.copy()

    def _frame(self, inputs: ActorFrameInputs, k: int) -> np.ndarray:
        """Validate k rows of inputs and build normalized frames (k, 81). No state change."""
        if not isinstance(inputs, ActorFrameInputs):
            raise TypeError("inputs must be an ActorFrameInputs")
        v3, vj = (k, 3), (k, NUM_JOINTS)
        ang = _finite_array(inputs.native_angular_velocity, v3, "native_angular_velocity")
        grav = _finite_array(inputs.native_projected_gravity, v3, "native_projected_gravity")
        cmd = _finite_array(inputs.command, v3, "command")
        q = _finite_array(inputs.joint_position, vj, "joint_position")
        qd = _finite_array(inputs.joint_rate, vj, "joint_rate")
        act = _finite_array(inputs.previous_clipped_action, vj, "previous_clipped_action")
        if np.any(np.abs(act) > 1.0):
            raise ValueError("previous_clipped_action: must already be clipped to [-1, 1]")
        tgt = _finite_array(inputs.executed_target, vj, "executed_target")
        s = self._scales
        return np.concatenate([
            native_body_to_command_frame(ang) / s.angular_velocity,
            native_body_to_command_frame(grav) / s.projected_gravity,
            cmd / s.command,
            (q - self._neutral) / s.joint_position_error,
            qd / s.joint_rate,
            act / s.previous_action,
            (tgt - self._neutral) / s.target_error,
        ], axis=1)

    def reset_rows(self, rows, inputs: ActorFrameInputs) -> None:
        """Fill all five slots of the selected rows with this first valid frame.

        Other rows are untouched. inputs must have len(rows) rows.
        """
        idx = _row_indices(rows, self._num_envs)
        frame = self._frame(inputs, idx.size)
        self._history[idx] = frame[:, np.newaxis, :]
        self._initialized[idx] = True

    def push(self, inputs: ActorFrameInputs) -> None:
        """Append one frame for every row, dropping the oldest. Validates before mutating."""
        self._require_initialized()
        frame = self._frame(inputs, self._num_envs)
        self._history[:, :-1] = self._history[:, 1:].copy()
        self._history[:, -1] = frame

    def actor_observation(self) -> np.ndarray:
        """Owned (num_envs, 405) array: frames oldest to newest, each laid out per FRAME_LAYOUT."""
        self._require_initialized()
        return self._history.reshape(self._num_envs, ACTOR_OBS_SIZE).copy()

    def critic_observation(self, native_linear_velocity) -> np.ndarray:
        """PRIVILEGED: actor observation plus body linear velocity mapped to command axes.

        native_linear_velocity: (num_envs, 3) in native body axes at the caller's
        chosen measurement origin. Never available to the actor; never stored.
        """
        lin = _finite_array(native_linear_velocity, (self._num_envs, 3), "native_linear_velocity")
        privileged = native_body_to_command_frame(lin) / self._scales.linear_velocity
        return np.concatenate([self.actor_observation(), privileged], axis=1)

    def _require_initialized(self) -> None:
        if not self._initialized.all():
            missing = np.flatnonzero(~self._initialized).tolist()
            raise RuntimeError(f"rows {missing} have no valid frame; call reset_rows first")
