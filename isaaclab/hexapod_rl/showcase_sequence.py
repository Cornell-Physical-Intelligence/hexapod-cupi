"""Pure helpers for the deterministic Phase 2 joystick showcase.

This module intentionally has no Isaac Lab or PyTorch dependency.  It owns the
timed navigation-command script, acceptance gates, and optional RGB-frame HUD so
those pieces can be checked with ordinary CPU unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class CommandSegment:
    """One constant navigation-frame command in the showcase."""

    key: str
    label: str
    command: tuple[float, float, float]
    duration_s: float


@dataclass(frozen=True)
class ScheduledCommandSegment:
    """A command segment quantized to whole policy steps."""

    segment: CommandSegment
    index: int
    start_step: int
    stop_step: int
    settle_steps: int
    policy_dt_s: float

    @property
    def steps(self) -> int:
        return self.stop_step - self.start_step

    @property
    def measured_steps(self) -> int:
        return self.steps - self.settle_steps

    @property
    def start_s(self) -> float:
        return self.start_step * self.policy_dt_s

    @property
    def stop_s(self) -> float:
        return self.stop_step * self.policy_dt_s


def default_showcase_segments(
    *,
    stand_seconds: float = 2.0,
    motion_seconds: float = 3.0,
) -> tuple[CommandSegment, ...]:
    """Return the fixed, anatomical-navigation Phase 2 demonstration script."""

    _require_positive_finite("stand_seconds", stand_seconds)
    _require_positive_finite("motion_seconds", motion_seconds)
    return (
        CommandSegment("stand", "STAND", (0.0, 0.0, 0.0), stand_seconds),
        CommandSegment("forward", "FORWARD", (0.30, 0.0, 0.0), motion_seconds),
        CommandSegment(
            "strafe_left", "STRAFE LEFT", (0.0, 0.15, 0.0), motion_seconds
        ),
        CommandSegment(
            "strafe_right", "STRAFE RIGHT", (0.0, -0.15, 0.0), motion_seconds
        ),
        CommandSegment("turn_left", "TURN LEFT", (0.0, 0.0, 0.35), motion_seconds),
        CommandSegment(
            "turn_right", "TURN RIGHT", (0.0, 0.0, -0.35), motion_seconds
        ),
        CommandSegment(
            "diagonal_forward_left",
            "DIAGONAL FORWARD LEFT",
            (0.25, 0.12, 0.0),
            motion_seconds,
        ),
        CommandSegment("backward", "BACKWARD", (-0.15, 0.0, 0.0), motion_seconds),
    )


def schedule_segments(
    segments: Iterable[CommandSegment],
    *,
    policy_dt_s: float,
    settle_seconds: float,
) -> tuple[ScheduledCommandSegment, ...]:
    """Quantize a sequence without gaps and reserve steady-state metric samples."""

    _require_positive_finite("policy_dt_s", policy_dt_s)
    if not math.isfinite(settle_seconds) or settle_seconds < 0.0:
        raise ValueError("settle_seconds must be finite and non-negative")

    scheduled: list[ScheduledCommandSegment] = []
    start_step = 0
    requested_settle_steps = int(math.ceil(settle_seconds / policy_dt_s))
    for index, segment in enumerate(segments):
        _require_positive_finite(f"segments[{index}].duration_s", segment.duration_s)
        if len(segment.command) != 3 or not all(
            math.isfinite(value) for value in segment.command
        ):
            raise ValueError(
                f"segments[{index}].command must contain three finite values"
            )
        steps = int(math.ceil(segment.duration_s / policy_dt_s))
        if requested_settle_steps >= steps:
            raise ValueError(
                f"settle_seconds leaves no measured samples in segment {segment.key!r}: "
                f"settle_steps={requested_settle_steps}, segment_steps={steps}"
            )
        stop_step = start_step + steps
        scheduled.append(
            ScheduledCommandSegment(
                segment=segment,
                index=index,
                start_step=start_step,
                stop_step=stop_step,
                settle_steps=requested_settle_steps,
                policy_dt_s=policy_dt_s,
            )
        )
        start_step = stop_step

    if not scheduled:
        raise ValueError("at least one showcase segment is required")
    return tuple(scheduled)


def acceptance_thresholds() -> dict[str, float]:
    """Return the nominal Phase 2 screen gates embedded in showcase reports."""

    return {
        "stand_planar_speed_mps": 0.06,
        "stand_abs_yaw_rate_radps": 0.10,
        "axis_translation_planar_rmse_mps": 0.12,
        "diagonal_translation_planar_rmse_mps": 0.15,
        "translation_abs_yaw_rate_radps": 0.15,
        "yaw_only_yaw_rmse_radps": 0.18,
        "yaw_only_planar_speed_mps": 0.10,
        "maximum_tilt_degrees": 20.0,
        "max_per_joint_rms_applied_nm": 1.60,
        "computed_demand_over_rating_fraction": 0.20,
        "maximum_computed_over_rating_burst_s": 0.20,
        "peak_abs_computed_nm": 5.50,
    }


def evaluate_segment_acceptance(
    row: dict[str, Any],
    command: tuple[float, float, float],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply deterministic tracking, stability, and RS05 load gates to one row."""

    limits = acceptance_thresholds() if thresholds is None else thresholds
    vx, vy, yaw = command
    mean_linear = row["mean_command_frame_linear_velocity_mps"]
    mean_angular = row["mean_command_frame_angular_velocity_radps"]
    planar_speed = math.hypot(float(mean_linear[0]), float(mean_linear[1]))
    abs_yaw_rate = abs(float(mean_angular[2]))
    active_planar_axes = sum(abs(value) > 1.0e-9 for value in (vx, vy))
    yaw_active = abs(yaw) > 1.0e-9
    reasons: list[str] = []

    def maximum(name: str, actual: float, limit: float, unit: str = "") -> None:
        if not math.isfinite(actual):
            reasons.append(f"{name} is non-finite ({actual!r})")
        elif actual > limit:
            suffix = unit and f" {unit}"
            reasons.append(f"{name} {actual:.4f}{suffix} exceeds {limit:.4f}{suffix}")

    if int(row["falls"]) != 0:
        reasons.append(f"falls={int(row['falls'])}, expected 0")

    if active_planar_axes == 0 and not yaw_active:
        maximum(
            "stand planar speed",
            planar_speed,
            limits["stand_planar_speed_mps"],
            "m/s",
        )
        maximum(
            "stand absolute yaw rate",
            abs_yaw_rate,
            limits["stand_abs_yaw_rate_radps"],
            "rad/s",
        )
    elif yaw_active and active_planar_axes == 0:
        achieved_yaw = float(mean_angular[2])
        if not math.isfinite(achieved_yaw):
            reasons.append(f"achieved yaw rate is non-finite ({achieved_yaw!r})")
        elif achieved_yaw * yaw <= 0.0:
            reasons.append(
                f"yaw sign is wrong: command={yaw:+.3f}, achieved={achieved_yaw:+.3f}"
            )
        maximum(
            "yaw-rate RMSE",
            float(row["yaw_rate_rmse_radps"]),
            limits["yaw_only_yaw_rmse_radps"],
            "rad/s",
        )
        maximum(
            "yaw-only planar speed",
            planar_speed,
            limits["yaw_only_planar_speed_mps"],
            "m/s",
        )
    else:
        for axis_name, commanded, achieved in zip(
            ("vx", "vy"), (vx, vy), mean_linear[:2]
        ):
            achieved = float(achieved)
            if abs(commanded) <= 1.0e-9:
                continue
            if not math.isfinite(achieved):
                reasons.append(f"achieved {axis_name} is non-finite ({achieved!r})")
            elif achieved * commanded <= 0.0:
                reasons.append(
                    f"{axis_name} sign is wrong: command={commanded:+.3f}, "
                    f"achieved={achieved:+.3f}"
                )
        rmse_limit = (
            limits["axis_translation_planar_rmse_mps"]
            if active_planar_axes == 1
            else limits["diagonal_translation_planar_rmse_mps"]
        )
        maximum(
            "planar velocity RMSE",
            float(row["planar_velocity_rmse_mps"]),
            rmse_limit,
            "m/s",
        )
        maximum(
            "translation absolute yaw rate",
            abs_yaw_rate,
            limits["translation_abs_yaw_rate_radps"],
            "rad/s",
        )

    maximum(
        "maximum tilt",
        float(row["maximum_tilt_degrees"]),
        limits["maximum_tilt_degrees"],
        "deg",
    )
    torque = row["torque"]
    maximum(
        "max per-joint RMS applied torque",
        float(torque["max_per_joint_rms_applied_nm"]),
        limits["max_per_joint_rms_applied_nm"],
        "Nm",
    )
    maximum(
        "computed demand over rating fraction",
        float(torque["computed_demand_over_rating_fraction"]),
        limits["computed_demand_over_rating_fraction"],
    )
    maximum(
        "computed over-rating burst",
        float(torque["maximum_computed_over_rating_burst_s"]),
        limits["maximum_computed_over_rating_burst_s"],
        "s",
    )
    maximum(
        "peak absolute computed torque",
        float(torque["peak_abs_computed_nm"]),
        limits["peak_abs_computed_nm"],
        "Nm",
    )
    return {"accepted": not reasons, "rejection_reasons": reasons}


def annotate_showcase_frame(
    frame: np.ndarray,
    *,
    label: str,
    command: tuple[float, float, float],
    segment_index: int,
    segment_count: int,
    progress: float,
) -> np.ndarray:
    """Return a copy of an RGB frame with a compact command HUD and direction cue."""

    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"expected an HxWx3/4 frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"expected a uint8 frame, got {frame.dtype}")
    if segment_count < 1 or not 0 <= segment_index < segment_count:
        raise ValueError("segment index/count are inconsistent")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - Isaac images include OpenCV.
        raise RuntimeError(
            "OpenCV is required for the showcase HUD; rerun with --no-overlay"
        ) from exc

    output = np.ascontiguousarray(frame).copy()
    height, width = output.shape[:2]
    scale = max(0.55, min(width / 1920.0, height / 1080.0))
    margin = max(12, int(round(24 * scale)))
    panel_height = max(108, int(round(150 * scale)))
    panel_width = min(width - 2 * margin, max(430, int(round(700 * scale))))
    x0, y0 = margin, margin
    x1, y1 = x0 + panel_width, min(height - margin, y0 + panel_height)

    # Alpha-blend a black panel without mutating the simulator's reusable render buffer.
    roi = output[y0:y1, x0:x1, :3]
    roi[:] = np.rint(roi.astype(np.float32) * 0.38).astype(np.uint8)
    accent = (255, 190, 40)
    white = (255, 255, 255)
    muted = (190, 200, 210)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.72 * scale
    thickness = max(1, int(round(2 * scale)))

    cv2.putText(
        output,
        f"PHASE 2  {segment_index + 1}/{segment_count}",
        (x0 + margin, y0 + int(30 * scale)),
        font,
        0.52 * scale,
        muted,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        label,
        (x0 + margin, y0 + int(70 * scale)),
        font,
        font_scale,
        white,
        thickness,
        cv2.LINE_AA,
    )
    vx, vy, yaw = command
    cv2.putText(
        output,
        f"NAV  vx {vx:+.2f} m/s   vy {vy:+.2f} m/s   yaw {yaw:+.2f} rad/s",
        (x0 + margin, y0 + int(105 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46 * scale,
        muted,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )

    bar_left = x0 + margin
    bar_right = x1 - margin
    bar_top = y1 - max(13, int(round(18 * scale)))
    bar_bottom = y1 - max(7, int(round(10 * scale)))
    cv2.rectangle(output, (bar_left, bar_top), (bar_right, bar_bottom), muted, 1)
    fraction = min(1.0, max(0.0, float(progress)))
    fill_right = bar_left + int(round((bar_right - bar_left) * fraction))
    if fill_right > bar_left:
        cv2.rectangle(
            output, (bar_left, bar_top), (fill_right, bar_bottom), accent, -1
        )

    # A top-down navigation cue: screen-up is anatomical forward, screen-left is +vy.
    cue_radius = max(38, int(round(58 * scale)))
    cue_center = (width - margin - cue_radius, margin + cue_radius)
    cv2.circle(output, cue_center, cue_radius, white, max(1, thickness - 1), cv2.LINE_AA)
    cv2.putText(
        output,
        "F",
        (cue_center[0] - int(6 * scale), cue_center[1] - cue_radius - int(7 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45 * scale,
        white,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    planar_norm = math.hypot(vx, vy)
    if planar_norm > 1.0e-9:
        arrow_length = int(round(cue_radius * 0.78))
        end = (
            cue_center[0] + int(round((-vy / planar_norm) * arrow_length)),
            cue_center[1] + int(round((-vx / planar_norm) * arrow_length)),
        )
        cv2.arrowedLine(
            output,
            cue_center,
            end,
            accent,
            max(2, thickness + 1),
            cv2.LINE_AA,
            tipLength=0.28,
        )
    elif abs(yaw) > 1.0e-9:
        if yaw > 0.0:
            start_angle, end_angle = 35, 315
            tip_angle = math.radians(315)
        else:
            start_angle, end_angle = 225, -55
            tip_angle = math.radians(-55)
        arc_radius = int(round(cue_radius * 0.58))
        cv2.ellipse(
            output,
            cue_center,
            (arc_radius, arc_radius),
            0.0,
            start_angle,
            end_angle,
            accent,
            max(2, thickness + 1),
            cv2.LINE_AA,
        )
        tip = (
            cue_center[0] + int(round(math.cos(tip_angle) * arc_radius)),
            cue_center[1] + int(round(math.sin(tip_angle) * arc_radius)),
        )
        tangent_sign = 1.0 if yaw > 0.0 else -1.0
        tail = (
            tip[0] + int(round(math.sin(tip_angle) * 18 * scale * tangent_sign)),
            tip[1] - int(round(math.cos(tip_angle) * 18 * scale * tangent_sign)),
        )
        cv2.arrowedLine(
            output,
            tail,
            tip,
            accent,
            max(2, thickness + 1),
            cv2.LINE_AA,
            tipLength=0.65,
        )
    else:
        cv2.circle(output, cue_center, max(4, thickness + 2), accent, -1, cv2.LINE_AA)
    if output.shape[2] == 4:
        # OpenCV's three-component drawing colors otherwise author zero alpha.
        output[:, :, 3] = frame[:, :, 3]
    return output


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


__all__ = [
    "CommandSegment",
    "ScheduledCommandSegment",
    "acceptance_thresholds",
    "annotate_showcase_frame",
    "default_showcase_segments",
    "evaluate_segment_acceptance",
    "schedule_segments",
]
