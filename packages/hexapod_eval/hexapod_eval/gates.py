"""Machine-readable single source of truth for hexapod acceptance gates.

Every number here is transcribed verbatim from the script that actually
enforces it.  Each structure names its source file and the constant inside that
file; `isaaclab/tests/test_eval_gates_contract.py` re-extracts those literals
with `ast` on every test run, so a threshold that drifts in either place fails a
test instead of silently diverging.

This module is deliberately dependency-free: stdlib only, no imports from
`hexapod_env`, `hexapod_train`, or the evaluation scripts themselves.  It is
data, not policy.  **Never change a value here to admit a candidate.**  A
threshold changes only by an explicit user decision recorded in `STATUS.md`,
and then it changes in the enforcing grader first and here second.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


Command = tuple[float, float, float]


def _frozen(mapping: Mapping[str, float]) -> Mapping[str, float]:
    """Return a read-only view so a gate table cannot be mutated in place."""

    return MappingProxyType(dict(mapping))


# ---------------------------------------------------------------------------
# Screen parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenParameters:
    """One deterministic playback configuration a checkpoint can be run under.

    Screen classes are not interchangeable.  Only a screen with
    ``formal_admission_eligible=True`` may replace the current best.
    """

    name: str
    seed: int
    commands: tuple[Command, ...]
    warmup_steps: int
    requested_steps: int
    post_warmup_samples: int
    policy_step_seconds: float
    measured_seconds: float
    processed_joint_target_slew_limit_rad_per_20ms: float
    formal_admission_eligible: bool
    source: str


# Source: isaaclab/grade_stage2c_stable_forward.py
#   EXPECTED_SEED = 60
#   COMMAND_CONTRACT = (("stand", (0.00, 0.00, 0.00)),
#                       ("forward_0p16", (0.16, 0.00, 0.00)),
#                       ("forward_0p20", (0.20, 0.00, 0.00)),
#                       ("forward_0p30", (0.30, 0.00, 0.00)))
#   EXPECTED_WARMUP_STEPS = 25
#   EXPECTED_REQUESTED_STEPS = 500
#   EXPECTED_SAMPLES = EXPECTED_REQUESTED_STEPS - EXPECTED_WARMUP_STEPS  -> 475
#   EXPECTED_POLICY_STEP_SECONDS = 0.02
#   EXPECTED_MEASURED_SECONDS = EXPECTED_SAMPLES * 0.02                 -> 9.5
# The measured window is 475 samples = 9.5 s of post-warmup data inside the
# 500-step / 10.0 s requested run; `docs/TRAINING.md` and the probe ledger both
# call this "10 s / 475 samples", meaning the requested run length.
# Source of the playback limiter: isaaclab/merge_stage2c_probe_shards.py
#   EXPECTED_ACTION_PROCESSING["processed_joint_target_slew_limit_rad_per_20ms"]
#     = 0.04
# and isaaclab/deploy/screen-stage2c-probe-sharded
#   --processed-joint-target-slew-limit-rad-per-20ms 0.040
CANONICAL_FORMAL_SCREEN = ScreenParameters(
    name="stage2c_canonical_formal",
    seed=60,
    commands=(
        (0.00, 0.00, 0.00),
        (0.16, 0.00, 0.00),
        (0.20, 0.00, 0.00),
        (0.30, 0.00, 0.00),
    ),
    warmup_steps=25,
    requested_steps=500,
    post_warmup_samples=475,
    policy_step_seconds=0.02,
    measured_seconds=9.5,
    processed_joint_target_slew_limit_rad_per_20ms=0.04,
    formal_admission_eligible=True,
    source="isaaclab/grade_stage2c_stable_forward.py",
)

# Source: isaaclab/analyze_stage2c_probe_sweep.py `_sample_contract`, the
# "diagnostic_short_duration" branch:
#   requested_steps = diagnostic_expected_samples + EXPECTED_WARMUP_STEPS
#   measured_seconds = diagnostic_expected_samples * EXPECTED_POLICY_STEP_SECONDS
#   "formal_admission_eligible": False
# The 275-sample / 6 s instantiation is named in that script's
# `--diagnostic-expected-samples` help text and in the probe ledger.
DIAGNOSTIC_SHORT_SCREEN = ScreenParameters(
    name="stage2c_diagnostic_short_duration",
    seed=60,
    commands=CANONICAL_FORMAL_SCREEN.commands,
    warmup_steps=25,
    requested_steps=300,
    post_warmup_samples=275,
    policy_step_seconds=0.02,
    measured_seconds=5.5,
    processed_joint_target_slew_limit_rad_per_20ms=0.04,
    formal_admission_eligible=False,
    source="isaaclab/analyze_stage2c_probe_sweep.py",
)


# ---------------------------------------------------------------------------
# RS05 actuator safety envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RS05SafetyEnvelope:
    """The actuator limits every stage inherits unchanged."""

    continuous_rated_torque_nm: float
    raw_demand_termination_nm: float
    raw_demand_termination_duration_s: float
    raw_demand_termination_grace_s: float
    source: str


# Source: isaaclab/grade_stage2_command_transitions.py EXPECTED_CONFIG_SNAPSHOT
#   "rated_torque_nm": 1.60
#   "terminate_on_computed_torque_demand_nm": 5.5
#   "terminate_on_computed_torque_demand_duration_s": 0.10
#   "torque_demand_termination_grace_s": 0.50
# That snapshot pins the resolved task config; the config origin is
# packages/hexapod_env/hexapod_env/env_cfg.py `rated_torque_nm = 1.6` and
# packages/hexapod_env/hexapod_env/phase2_cfg.py
# `terminate_on_computed_torque_demand_nm` / `_duration_s` /
# `torque_demand_termination_grace_s`.
RS05_SAFETY = RS05SafetyEnvelope(
    continuous_rated_torque_nm=1.60,
    raw_demand_termination_nm=5.5,
    raw_demand_termination_duration_s=0.10,
    raw_demand_termination_grace_s=0.50,
    source="isaaclab/grade_stage2_command_transitions.py",
)


# ---------------------------------------------------------------------------
# Stage 2C absolute admission gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage2CAdmissionGates:
    """The absolute gates a Stage2C candidate must clear to be admitted.

    These restate, in one place, what `grade_stage2c_stable_forward.py`
    enforces row by row.  They do not replace the grader; the grader is the
    authority and this mirror is bound to it by contract test.
    """

    maximum_yaw_rate_rmse_radps: float
    maximum_moving_normalized_deck_composite: float
    maximum_falls: int
    maximum_timeouts: int
    minimum_forward_command_fraction: float
    maximum_forward_command_fraction: float
    headline_forward_command_mps: float
    rs05: RS05SafetyEnvelope
    screen: ScreenParameters
    source: str

    @property
    def minimum_achieved_forward_mps_at_headline_command(self) -> float:
        """0.30 m/s command x 0.80 command fraction = 0.240 m/s achieved."""

        return self.headline_forward_command_mps * self.minimum_forward_command_fraction


# Source: isaaclab/grade_stage2c_stable_forward.py
#   TAIL_STABILITY_COMPONENTS ("yaw_rate_rmse_radps", ..., 0.080)
#     -> maximum_yaw_rate_rmse_radps
#   `stability_composite <= 1.0 + GATE_ABS_TOLERANCE` in `_grade_operational_row`
#     -> maximum_moving_normalized_deck_composite (1.000)
#   `if falls: reasons.append(f"falls={falls}, expected 0")`      -> 0 falls
#   `if timeouts: reasons.append(f"timeouts={timeouts}, expected 0")` -> 0 timeouts
#   THRESHOLDS["minimum_forward_command_fraction"] = 0.80
#   THRESHOLDS["maximum_forward_command_fraction"] = 1.30
#   COMMAND_CONTRACT ("forward_0p30", (0.30, 0.00, 0.00))
# NOTE ON MECHANISM, NOT VALUE: the prose ledgers state the deck gate as
# "moving normalized deck composite at most 1.000".  The grader reports exactly
# that flag (`moving_normalized_stability_composite_at_or_below_one`) but
# *enforces* admission through the four per-component absolute targets in
# STABILITY_COMPONENTS, each of which is the composite's own normalizer.  The
# value 1.000 is identical in both; only the enforcement path differs.
STAGE2C_ADMISSION = Stage2CAdmissionGates(
    maximum_yaw_rate_rmse_radps=0.080,
    maximum_moving_normalized_deck_composite=1.0,
    maximum_falls=0,
    maximum_timeouts=0,
    minimum_forward_command_fraction=0.80,
    maximum_forward_command_fraction=1.30,
    headline_forward_command_mps=0.30,
    rs05=RS05_SAFETY,
    screen=CANONICAL_FORMAL_SCREEN,
    source="isaaclab/grade_stage2c_stable_forward.py",
)


# ---------------------------------------------------------------------------
# Per-stage grader threshold tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageGates:
    """Numeric thresholds copied verbatim from one grader's own constants.

    `thresholds` mirrors that grader's module-level ``THRESHOLDS`` dict exactly:
    same keys, same values, nothing normalized, nothing added.  The component
    target maps mirror the grader's stability-component tuples, keyed by metric
    name.  A field is `None` when the grader has no such constant, or builds it
    dynamically rather than as a module-level literal.
    """

    stage: str
    source: str
    thresholds: Mapping[str, float]
    moving_rms_targets: Mapping[str, float] | None = None
    stand_rms_targets: Mapping[str, float] | None = None
    tail_targets: Mapping[str, float] | None = None
    command_contract: tuple[tuple[str, Command], ...] | None = None
    notes: str = ""


# Source: isaaclab/grade_stage1_recovery.py THRESHOLDS and COMMAND_CONTRACT.
STAGE1_RECOVERY = StageGates(
    stage="stage1-recovery",
    source="isaaclab/grade_stage1_recovery.py",
    thresholds=_frozen(
        {
            "minimum_forward_command_fraction": 0.80,
            "maximum_planar_velocity_rmse_mps": 0.12,
            "minimum_introduced_axis_command_fraction": 0.40,
            "maximum_lateral_rmse_mps": 0.08,
            "maximum_yaw_rate_rmse_radps": 0.12,
            "maximum_uncommanded_abs_yaw_rate_radps": 0.15,
            "maximum_tilt_degrees": 20.0,
            "max_per_joint_rms_applied_nm": 1.60,
            "computed_demand_over_rating_fraction": 0.20,
            "maximum_computed_over_rating_burst_s": 0.20,
            "peak_abs_computed_nm": 5.50,
            "preferred_maximum_tilt_degrees": 15.0,
            "preferred_base_height_std_m": 0.012,
            "preferred_vertical_velocity_rms_mps": 0.10,
            "preferred_roll_pitch_angular_velocity_rms_radps": 0.35,
            "preferred_tilt_rms_degrees": 5.0,
            "preferred_rms_applied_nm": 1.10,
            "preferred_computed_demand_over_rating_fraction": 0.15,
        }
    ),
    command_contract=(
        ("forward_0p20", (0.20, 0.00, 0.00)),
        ("forward_0p30", (0.30, 0.00, 0.00)),
        ("forward_yaw_left", (0.25, 0.00, 0.15)),
        ("forward_yaw_right", (0.25, 0.00, -0.15)),
        ("forward_lateral_left", (0.25, 0.06, 0.00)),
        ("forward_lateral_right", (0.25, -0.06, 0.00)),
        ("gentle_three_axis_left", (0.25, 0.05, 0.12)),
        ("gentle_three_axis_right", (0.25, -0.05, -0.12)),
    ),
    notes=(
        "The `preferred_*` keys are reported preferences, not rejection gates; "
        "the grader's own comment says so."
    ),
)


# Source: isaaclab/grade_stage2_axis_acquisition.py THRESHOLDS and
# COMMAND_CONTRACT.  This grader's STABILITY_COMPONENTS carry no numeric
# targets (they are 3-tuples), so there are no component target maps here.
STAGE2_AXIS_ACQUISITION = StageGates(
    stage="stage2-axis-acquisition",
    source="isaaclab/grade_stage2_axis_acquisition.py",
    thresholds=_frozen(
        {
            "minimum_forward_command_fraction": 0.80,
            "maximum_planar_velocity_rmse_mps": 0.12,
            "minimum_introduced_axis_command_fraction": 0.40,
            "maximum_lateral_rmse_mps": 0.10,
            "maximum_yaw_rate_rmse_radps": 0.18,
            "maximum_uncommanded_abs_yaw_rate_radps": 0.15,
            "maximum_tilt_degrees": 20.0,
            "max_per_joint_rms_applied_nm": 1.60,
            "computed_demand_over_rating_fraction": 0.20,
            "maximum_computed_over_rating_burst_s": 0.20,
            "peak_abs_computed_nm": 5.50,
            "maximum_stability_regression_fraction": 0.10,
            "preferred_base_height_std_m": 0.012,
            "preferred_vertical_velocity_rms_mps": 0.10,
            "preferred_roll_pitch_angular_velocity_rms_radps": 0.35,
            "preferred_tilt_rms_degrees": 5.0,
        }
    ),
    command_contract=(
        ("forward_0p20", (0.20, 0.00, 0.00)),
        ("forward_0p30", (0.30, 0.00, 0.00)),
        ("forward_lateral_left", (0.25, 0.10, 0.00)),
        ("forward_lateral_right", (0.25, -0.10, 0.00)),
        ("forward_yaw_left", (0.25, 0.00, 0.25)),
        ("forward_yaw_right", (0.25, 0.00, -0.25)),
    ),
    notes="Evaluation seed 57 (EXPECTED_SEED).",
)


# Source: isaaclab/grade_stage2b_lateral_acquisition.py THRESHOLDS and
# COMMAND_CONTRACT.
STAGE2B_LATERAL_ACQUISITION = StageGates(
    stage="stage2b-lateral-acquisition",
    source="isaaclab/grade_stage2b_lateral_acquisition.py",
    thresholds=_frozen(
        {
            "minimum_forward_command_fraction": 0.80,
            "minimum_pure_lateral_abs_velocity_mps": 0.040,
            "minimum_bridge_lateral_abs_velocity_mps": 0.032,
            "minimum_yaw_abs_rate_radps": 0.10,
            "minimum_lateral_pair_symmetry_ratio": 0.65,
            "maximum_pure_lateral_abs_forward_velocity_mps": 0.08,
            "maximum_uncommanded_abs_yaw_rate_radps": 0.15,
            "maximum_planar_velocity_rmse_mps": 0.12,
            "maximum_lateral_rmse_mps": 0.10,
            "maximum_yaw_rate_rmse_radps": 0.18,
            "maximum_tilt_degrees": 20.0,
            "max_per_joint_rms_applied_nm": 1.60,
            "computed_demand_over_rating_fraction": 0.20,
            "maximum_computed_over_rating_burst_s": 0.20,
            "peak_abs_computed_nm": 5.50,
            "maximum_stability_regression_fraction": 0.10,
            "preferred_base_height_std_m": 0.012,
            "preferred_vertical_velocity_rms_mps": 0.10,
            "preferred_roll_pitch_angular_velocity_rms_radps": 0.35,
            "preferred_tilt_rms_degrees": 5.0,
        }
    ),
    command_contract=(
        ("forward_0p20", (0.20, 0.00, 0.00)),
        ("forward_0p30", (0.30, 0.00, 0.00)),
        ("pure_y_positive", (0.00, 0.10, 0.00)),
        ("pure_y_negative", (0.00, -0.10, 0.00)),
        ("bridge_y_positive", (0.22, 0.08, 0.00)),
        ("bridge_y_negative", (0.22, -0.08, 0.00)),
        ("yaw_positive", (0.23, 0.00, 0.25)),
        ("yaw_negative", (0.23, 0.00, -0.25)),
    ),
    notes="Evaluation seed 58 (EXPECTED_SEED).",
)


# Source: isaaclab/grade_stage2c_stable_forward.py THRESHOLDS,
# STABILITY_COMPONENTS, STAND_STABILITY_COMPONENTS, TAIL_STABILITY_COMPONENTS,
# COMMAND_CONTRACT.  The moving RMS targets are simultaneously the composite's
# normalizers and hard absolute targets, per the grader's own comment.
STAGE2C_STABLE_FORWARD = StageGates(
    stage="stage2c-stable-forward",
    source="isaaclab/grade_stage2c_stable_forward.py",
    thresholds=_frozen(
        {
            "minimum_forward_command_fraction": 0.80,
            "maximum_forward_command_fraction": 1.30,
            "maximum_planar_velocity_rmse_mps": 0.10,
            "maximum_stand_planar_speed_mps": 0.03,
            "maximum_stand_abs_yaw_rate_radps": 0.08,
            "peak_abs_computed_nm": 4.40,
            "computed_demand_over_rating_fraction": 0.15,
            "maximum_computed_over_rating_burst_s": 0.14,
            "max_per_joint_rms_applied_nm": 1.40,
            "maximum_per_joint_computed_demand_over_rating_fraction": 0.25,
            "minimum_moving_stability_improvement_fraction": 0.15,
            "maximum_moving_component_regression_fraction": 0.05,
            "maximum_stand_component_regression_fraction": 0.10,
        }
    ),
    moving_rms_targets=_frozen(
        {
            "base_height_std_m": 0.0035,
            "vertical_velocity_rms_mps": 0.070,
            "roll_pitch_angular_velocity_rms_radps": 0.28,
            "tilt_rms_degrees": 0.90,
        }
    ),
    stand_rms_targets=_frozen(
        {
            "base_height_std_m": 0.001,
            "vertical_velocity_rms_mps": 0.020,
            "roll_pitch_angular_velocity_rms_radps": 0.050,
            "tilt_rms_degrees": 0.50,
        }
    ),
    tail_targets=_frozen(
        {
            "base_height_peak_to_peak_m": 0.0105,
            "vertical_velocity_abs_p95_mps": 0.11,
            "roll_pitch_angular_velocity_p95_radps": 0.45,
            "tilt_p95_degrees": 1.40,
            "maximum_tilt_degrees": 1.70,
            "yaw_rate_rmse_radps": 0.080,
        }
    ),
    command_contract=(
        ("stand", (0.00, 0.00, 0.00)),
        ("forward_0p16", (0.16, 0.00, 0.00)),
        ("forward_0p20", (0.20, 0.00, 0.00)),
        ("forward_0p30", (0.30, 0.00, 0.00)),
    ),
    notes=(
        "Evaluation seed 60 (EXPECTED_SEED). The moving RMS targets are the "
        "normalizers of the deck composite the ledgers cap at 1.000."
    ),
)


# Source: isaaclab/grade_stage2c_robustness.py THRESHOLDS,
# MOVING_RMS_COMPONENTS, STAND_RMS_COMPONENTS, TAIL_COMPONENTS,
# COMMAND_CONTRACT.
STAGE2C_ROBUSTNESS = StageGates(
    stage="stage2c-robustness",
    source="isaaclab/grade_stage2c_robustness.py",
    thresholds=_frozen(
        {
            "minimum_forward_command_fraction": 0.80,
            "maximum_forward_command_fraction": 1.30,
            "maximum_planar_velocity_rmse_mps": 0.10,
            "maximum_stand_planar_speed_mps": 0.03,
            "maximum_stand_abs_yaw_rate_radps": 0.08,
            "peak_abs_computed_nm": 4.40,
            "computed_demand_over_rating_fraction": 0.15,
            "maximum_computed_over_rating_burst_s": 0.14,
            "max_per_joint_rms_applied_nm": 1.40,
            "maximum_per_joint_computed_demand_over_rating_fraction": 0.35,
        }
    ),
    moving_rms_targets=_frozen(
        {
            "base_height_std_m": 0.0035,
            "vertical_velocity_rms_mps": 0.070,
            "roll_pitch_angular_velocity_rms_radps": 0.28,
            "tilt_rms_degrees": 0.90,
        }
    ),
    stand_rms_targets=_frozen(
        {
            "base_height_std_m": 0.001,
            "vertical_velocity_rms_mps": 0.020,
            "roll_pitch_angular_velocity_rms_radps": 0.050,
            "tilt_rms_degrees": 0.50,
        }
    ),
    tail_targets=_frozen(
        {
            "base_height_peak_to_peak_m": 0.0105,
            "vertical_velocity_abs_p95_mps": 0.11,
            "roll_pitch_angular_velocity_p95_radps": 0.45,
            "tilt_p95_degrees": 1.40,
            "maximum_tilt_degrees": 1.70,
            "yaw_rate_rmse_radps": 0.080,
        }
    ),
    command_contract=(
        ("stand", (0.00, 0.00, 0.00)),
        ("forward_0p16", (0.16, 0.00, 0.00)),
        ("forward_0p20", (0.20, 0.00, 0.00)),
        ("forward_0p30", (0.30, 0.00, 0.00)),
    ),
    notes=(
        "Randomized-copy screen at seed 61 (EXPECTED_SEED), 8 copies per "
        "command (EXPECTED_COPIES_PER_COMMAND), copy-target multiplier 1.25 "
        "(MAXIMUM_COPY_TARGET_MULTIPLIER). Its per-joint duty ceiling of 0.35 "
        "is deliberately wider than the nominal 0.25; the grader says so."
    ),
)


# Source: isaaclab/grade_stage2d_homotopy.py THRESHOLDS,
# RMS_STABILITY_COMPONENTS, TAIL_STABILITY_COMPONENTS.  Its per-command
# contract is built per stage inside the grader, so no literal command
# contract is mirrored here.
STAGE2D_HOMOTOPY = StageGates(
    stage="stage2d-homotopy",
    source="isaaclab/grade_stage2d_homotopy.py",
    thresholds=_frozen(
        {
            "minimum_forward_anchor_fraction": 0.80,
            "maximum_forward_anchor_fraction": 1.30,
            "minimum_oblique_forward_fraction": 0.60,
            "maximum_oblique_forward_fraction": 1.60,
            "minimum_yaw_command_fraction": 0.40,
            "maximum_yaw_command_fraction": 1.60,
            "maximum_signed_lateral_fraction": 1.75,
            "minimum_lateral_pair_symmetry_ratio": 0.65,
            "maximum_anchor_abs_lateral_velocity_mps": 0.06,
            "maximum_pure_y_abs_forward_velocity_mps": 0.06,
            "maximum_uncommanded_abs_yaw_rate_radps": 0.10,
            "maximum_anchor_planar_velocity_rmse_mps": 0.12,
            "maximum_oblique_planar_velocity_rmse_mps": 0.14,
            "maximum_yaw_rate_rmse_radps": 0.18,
            "rated_continuous_nm": 1.60,
            "peak_abs_computed_nm": 4.40,
            "computed_demand_over_rating_fraction": 0.15,
            "maximum_computed_over_rating_burst_s": 0.14,
            "max_per_joint_rms_applied_nm": 1.40,
            "maximum_per_joint_rms_applied_nm": 1.60,
            "maximum_per_joint_computed_demand_over_rating_fraction": 0.25,
            "maximum_per_joint_computed_over_rating_burst_s": 0.20,
            "maximum_per_joint_peak_abs_computed_nm": 5.50,
            "maximum_stability_regression_fraction": 0.10,
            "minimum_anchor_tracking_retention_fraction": 0.90,
        }
    ),
    moving_rms_targets=_frozen(
        {
            "base_height_std_m": 0.0035,
            "vertical_velocity_rms_mps": 0.070,
            "roll_pitch_angular_velocity_rms_radps": 0.28,
            "tilt_rms_degrees": 0.90,
        }
    ),
    tail_targets=_frozen(
        {
            "base_height_peak_to_peak_m": 0.0105,
            "vertical_velocity_abs_p95_mps": 0.11,
            "roll_pitch_angular_velocity_p95_radps": 0.45,
            "tilt_p95_degrees": 1.40,
            "maximum_tilt_degrees": 1.70,
            "yaw_rate_rmse_radps": 0.080,
        }
    ),
    notes=(
        "Stages C0-C5; each stage's evaluation seed is 61 + index and its "
        "command ranges live in STAGE_SPECS. Configured, not admitted."
    ),
)


# Source: isaaclab/grade_stage2e_static.py THRESHOLDS and
# STAND_RMS_STABILITY_COMPONENTS.  Its moving RMS and tail limits are inherited
# by import from grade_stage2d_homotopy (`import ... as common`), so they are
# not re-mirrored here; see STAGE2D_HOMOTOPY.
STAGE2E_STATIC = StageGates(
    stage="stage2e-static",
    source="isaaclab/grade_stage2e_static.py",
    thresholds=_frozen(
        {
            "minimum_forward_fraction": 0.80,
            "minimum_combined_forward_fraction": 0.60,
            "minimum_lateral_fraction": 0.40,
            "minimum_combined_lateral_fraction": 0.35,
            "minimum_yaw_fraction": 0.40,
            "minimum_combined_yaw_fraction": 0.35,
            "maximum_signed_axis_fraction": 1.75,
            "minimum_anchor_retention_fraction": 0.90,
            "maximum_stand_planar_speed_mps": 0.03,
            "maximum_stand_planar_rmse_mps": 0.04,
            "maximum_stand_abs_yaw_rate_radps": 0.08,
            "maximum_uncommanded_axis_speed": 0.06,
            "maximum_anchor_planar_rmse_mps": 0.14,
            "maximum_combined_planar_rmse_mps": 0.18,
            "maximum_yaw_rmse_radps": 0.20,
            "maximum_stability_regression_fraction": 0.10,
            "maximum_mean_base_height_error_m": 0.010,
        }
    ),
    stand_rms_targets=_frozen(
        {
            "base_height_std_m": 0.001,
            "vertical_velocity_rms_mps": 0.020,
            "roll_pitch_angular_velocity_rms_radps": 0.050,
            "tilt_rms_degrees": 0.50,
        }
    ),
    notes=(
        "Stages E0-E2 with evaluation seeds 72/73/74 in STAGE_SPECS. "
        "Configured, not admitted."
    ),
)


# Source: isaaclab/grade_stage2_command_transitions.py THRESHOLDS,
# RMS_MOVING_COMPONENTS, RMS_STAND_COMPONENTS, TAIL_COMPONENTS.  The
# "maximum_settling_time_s" entry is the imported constant
# MAXIMUM_SETTLING_TIME_SECONDS = 1.0 from
# isaaclab/stage2_command_transition_contract.py.
STAGE2_COMMAND_TRANSITIONS = StageGates(
    stage="stage2-command-transitions",
    source="isaaclab/grade_stage2_command_transitions.py",
    thresholds=_frozen(
        {
            "minimum_forward_fraction": 0.60,
            "maximum_forward_fraction": 1.60,
            "minimum_forward_anchor_fraction": 0.80,
            "maximum_forward_anchor_fraction": 1.30,
            "minimum_lateral_fraction": 0.45,
            "maximum_lateral_fraction": 1.75,
            "minimum_lateral_pair_symmetry_ratio": 0.65,
            "minimum_yaw_pair_symmetry_ratio": 0.65,
            "minimum_yaw_fraction": 0.40,
            "maximum_yaw_fraction": 1.60,
            "maximum_inactive_translation_mps": 0.06,
            "maximum_inactive_yaw_radps": 0.10,
            "maximum_planar_velocity_rmse_mps": 0.14,
            "maximum_forward_anchor_planar_velocity_rmse_mps": 0.12,
            "maximum_stand_planar_speed_mps": 0.03,
            "maximum_stand_planar_velocity_rmse_mps": 0.04,
            "maximum_stand_abs_yaw_rate_radps": 0.08,
            "maximum_mean_base_height_error_m": 0.010,
            "rated_continuous_nm": 1.60,
            "steady_peak_abs_computed_nm": 4.40,
            "steady_computed_demand_over_rating_fraction": 0.15,
            "steady_maximum_computed_over_rating_burst_s": 0.14,
            "steady_max_per_joint_rms_applied_nm": 1.40,
            "transient_peak_abs_computed_nm": 5.50,
            "transient_computed_demand_over_rating_fraction": 0.20,
            "transient_maximum_computed_over_rating_burst_s": 0.20,
            "transient_max_per_joint_rms_applied_nm": 1.60,
            "rollout_peak_abs_computed_nm": 5.50,
            "rollout_computed_demand_over_rating_fraction": 0.15,
            "rollout_maximum_computed_over_rating_burst_s": 0.20,
            "rollout_max_per_joint_rms_applied_nm": 1.40,
            "per_joint_rms_applied_nm": 1.60,
            "steady_per_joint_computed_demand_over_rating_fraction": 0.25,
            "transient_per_joint_computed_demand_over_rating_fraction": 0.35,
            "rollout_per_joint_computed_demand_over_rating_fraction": 0.25,
            "per_joint_maximum_computed_over_rating_burst_s": 0.20,
            "per_joint_peak_abs_computed_nm": 5.50,
            "transient_planar_error_overshoot_margin_mps": 0.15,
            "transient_yaw_error_overshoot_margin_radps": 0.15,
            "transient_base_height_peak_to_peak_m": 0.021,
            "transient_maximum_abs_vertical_velocity_mps": 0.22,
            "transient_maximum_roll_pitch_angular_velocity_radps": 0.90,
            "transient_maximum_tilt_degrees": 3.40,
            "maximum_settling_time_s": 1.0,
        }
    ),
    moving_rms_targets=_frozen(
        {
            "base_height_std_m": 0.0035,
            "vertical_velocity_rms_mps": 0.070,
            "roll_pitch_angular_velocity_rms_radps": 0.28,
            "tilt_rms_degrees": 0.90,
        }
    ),
    stand_rms_targets=_frozen(
        {
            "base_height_std_m": 0.001,
            "vertical_velocity_rms_mps": 0.020,
            "roll_pitch_angular_velocity_rms_radps": 0.050,
            "tilt_rms_degrees": 0.50,
        }
    ),
    tail_targets=_frozen(
        {
            "base_height_peak_to_peak_m": 0.0105,
            "vertical_velocity_abs_p95_mps": 0.11,
            "roll_pitch_angular_velocity_p95_radps": 0.45,
            "tilt_p95_degrees": 1.40,
            "maximum_tilt_degrees": 1.70,
            "yaw_rate_rmse_radps": 0.080,
        }
    ),
    notes=(
        "Terminal joystick-transition screen. Steady, transient, and rollout "
        "windows carry separate RS05 limits."
    ),
)


# Source: isaaclab/summarize_stance_validation.py module constants
# APPLIED_TORQUE_LIMIT_NM / COMPUTED_TORQUE_PEAK_LIMIT_NM /
# SATURATION_FRACTION_LIMIT / MINIMUM_BASE_HEIGHT_M.  This is a structural
# stance gate on the asset and task, not a policy admission screen.
STANCE_VALIDATION = StageGates(
    stage="stance-validation",
    source="isaaclab/summarize_stance_validation.py",
    thresholds=_frozen(
        {
            "applied_torque_limit_nm": 1.61,
            "computed_torque_peak_limit_nm": 5.50,
            "saturation_fraction_limit": 0.005,
            "minimum_base_height_m": 0.055,
        }
    ),
    notes=(
        "Named constants, not a THRESHOLDS dict: APPLIED_TORQUE_LIMIT_NM, "
        "COMPUTED_TORQUE_PEAK_LIMIT_NM, SATURATION_FRACTION_LIMIT, "
        "MINIMUM_BASE_HEIGHT_M."
    ),
)


STAGES: Mapping[str, StageGates] = MappingProxyType(
    {
        gate.stage: gate
        for gate in (
            STAGE1_RECOVERY,
            STAGE2_AXIS_ACQUISITION,
            STAGE2B_LATERAL_ACQUISITION,
            STAGE2C_STABLE_FORWARD,
            STAGE2C_ROBUSTNESS,
            STAGE2D_HOMOTOPY,
            STAGE2E_STATIC,
            STAGE2_COMMAND_TRANSITIONS,
            STANCE_VALIDATION,
        )
    }
)

SCREENS: Mapping[str, ScreenParameters] = MappingProxyType(
    {
        screen.name: screen
        for screen in (CANONICAL_FORMAL_SCREEN, DIAGNOSTIC_SHORT_SCREEN)
    }
)


__all__ = [
    "CANONICAL_FORMAL_SCREEN",
    "Command",
    "DIAGNOSTIC_SHORT_SCREEN",
    "RS05_SAFETY",
    "RS05SafetyEnvelope",
    "SCREENS",
    "STAGE1_RECOVERY",
    "STAGE2B_LATERAL_ACQUISITION",
    "STAGE2C_ADMISSION",
    "STAGE2C_ROBUSTNESS",
    "STAGE2C_STABLE_FORWARD",
    "STAGE2D_HOMOTOPY",
    "STAGE2E_STATIC",
    "STAGE2_AXIS_ACQUISITION",
    "STAGE2_COMMAND_TRANSITIONS",
    "STAGES",
    "STANCE_VALIDATION",
    "ScreenParameters",
    "Stage2CAdmissionGates",
    "StageGates",
]
