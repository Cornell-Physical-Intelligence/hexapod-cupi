"""Torch-only provisional RS05 envelope/budget engine, independently CPU-testable.

Headroom is a dimensionless exposure budget, NOT winding temperature. It is
updated once per physics step, persists across changing operating points, and
never permits a full step to overdraw the remaining budget. All joints use
output-side coordinates. No 7.75 gearbox multiplier belongs here.
"""
from __future__ import annotations

import bisect
import math

import torch

from hexapod_core.rs05_v2 import CONFIG, validate_operating_assumptions

TELEMETRY_FIELDS = (
    "burst_headroom", "continuous_limit_nm", "torque_speed_limit_nm",
    "raw_demand_nm", "clipping_nm", "continuous_overload_nm", "applied_overload_nm",
    "applied_overload_exposure_s", "applied_peak_exposure_s",
    "estimated_phase_current_arms", "estimated_phase_current_apk", "invalid_input",
    "budget_consumed", "budget_recovered", "instantaneous_limit_nm", "envelope_violation_nm",
)


def _interpolate_scalar(x, points):
    i = max(0, min(len(points) - 2, bisect.bisect_right([p[0] for p in points], x) - 1))
    x0, y0 = points[i]
    x1, y1 = points[i + 1]
    return y0 + min(1.0, max(0.0, (x - x0) / (x1 - x0))) * (y1 - y0)


def _budget_rate_knots():
    """Common normalized-excess axis makes the blended curve exactly invertible."""
    curves = []
    peak = CONFIG["vendor"]["peak_output_torque_nm"]
    for key in ("stall_overload_nm_s", "rotating_overload_nm_s"):
        table = CONFIG["vendor"][key]
        continuous = table[0][0]
        curves.append([((torque - continuous) / (peak - continuous), 0.0 if duration is None else 1.0 / duration)
                       for torque, duration in table])
    knots = sorted({p[0] for curve in curves for p in curve})
    return knots, [[_interpolate_scalar(x, curve) for x in knots] for curve in curves]


class RS05V2BudgetModel:
    """Vectorized motor state of shape (environments, active motors).

    Invalid inputs deliver zero and latch ``invalid_input`` until reset; the
    environment MUST terminate those environments. Avoids a CPU/GPU sync per
    physics step. Shape/dtype misuse raises immediately.
    """

    def __init__(self, shape, *, device="cpu", dtype=torch.float32, physics_dt_s=0.005,
                 assumed_bus_voltage_v=48.0, reset_burst_headroom=None):
        validate_operating_assumptions(physics_dt_s, assumed_bus_voltage_v)
        if len(shape) != 2 or any(n <= 0 for n in shape) or dtype not in (torch.float32, torch.float64):
            raise ValueError("RS05 model needs positive2D shape and float32/float64 dtype")
        self.shape = tuple(shape)
        self.dt = physics_dt_s
        self.voltage_scale = min(assumed_bus_voltage_v / 48.0, 1.0)
        self.reset_headroom = CONFIG["provisional"]["reset_burst_headroom"] if reset_burst_headroom is None else reset_burst_headroom
        if not math.isfinite(self.reset_headroom) or not 0 <= self.reset_headroom <= 1:
            raise ValueError("Reset burst headroom must be finite in[0,1]")
        self.burst_headroom = torch.full(shape, self.reset_headroom, dtype=dtype, device=device)
        self.invalid_input = torch.zeros(shape, dtype=torch.bool, device=device)
        for name in TELEMETRY_FIELDS:
            if name not in ("burst_headroom", "invalid_input"):
                setattr(self, name, torch.zeros_like(self.burst_headroom))
        knots, rates = _budget_rate_knots()
        self._knots = torch.tensor(knots, device=device, dtype=dtype)
        self._stall_rates, self._rotating_rates = (torch.tensor(row, device=device, dtype=dtype) for row in rates)
        table = CONFIG["vendor"]["torque_speed_rpm_nm"]
        self._speed_x = torch.tensor([row[0] for row in table], device=device, dtype=dtype)
        self._speed_y = torch.tensor([row[1] for row in table], device=device, dtype=dtype)

    def reset(self, env_ids=None):
        """Reset only selected independent episodes; unrelated budgets persist."""
        ids = slice(None) if env_ids is None else env_ids
        self.burst_headroom[ids] = self.reset_headroom
        self.invalid_input[ids] = False
        for name in TELEMETRY_FIELDS:
            if name not in ("burst_headroom", "invalid_input"):
                getattr(self, name)[ids] = 0

    def _speed_limit(self, rpm):
        x = rpm / self.voltage_scale
        i = torch.searchsorted(self._speed_x, x.contiguous(), right=True).sub(1).clamp(0, len(self._speed_x) - 2)
        fraction = ((x - self._speed_x[i]) / (self._speed_x[i + 1] - self._speed_x[i])).clamp(0, 1)
        return (self._speed_y[i] + fraction * (self._speed_y[i + 1] - self._speed_y[i])) * self.voltage_scale

    def _continuous(self, rpm):
        v = CONFIG["vendor"]
        blend = (rpm / v["rated_rotating_rpm"]).clamp(0, 1)
        low_speed = v["stall_continuous_nm"] + blend * (v["rotating_continuous_nm"] - v["stall_continuous_nm"])
        power_cap = v["rotating_continuous_nm"] * v["rated_rotating_rpm"] / rpm.clamp_min(v["rated_rotating_rpm"])
        return torch.minimum(low_speed, power_cap) * self.voltage_scale, blend

    def step(self, raw_demand_nm, joint_velocity_rad_s):
        if (tuple(raw_demand_nm.shape) != self.shape or tuple(joint_velocity_rad_s.shape) != self.shape
                or raw_demand_nm.device != self.burst_headroom.device or joint_velocity_rad_s.device != self.burst_headroom.device
                or raw_demand_nm.dtype != self.burst_headroom.dtype or joint_velocity_rad_s.dtype != self.burst_headroom.dtype):
            raise ValueError("RS05 demand and velocity must match model shape, dtype and device")
        self.raw_demand_nm.copy_(raw_demand_nm)
        finite = torch.isfinite(raw_demand_nm) & torch.isfinite(joint_velocity_rad_s) & torch.isfinite(self.burst_headroom)
        in_bounds = (self.burst_headroom >= 0) & (self.burst_headroom <= 1)
        self.invalid_input.logical_or_(~finite | ~in_bounds)
        valid = ~self.invalid_input
        demand = torch.where(valid, raw_demand_nm, 0.0)
        # Speeds above the published no-load endpoint already give zero
        # authority. Bound only this internal conversion to prevent finite
        # extreme inputs overflowing into NaN budget arithmetic.
        rpm = torch.where(valid, joint_velocity_rad_s, 0.0).abs().clamp_max(10000.0) * (60 / (2 * math.pi))
        remaining = torch.where(valid, self.burst_headroom, 0.0)
        speed_limit = self._speed_limit(rpm)
        continuous, blend = self._continuous(rpm)
        # The thermal baseline stays separate from the instantaneous speed cap.
        # This preserves the rated-point budget curve even when speed is limiting.
        peak = CONFIG["vendor"]["peak_output_torque_nm"] * self.voltage_scale
        rates = self._stall_rates + blend.unsqueeze(-1) * (self._rotating_rates - self._stall_rates)
        max_rate = remaining / self.dt
        index = (rates <= max_rate.unsqueeze(-1)).sum(-1).sub(1).clamp(0, len(self._knots) - 2)
        r0 = rates.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        r1 = rates.gather(-1, (index + 1).unsqueeze(-1)).squeeze(-1)
        allowed_fraction = self._knots[index] + ((max_rate - r0) / (r1 - r0)).clamp(0, 1) * (self._knots[index + 1] - self._knots[index])
        budget_limit = continuous + allowed_fraction * (peak - continuous)
        limit = torch.minimum(speed_limit, budget_limit)
        applied = torch.where(valid, demand.sign() * torch.minimum(demand.abs(), limit), 0.0)
        normalized = ((applied.abs() - continuous) / (peak - continuous)).clamp(0, 1)
        index = torch.searchsorted(self._knots, normalized.contiguous(), right=True).sub(1).clamp(0, len(self._knots) - 2)
        r0 = rates.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        r1 = rates.gather(-1, (index + 1).unsqueeze(-1)).squeeze(-1)
        rate = r0 + (normalized - self._knots[index]) / (self._knots[index + 1] - self._knots[index]) * (r1 - r0)
        cost = torch.minimum(remaining, rate * self.dt)
        recovery_threshold = continuous * CONFIG["provisional"]["recovery_below_continuous_fraction"]
        recovery = (1 - applied.abs() / recovery_threshold).clamp(0, 1) * (self.dt / CONFIG["provisional"]["zero_load_full_recovery_s"])
        recovery = torch.where(valid, recovery, 0.0)
        self.burst_headroom.copy_((remaining - cost + recovery).clamp(0, 1))
        self.budget_consumed.copy_(cost)
        self.budget_recovered.copy_(torch.minimum(recovery, 1 - (remaining - cost)))
        self.torque_speed_limit_nm.copy_(speed_limit)
        self.instantaneous_limit_nm.copy_(torch.where(valid, limit, 0.0))
        self.envelope_violation_nm.copy_((applied.abs() - self.instantaneous_limit_nm).clamp_min(0))
        self.continuous_limit_nm.copy_(torch.minimum(continuous, speed_limit))
        self.clipping_nm.copy_((raw_demand_nm - applied).abs())
        self.continuous_overload_nm.copy_((raw_demand_nm.abs() - self.continuous_limit_nm).clamp_min(0))
        self.applied_overload_nm.copy_((applied.abs() - self.continuous_limit_nm).clamp_min(0))
        self.applied_overload_exposure_s.add_((self.applied_overload_nm > 1e-6).to(applied.dtype) * self.dt)
        self.applied_peak_exposure_s.add_((applied.abs() >= CONFIG["vendor"]["peak_output_torque_nm"] * CONFIG["provisional"]["peak_exposure_threshold_fraction"]).to(applied.dtype) * self.dt)
        self.estimated_phase_current_arms.copy_(applied.abs() / CONFIG["vendor"]["torque_constant_nm_per_arms"])
        self.estimated_phase_current_apk.copy_(self.estimated_phase_current_arms * math.sqrt(2))
        return applied
