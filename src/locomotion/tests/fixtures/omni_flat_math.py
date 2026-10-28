"""CPU-testable command, reward and evaluation contracts for omni locomotion."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import math
import torch


def sample_commands(count, device, max_speed=.20, max_yaw=.40):
    """10% stand, 20% pure yaw, 45% translation, 25% combined; no preferred bearing."""
    u = torch.rand(count, 4, device=device)
    angle = 2 * math.pi * u[:, 0]
    speed = .06 + (max_speed - .06) * u[:, 1]
    sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
    # Combined commands include near-zero yaw for long-radius survey curves.
    # Pure turns retain useful angular speed; their zero case is the stand mode.
    yaw = torch.where(u[:, 3] >= .75, max_yaw * u[:, 2],
                      .12 + (max_yaw - .12) * u[:, 2]) * sign
    category = u[:, 3]
    translating = category >= .30
    turning = ((category >= .10) & (category < .30)) | (category >= .75)
    return torch.stack((speed * angle.cos() * translating,
                        speed * angle.sin() * translating, yaw * turning), -1)


def slew_commands(current, target, dt, linear_accel=.25, yaw_accel=.8):
    """Bound vector acceleration, preserving bearing; yaw has its own units."""
    delta = target[:, :2] - current[:, :2]
    factor = (linear_accel * dt / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.)
    planar = current[:, :2] + delta * factor
    yaw = current[:, 2] + (target[:, 2] - current[:, 2]).clamp(-yaw_accel * dt, yaw_accel * dt)
    return torch.cat((planar, yaw[:, None]), -1)


def tracking_terms(velocity, yaw_rate, commands):
    """Bearing-invariant tracking; overspeed and commanded yaw are handled explicitly."""
    speed = commands[:, :2].norm(dim=-1)
    sigma = .04 + .30 * speed
    linear = torch.exp(-((velocity - commands[:, :2]) ** 2).sum(-1) / sigma.square())
    yaw_sigma = .10 + .25 * commands[:, 2].abs()
    yaw = torch.exp(-(yaw_rate - commands[:, 2]).square() / yaw_sigma.square())
    progress = ((velocity * commands[:, :2]).sum(-1) / speed.square().clamp_min(.0025)).clamp(-1., 1.)
    progress = torch.where(speed > .03, progress, torch.zeros_like(progress))
    yaw_progress = (yaw_rate * commands[:, 2] / commands[:, 2].square().clamp_min(.01)).clamp(-1., 1.)
    yaw_progress = torch.where(commands[:, 2].abs() > .05, yaw_progress, torch.zeros_like(yaw_progress))
    return {"linear_tracking": linear, "yaw_tracking": yaw,
            "linear_progress": progress, "yaw_progress": yaw_progress}


def evaluation_scenarios():
    rows = [{"name": "stand", "command": [0., 0., 0.]}]
    for speed in (.10, .20):
        for i in range(16):
            a = 2 * math.pi * i / 16
            rows.append({"name": f"translate_{speed:.2f}_{i*22.5:g}deg",
                         "command": [speed * math.cos(a), speed * math.sin(a), 0.]})
    for yaw in (-.4, -.2, .2, .4):
        rows.append({"name": f"turn_{yaw:+.2f}", "command": [0., 0., yaw]})
    for i in range(4):
        a = math.pi * i / 2
        for yaw in (-.2, .2):
            rows.append({"name": f"combined_{i*90}deg_{yaw:+.2f}",
                         "command": [.10 * math.cos(a), .10 * math.sin(a), yaw]})
    for speed, yaw_magnitude, label in ((.15, .05, 'gentle_arc'), (.20, .40, 'tight_arc')):
        for i in range(8):
            a = math.pi * i / 4
            for yaw in (-yaw_magnitude, yaw_magnitude):
                rows.append({'name':f'{label}_{i*45}deg_{yaw:+.2f}',
                             'command':[speed*math.cos(a),speed*math.sin(a),yaw]})
    return rows


def transition_sequence():
    """Continuous episode; turns, reversals and stop transitions cannot be hidden by resets."""
    return [
        ("stand", 3., [0., 0., 0.]), ("forward", 4., [.10, 0., 0.]),
        ("left", 4., [0., .10, 0.]), ("reverse", 4., [-.10, 0., 0.]),
        ("right", 4., [0., -.10, 0.]), ("turn_left", 4., [0., 0., .25]),
        ("turn_right", 4., [0., 0., -.25]),
        ("diagonal_turn_left", 4., [.071, .071, .20]),
        ("reverse_diagonal_turn_right", 4., [-.071, -.071, -.20]),
        ("gentle_arc", 6., [.15, 0., .05]),
        ("strafe_arc", 6., [0., .10, .15]),
        ("s_curve", 10., [.12, 0., 0.]),
        ("fixed_heading_bend", 8., [.10, 0., 0.]),
        ("stop", 5., [0., 0., 0.]),
    ]


def trajectory_command(name, elapsed, duration, command):
    """Planner-like continuous profiles; translation direction and heading are independent."""
    if name == 's_curve':
        return [.12, 0., .15 * math.sin(2 * math.pi * elapsed / duration)]
    if name == 'fixed_heading_bend':
        angle = .5 * math.pi * elapsed / duration
        return [.10 * math.cos(angle), .10 * math.sin(angle), 0.]
    return command


def integrate_body_twist(pose, command, dt):
    """Exact planar constant-twist integration; x/y/yaw, heading follows forward axis.

    This is a reference trajectory calculation, never a constraint on simulator motion.
    It handles straight motion and turn-in-place without a singular turning radius.
    """
    angle = command[:, 2] * dt
    # torch.sinc(x) = sin(pi*x)/(pi*x); both expressions are smooth at zero yaw.
    a = dt * torch.sinc(angle / math.pi)
    b = dt * .5 * angle * torch.sinc(angle / (2 * math.pi)).square()
    local_x = a * command[:, 0] - b * command[:, 1]
    local_y = b * command[:, 0] + a * command[:, 1]
    c, s = pose[:, 2].cos(), pose[:, 2].sin()
    return torch.stack((pose[:, 0] + c*local_x - s*local_y,
                        pose[:, 1] + s*local_x + c*local_y,
                        pose[:, 2] + angle), -1)


class ObservationHistory:
    """Idempotent reads within a step; partial resets cannot leak prior episodes."""
    def __init__(self, count, frames, width, device):
        self.data = torch.zeros(count, frames, width, device=device)
        self.fresh = torch.ones(count, dtype=torch.bool, device=device)
        self.step = -1

    def reset(self, ids):
        self.data[ids] = 0
        self.fresh[ids] = True

    def observe(self, current, step):
        if step != self.step:
            self.data[:, :-1] = self.data[:, 1:].clone()
            self.step = step
        self.data[:, -1] = current
        self.data[self.fresh] = current[self.fresh, None, :]
        self.fresh.zero_()
        return self.data.flatten(1).clone()


def scenario_gate(row):
    speed = math.hypot(*row["command"][:2]); yaw = abs(row["command"][2])
    return (row["terminations"] == 0 and row["truncations"] == 0
            and row["nonfoot_fraction"] <= .001 and row["torque_saturation_fraction"] <= .005
            and row["planar_error_mps"] <= max(.025, .25 * speed)
            and row["yaw_error_rad_s"] <= max(.06, .25 * yaw)
            and row["tilt_rms_deg"] <= 5 and row["vertical_velocity_rms_mps"] <= .04)
