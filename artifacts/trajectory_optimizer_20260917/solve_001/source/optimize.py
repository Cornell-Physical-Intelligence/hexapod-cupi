"""Optimize a periodic forward tripod cycle with full-body inverse dynamics."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np
from scipy.optimize import least_squares

from .model import NEUTRAL, RobotModel, digest


@dataclass(frozen=True)
class Config:
    period_s: float = 1.2
    control_dt_s: float = .02
    forward_mps: float = .05
    duty_factor: float = .65
    lift_m: float = .018
    friction: float = .8
    max_iterations: int = 600

    def validate(self):
        values = [self.period_s, self.control_dt_s, self.forward_mps,
                  self.duty_factor, self.lift_m, self.friction]
        if not np.isfinite(values).all() or self.control_dt_s != .02:
            raise ValueError("Finite configuration and unchanged 50 Hz targets required")
        if not .4 <= self.period_s <= 3. or not 0 <= self.forward_mps <= .1:
            raise ValueError("Forward experiment requires period 0.4..3 s and speed 0..0.1 m/s")
        if not .5 <= self.duty_factor <= .8 or not .005 <= self.lift_m <= .03:
            raise ValueError("Invalid stance fraction or swing clearance")
        if not 0 < self.friction <= 1. or type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("Invalid contact friction or solver budget")
        if not np.isclose(self.period_s/self.control_dt_s, round(self.period_s/self.control_dt_s)):
            raise ValueError("Cycle must contain complete 50 Hz controls")


def schedule(config, times):
    phase = (np.asarray(times)[:, None]/config.period_s + np.array([0, .5, 0, .5, 0, .5])) % 1
    return phase, phase < config.duty_factor


def initial_trajectory(model, config):
    n = round(config.period_s/config.control_dt_s)
    times = np.arange(n+1)*config.control_dt_s
    phase, stance = schedule(config, times)
    velocity = np.array([0., -config.forward_mps, 0.])
    q = np.tile(np.r_[0, 0, .09780231400684256, 0, 0, 0, NEUTRAL], (n+1, 1))
    q[:, :3] += times[:, None]*velocity
    nominal = np.asarray(model.kinematics(q[0])[0]).T
    nominal[:, 2] = 0
    feet = np.empty((n+1, 6, 3))
    sx = ca.SX.sym('seed_q', 24)
    foot_function = ca.Function('seed_feet', [sx], [ca.vec(model.kinematics(sx)[0])])
    jac_function = ca.Function('seed_jac', [sx], [ca.jacobian(foot_function(sx), sx)[:, 6:]])
    for k in range(n+1):
        for leg in range(6):
            ph = phase[k, leg]
            if stance[k, leg]:
                displacement = config.period_s*(config.duty_factor/2-ph)*velocity
                height = 0.
            else:
                u = (ph-config.duty_factor)/(1-config.duty_factor)
                displacement = config.period_s*(-config.duty_factor/2
                    -(1-config.duty_factor)*u+3*u*u-2*u*u*u)*velocity
                height = config.lift_m*np.sin(np.pi*u)**2
            feet[k, leg] = nominal[leg] + times[k]*velocity + displacement
            feet[k, leg, 2] = height
        def pack(joint):
            return np.r_[q[k, :6], joint]
        def residual(joint):
            return np.asarray(foot_function(pack(joint))).ravel()-feet[k].ravel()
        fitted = least_squares(residual, q[k-1, 6:] if k else NEUTRAL,
            jac=lambda joint: np.asarray(jac_function(pack(joint))),
            bounds=(model.lower+.005, model.upper-.005),
            ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=100)
        if np.max(abs(fitted.fun)) > 1e-5:
            raise ValueError("Initial foot path lies outside the joint workspace")
        q[k, 6:] = fitted.x
    # Periodic finite differences include the translated next cycle.
    displacement = q[-1]-q[0]
    q[-1, 2:] = q[0, 2:]
    extended = np.vstack([q[-2]-displacement, q, q[1]+displacement])
    v = (extended[2:]-extended[:-2])/(2*config.control_dt_s)
    a = np.diff(v, axis=0)/config.control_dt_s
    _, force_stance = schedule(config, times[:-1]+config.control_dt_s/2)
    f = np.zeros((n, 6, 3))
    f[:, :, 2] = force_stance*model.mass*9.81/force_stance.sum(axis=1)[:, None]
    return q, v, a, f, feet


def build_problem(model, config):
    config.validate()
    q0, v0, a0, f0, desired_feet = initial_trajectory(model, config)
    n, dt = len(a0), config.control_dt_s
    opt = ca.Opti()
    q, v = opt.variable(24, n+1), opt.variable(24, n+1)
    a, f = opt.variable(24, n), opt.variable(18, n)
    opt.set_initial(q, q0.T); opt.set_initial(v, v0.T)
    opt.set_initial(a, a0.T); opt.set_initial(f, f0.reshape(n, 18).T)
    opt.subject_to(opt.bounded(.075, q[2, :], .13))
    opt.subject_to(opt.bounded(-.15, q[3:6, :], .15))
    opt.subject_to(opt.bounded(ca.repmat(ca.DM(model.lower+.005), 1, n+1), q[6:, :],
        ca.repmat(ca.DM(model.upper-.005), 1, n+1)))
    opt.subject_to(opt.bounded(-8., v[6:, :], 8.))
    opt.subject_to(opt.bounded(-1., v[:6, :], 1.))
    opt.subject_to(q[:2, 0] == 0)
    shift = np.zeros(24); shift[1] = -config.forward_mps*config.period_s
    opt.subject_to(q[:, -1] == q[:, 0] + shift)
    opt.subject_to(v[:, -1] == v[:, 0])
    phase, node_stance = schedule(config, np.arange(n+1)*dt)
    _, force_stance = schedule(config, (np.arange(n)+.5)*dt)
    objective = 0
    motor_targets, torques = [], []
    for k in range(n+1):
        feet, _ = model.kinematics(q[:, k])
        for leg in range(6):
            target = desired_feet[k, leg]
            if node_stance[k, leg] or np.isclose(phase[k, leg], config.duty_factor):
                opt.subject_to(feet[:, leg] == target)
            else:
                opt.subject_to(feet[2, leg] >= .85*target[2])
                objective += 100*ca.sumsqr(feet[:, leg]-target)
        objective += 10*ca.sumsqr(q[:3, k]-q0[k, :3]) + 2*ca.sumsqr(q[3:6, k])
        objective += .02*ca.sumsqr(q[6:, k]-q0[k, 6:])
        if k == n:
            continue
        opt.subject_to(v[:, k+1] == v[:, k]+dt*a[:, k])
        opt.subject_to(q[:, k+1] == q[:, k]+dt*(v[:, k]+v[:, k+1])/2)
        mid_q = q[:, k]+dt*v[:, k]/2+dt*dt*a[:, k]/8
        mid_v = v[:, k]+dt*a[:, k]/2
        contact = ca.reshape(f[:, k], 3, 6)
        dynamics = model.inverse_dynamics(mid_q, mid_v, a[:, k], contact)
        opt.subject_to(dynamics[:3]/(model.mass*9.81) == 0)
        opt.subject_to(dynamics[3:6] == 0)
        tau = dynamics[6:]
        opt.subject_to(opt.bounded(-1.6, tau, 1.6))
        # Invert the unchanged PD law at the collocation state. Isaac replay
        # applies these held targets at 400 Hz and records the approximation.
        target = mid_q[6:]+(tau+ca.DM(model.kd)*mid_v[6:])/12.
        opt.subject_to(opt.bounded(NEUTRAL-.35, target, NEUTRAL+.35))
        opt.subject_to(opt.bounded(model.lower, target, model.upper))
        motor_targets.append(target); torques.append(tau)
        for leg in range(6):
            force = contact[:, leg]
            if not force_stance[k, leg]:
                opt.subject_to(force == 0)
            else:
                opt.subject_to(opt.bounded(0, force[2], model.mass*9.81))
                opt.subject_to(force[0]**2+force[1]**2 <= (config.friction*force[2])**2)
        objective += .005*ca.sumsqr(tau/1.6) + .0001*ca.sumsqr(mid_v[6:])
        objective += .00001*ca.sumsqr(contact)
    targets = ca.horzcat(*motor_targets)
    for k in range(n):
        step = targets[:, (k+1) % n]-targets[:, k]
        opt.subject_to(opt.bounded(-.040, step, .040))
        objective += .01*ca.sumsqr(step/.04)
    opt.minimize(objective/n)
    opt.solver('ipopt', {'expand': True, 'print_time': False}, {
        'max_iter': config.max_iterations, 'tol': 1e-6, 'constr_viol_tol': 1e-7,
        'acceptable_tol': 1e-5, 'acceptable_constr_viol_tol': 1e-6,
        'print_level': 4, 'sb': 'yes', 'linear_solver': 'mumps'})
    return opt, {'q': q, 'v': v, 'a': a, 'force': f,
                 'target': targets, 'torque': ca.horzcat(*torques)}, desired_feet


def audit(model, config, arrays, desired_feet):
    q, v, a = (arrays[key] for key in ('q', 'v', 'a'))
    forces = arrays['force'].reshape(-1, 6, 3)
    target, dt = arrays['target'], config.control_dt_s
    n = len(a)
    _, stance = schedule(config, np.arange(n+1)*dt)
    _, force_stance = schedule(config, (np.arange(n)+.5)*dt)
    residuals, torques, contact_errors, swing_clearance = [], [], [], []
    for k in range(n):
        mid_q = q[k]+dt*v[k]/2+dt*dt*a[k]/8
        mid_v = v[k]+dt*a[k]/2
        value = np.asarray(model.inverse_dynamics(mid_q, mid_v, a[k], forces[k].T)).ravel()
        residuals.append(value[:6]); torques.append(value[6:])
        feet = np.asarray(model.kinematics(q[k])[0]).T
        contact_errors.extend((feet[stance[k]]-desired_feet[k, stance[k]]).ravel())
        swing_clearance.extend(feet[~stance[k], 2]-.85*desired_feet[k, ~stance[k], 2])
    torque = np.array(torques)
    shift = np.zeros(24); shift[1] = -config.forward_mps*config.period_s
    violations = {
        'root_force_moment': float(np.max(abs(residuals))),
        'position_integration': float(np.max(abs(np.diff(q, axis=0)-dt*(v[:-1]+v[1:])/2))),
        'velocity_integration': float(np.max(abs(np.diff(v, axis=0)-dt*a))),
        'periodic_position': float(np.max(abs(q[-1]-q[0]-shift))),
        'periodic_velocity': float(np.max(abs(v[-1]-v[0]))),
        'stance_position': float(np.max(abs(contact_errors))),
        'swing_clearance': float(max(0, -min(swing_clearance, default=0))),
        'swing_force': float(np.max(abs(forces[~force_stance]), initial=0)),
        'negative_normal_force': float(max(0, -forces[:, :, 2].min())),
        'friction': float(np.maximum(np.linalg.norm(forces[:, :, :2], axis=2)
            -config.friction*forces[:, :, 2], 0).max()),
        'torque_cap': float(max(0, abs(torque).max()-1.6)),
        'cyclic_target_slew': float(max(0, abs(np.roll(target, -1, axis=0)-target).max()-.04)),
        'target_range': float(max(0, abs(target-NEUTRAL).max()-.35)),
        'joint_position': float(max(0, (model.lower+.005-q[:, 6:]).max(),
                                    (q[:, 6:]-model.upper+.005).max())),
    }
    return {'constraint_violation': violations, 'passed': all(x <= 2e-5 for x in violations.values()),
        'maximum_motor_torque_nm': float(abs(torque).max()),
        'maximum_target_step_rad': float(abs(np.roll(target, -1, axis=0)-target).max()),
        'mean_root_velocity_native_mps': v[:, :3].mean(axis=0).tolist(),
        'scope': 'CPU collocation feasibility; native 400 Hz replay and mesh contacts remain required.'}


def run(output, config, root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = RobotModel() if root is None else RobotModel(root)
    declaration = {'schema': 'canonical_full_body_trajectory_optimization_v1',
        'config': asdict(config), 'model': model.identity(), 'casadi_version': ca.__version__,
        'source_files': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'method': 'Fixed-contact-schedule midpoint inverse dynamics, ZYX floating root, all 19 rigid bodies.',
        'limitations': ['Point contacts omit mesh deformation and impacts.',
            'The inverse PD relation holds at collocation midpoints; native held-target replay is separate.'],
        'stage2_complete': False, 'native_accepted': False}
    (output/'INPUT.json').write_text(json.dumps(declaration, indent=2)+'\n')
    started = time.monotonic()
    opt, variables, feet = build_problem(model, config)
    try:
        solution = opt.solve()
        status = 'solved'
    except RuntimeError:
        solution = opt.debug
        status = 'solver_failed'
    arrays = {key: np.asarray(solution.value(value)).T for key, value in variables.items()}
    arrays['desired_feet'] = feet
    np.savez_compressed(output/'trajectory.npz', **arrays)
    result = {'status': status, 'solver_status': opt.stats()['return_status'],
              'solver_iterations': opt.stats()['iter_count'],
              'wall_seconds': time.monotonic()-started, 'audit': audit(model, config, arrays, feet),
              'trajectory_sha256': digest(output/'trajectory.npz'), 'stage2_complete': False}
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'SHA256.json').write_text(json.dumps({p.name: digest(p) for p in sorted(output.iterdir())
                                                if p.is_file()}, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forward-mps', type=float, default=.05)
    parser.add_argument('--period-s', type=float, default=1.2)
    parser.add_argument('--max-iterations', type=int, default=600)
    args = parser.parse_args()
    result = run(args.output, Config(forward_mps=args.forward_mps,
        period_s=args.period_s, max_iterations=args.max_iterations))
    return 0 if result['status'] == 'solved' and result['audit']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
