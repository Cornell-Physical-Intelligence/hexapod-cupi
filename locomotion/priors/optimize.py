"""Optimize a periodic tripod cycle with full-body inverse dynamics."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
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
    target_curvature_weight: float = 0.
    root_velocity_weight: float = 0.
    left_mps: float = 0.
    yaw_rate_rad_s: float = 0.

    @property
    def command(self):
        return (self.forward_mps, self.left_mps, self.yaw_rate_rad_s)

    def validate(self):
        values = [self.period_s, self.control_dt_s, self.forward_mps,
                  self.duty_factor, self.lift_m, self.friction,
                  self.target_curvature_weight, self.root_velocity_weight,
                  self.left_mps, self.yaw_rate_rad_s]
        if not np.isfinite(values).all() or self.control_dt_s != .02:
            raise ValueError("Finite configuration and unchanged 50 Hz targets required")
        if (not .4 <= self.period_s <= 3.
                or np.hypot(self.forward_mps, self.left_mps) > .1000000001
                or abs(self.yaw_rate_rad_s) > .2):
            raise ValueError("Motion requires period 0.4..3 s, speed <=0.1 m/s and yaw <=0.2 rad/s")
        if not .5 <= self.duty_factor <= .8 or not .005 <= self.lift_m <= .03:
            raise ValueError("Invalid stance fraction or swing clearance")
        if not 0 < self.friction <= 1. or type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("Invalid contact friction or solver budget")
        if min(self.target_curvature_weight, self.root_velocity_weight) < 0:
            raise ValueError('Smoothing weights must be nonnegative')
        if not np.isclose(self.period_s/self.control_dt_s, round(self.period_s/self.control_dt_s)):
            raise ValueError("Cycle must contain complete 50 Hz controls")


def schedule(config, times):
    phase = (np.asarray(times)[:, None]/config.period_s + np.array([0, .5, 0, .5, 0, .5])) % 1
    return phase, phase < config.duty_factor


def planar_pose(config, time_s):
    """Integrate a constant body command in native +X-left, -Y-forward axes."""
    angle = config.yaw_rate_rad_s*time_s
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
    velocity = np.array([config.left_mps, -config.forward_mps, 0.])
    tangent = np.array([-velocity[1], velocity[0], 0.])
    displacement = time_s*(np.sinc(angle/np.pi)*velocity
        + .5*angle*np.sinc(angle/(2*np.pi))**2*tangent)
    return rotation, displacement


def cycle_state(config, state, *, reverse=False, velocity=False):
    """Transform a root state between adjacent cycles; joint states repeat."""
    rotation, shift = planar_pose(config, config.period_s)
    result = state*1.
    if reverse:
        result[:3] = rotation.T@(state[:3] if velocity else state[:3]-shift)
    else:
        result[:3] = rotation@state[:3] + (0 if velocity else shift)
    if not velocity:
        result[5] = state[5] + (-1 if reverse else 1)*config.yaw_rate_rad_s*config.period_s
    return result


def initial_trajectory(model, config):
    n = round(config.period_s/config.control_dt_s)
    times = np.arange(n+1)*config.control_dt_s
    phase, stance = schedule(config, times)
    velocity = np.array([config.left_mps, -config.forward_mps, 0.])
    q = np.tile(np.r_[0, 0, .09780231400684256, 0, 0, 0, NEUTRAL], (n+1, 1))
    q[:, :3] += np.array([planar_pose(config, t)[1] for t in times])
    q[:, 5] = times*config.yaw_rate_rad_s
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
            if config.yaw_rate_rad_s:
                center = times[k] + config.period_s*(config.duty_factor/2-ph)
                rotation, origin = planar_pose(config, center)
                feet[k, leg] = rotation@nominal[leg]+origin
                if not stance[k, leg]:
                    next_rotation, next_origin = planar_pose(config, center+config.period_s)
                    blend = 3*u*u-2*u*u*u
                    feet[k, leg] = ((1-blend)*feet[k, leg]
                        + blend*(next_rotation@nominal[leg]+next_origin))
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
    # Include the rigid transform across each cycle boundary.
    q[-1] = cycle_state(config, q[0])
    extended = np.vstack([cycle_state(config, q[-2], reverse=True), q,
                          cycle_state(config, q[1])])
    v = (extended[2:]-extended[:-2])/(2*config.control_dt_s)
    a = np.diff(v, axis=0)/config.control_dt_s
    _, force_stance = schedule(config, times[:-1]+config.control_dt_s/2)
    f = np.zeros((n, 6, 3))
    f[:, :, 2] = force_stance*model.mass*9.81/force_stance.sum(axis=1)[:, None]
    return q, v, a, f, feet


def solver_options(config, mu_strategy):
    if mu_strategy not in ('monotone', 'adaptive'):
        raise ValueError('Unknown barrier update strategy')
    return {'max_iter': config.max_iterations, 'tol': 1e-6, 'constr_viol_tol': 1e-7,
        'acceptable_tol': 1e-5, 'acceptable_constr_viol_tol': 1e-6,
        'print_level': 4, 'sb': 'yes', 'linear_solver': 'mumps', 'mu_strategy': mu_strategy}


def build_problem(model, config, *, mu_strategy='monotone'):
    config.validate()
    q0, v0, a0, f0, desired_feet = initial_trajectory(model, config)
    n, dt = len(a0), config.control_dt_s
    opt = ca.Opti()
    q, v = opt.variable(24, n+1), opt.variable(24, n+1)
    a, f = opt.variable(24, n), opt.variable(18, n)
    opt.set_linear_scale(f, model.mass*9.81)
    opt.set_initial(q, q0.T); opt.set_initial(v, v0.T)
    opt.set_initial(a, a0.T); opt.set_initial(f, f0.reshape(n, 18).T)
    opt.subject_to(opt.bounded(.075, q[2, :], .13))
    opt.subject_to(opt.bounded(-.15, q[3:6, :]-q0[:, 3:6].T, .15))
    opt.subject_to(opt.bounded(ca.repmat(ca.DM(model.lower+.005), 1, n+1), q[6:, :],
        ca.repmat(ca.DM(model.upper-.005), 1, n+1)))
    opt.subject_to(opt.bounded(-8., v[6:, :], 8.))
    opt.subject_to(opt.bounded(-1., v[:6, :], 1.))
    opt.subject_to(q[:2, 0] == 0)
    opt.subject_to(q[:, -1] == cycle_state(config, q[:, 0]))
    opt.subject_to(v[:, -1] == cycle_state(config, v[:, 0], velocity=True))
    phase, node_stance = schedule(config, np.arange(n+1)*dt)
    _, force_stance = schedule(config, (np.arange(n)+.5)*dt)
    objective = 0
    motor_targets, torques = [], []
    for k in range(n+1):
        # Use cycle closure for terminal contacts; avoid duplicate equality rows.
        if k < n:
            feet, _ = model.kinematics(q[:, k])
            for leg in range(6):
                target = desired_feet[k, leg]
                if node_stance[k, leg] or np.isclose(phase[k, leg], config.duty_factor):
                    opt.subject_to(feet[:, leg] == target)
                else:
                    opt.subject_to(feet[2, leg] >= .85*target[2])
                    objective += 100*ca.sumsqr(feet[:, leg]-target)
        objective += 10*ca.sumsqr(q[:3, k]-q0[k, :3]) + 2*ca.sumsqr(q[3:6, k]-q0[k, 3:6])
        objective += .02*ca.sumsqr(q[6:, k]-q0[k, 6:])
        heading = q[5, k]
        desired_velocity = ca.vertcat(
            config.left_mps*ca.cos(heading)+config.forward_mps*ca.sin(heading),
            config.left_mps*ca.sin(heading)-config.forward_mps*ca.cos(heading), 0.)
        if not config.yaw_rate_rad_s:
            desired_velocity = ca.DM([config.left_mps, -config.forward_mps, 0.])
        objective += config.root_velocity_weight*ca.sumsqr((v[:3, k]-desired_velocity)/.05)
        if config.yaw_rate_rad_s:
            objective += config.root_velocity_weight*((v[5, k]-config.yaw_rate_rad_s)/.2)**2
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
        curvature = targets[:, (k+1) % n]-2*targets[:, k]+targets[:, (k-1) % n]
        objective += config.target_curvature_weight*ca.sumsqr(curvature/.04)
    opt.minimize(objective/n)
    opt.solver('ipopt', {'expand': True, 'print_time': False}, solver_options(config, mu_strategy))
    return opt, {'q': q, 'v': v, 'a': a, 'force': f,
                 'target': targets, 'torque': ca.horzcat(*torques)}, desired_feet


def audit(model, config, arrays, desired_feet):
    config.validate()
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite trajectory cannot pass the feasibility audit')
    q, v, a = (arrays[key] for key in ('q', 'v', 'a'))
    forces = arrays['force'].reshape(-1, 6, 3)
    target, dt = arrays['target'], config.control_dt_s
    n = len(a)
    _, stance = schedule(config, np.arange(n+1)*dt)
    _, force_stance = schedule(config, (np.arange(n)+.5)*dt)
    residuals, torques, pd_errors, contact_errors, swing_clearance = [], [], [], [], []
    for k in range(n):
        mid_q = q[k]+dt*v[k]/2+dt*dt*a[k]/8
        mid_v = v[k]+dt*a[k]/2
        value = np.asarray(model.inverse_dynamics(mid_q, mid_v, a[k], forces[k].T)).ravel()
        residuals.append(value[:6]); torques.append(value[6:])
        pd_errors.extend(target[k]-mid_q[6:]-(value[6:]+model.kd*mid_v[6:])/12.)
        feet = np.asarray(model.kinematics(q[k])[0]).T
        contact_errors.extend((feet[stance[k]]-desired_feet[k, stance[k]]).ravel())
        swing_clearance.extend(feet[~stance[k], 2]-.85*desired_feet[k, ~stance[k], 2])
    torque = np.array(torques)
    orientation_reference = np.zeros_like(q[:, 3:6])
    orientation_reference[:, 2] = np.arange(n+1)*dt*config.yaw_rate_rad_s
    violations = {
        'root_force_moment': float(np.max(np.abs(residuals))),
        'position_integration': float(np.max(abs(np.diff(q, axis=0)-dt*(v[:-1]+v[1:])/2))),
        'velocity_integration': float(np.max(abs(np.diff(v, axis=0)-dt*a))),
        'periodic_position': float(np.max(abs(q[-1]-cycle_state(config, q[0])))),
        'periodic_velocity': float(np.max(abs(v[-1]-cycle_state(config, v[0], velocity=True)))),
        'stance_position': float(np.max(np.abs(contact_errors))),
        'swing_clearance': float(max(0, -min(swing_clearance, default=0))),
        'swing_force': float(np.max(abs(forces[~force_stance]), initial=0)),
        'negative_normal_force': float(max(0, -forces[:, :, 2].min())),
        'friction': float(np.maximum(np.linalg.norm(forces[:, :, :2], axis=2)
            -config.friction*forces[:, :, 2], 0).max()),
        'torque_cap': float(max(0, abs(torque).max()-1.6)),
        'pd_target_identity': float(np.max(np.abs(pd_errors))),
        'stored_torque_identity': float(np.max(abs(arrays['torque']-torque))),
        'joint_speed': float(max(0, abs(v[:, 6:]).max()-8.)),
        'root_speed': float(max(0, abs(v[:, :6]).max()-1.)),
        'root_height': float(max(0, .075-q[:, 2].min(), q[:, 2].max()-.13)),
        'root_orientation': float(max(0, abs(q[:, 3:6]-orientation_reference).max()-.15)),
        'normal_force_cap': float(max(0, forces[:, :, 2].max()-model.mass*9.81)),
        'cyclic_target_slew': float(max(0, abs(np.roll(target, -1, axis=0)-target).max()-.04)),
        'target_range': float(max(0, abs(target-NEUTRAL).max()-.35)),
        'target_joint_limits': float(max(0, (model.lower-target).max(), (target-model.upper).max())),
        'joint_position': float(max(0, (model.lower+.005-q[:, 6:]).max(),
                                    (q[:, 6:]-model.upper+.005).max())),
    }
    return {'constraint_violation': violations, 'passed': all(x <= 2e-5 for x in violations.values()),
        'maximum_motor_torque_nm': float(abs(torque).max()),
        'maximum_target_step_rad': float(abs(np.roll(target, -1, axis=0)-target).max()),
        'mean_root_velocity_native_mps': v[:, :3].mean(axis=0).tolist(),
        'scope': 'CPU collocation feasibility; native 400 Hz replay and mesh contacts remain required.'}


def restart_values(path, config, model):
    """Read a model-bound primal iterate from the same optimization problem."""
    path = Path(path)
    declaration = json.loads(path.with_name('INPUT.json').read_text())
    result = json.loads(path.with_name('RESULT.json').read_text())
    previous = Config(**declaration['config'])
    previous.validate()
    if (declaration['model'] != model.identity()
            or replace(previous, max_iterations=config.max_iterations) != config
            or result.get('trajectory_sha256') != digest(path)):
        raise ValueError('Restart model, problem configuration or trajectory bytes differ')
    n = round(config.period_s/config.control_dt_s)
    shapes = {'q': (n+1, 24), 'v': (n+1, 24), 'a': (n, 24), 'force': (n, 18)}
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in shapes}
    if any(arrays[name].shape != shape or not np.isfinite(arrays[name]).all()
           for name, shape in shapes.items()):
        raise ValueError('Restart requires complete finite primal variables')
    return arrays


def run(output, config, root=None, *, initial=None, mu_strategy='monotone'):
    output = Path(output)
    model = RobotModel() if root is None else RobotModel(root)
    restart = restart_values(initial, config, model) if initial is not None else None
    output.mkdir(parents=True, exist_ok=False)
    (output/'source').mkdir()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        shutil.copy2(path, output/'source'/path.name)
    declaration = {'schema': 'canonical_full_body_trajectory_optimization_v2',
        'config': asdict(config), 'model': model.identity(), 'casadi_version': ca.__version__,
        'solver_options': solver_options(config, mu_strategy),
        'variable_scaling': {'contact_force_n': model.mass*9.81},
        'source_files': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'method': 'Fixed-contact-schedule midpoint inverse dynamics, ZYX floating root, all 19 rigid bodies.',
        'limitations': ['Point contacts omit mesh deformation and impacts.',
            'The inverse PD relation holds at collocation midpoints; native held-target replay is separate.'],
        'command': list(config.command),
        'stage2_complete': False, 'native_accepted': False}
    if initial is not None:
        initial = Path(initial)
        (output/'initial').mkdir()
        shutil.copy2(initial, output/'initial/trajectory.npz')
        for name in ('INPUT.json', 'RESULT.json'):
            shutil.copy2(initial.with_name(name), output/'initial'/name)
        declaration['initialization'] = {'kind': 'saved_primal_iterate',
            'path': str(initial.resolve()), 'files': {
                p.name: digest(p) for p in sorted((output/'initial').iterdir())}}
    (output/'INPUT.json').write_text(json.dumps(declaration, indent=2)+'\n')
    started = time.monotonic()
    try:
        opt, variables, feet = build_problem(model, config, mu_strategy=mu_strategy)
        if restart is not None:
            for name, values in restart.items():
                opt.set_initial(variables[name], values.T)
    except (ValueError, RuntimeError) as error:
        result = {'status': 'initialization_failed', 'error': str(error),
                  'audit': {'passed': False}, 'stage2_complete': False}
        (output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    try:
        solution = opt.solve()
        status = 'solved'
    except RuntimeError:
        solution = opt.debug
        status = 'solver_failed'
    arrays = {key: np.asarray(solution.value(value)).T for key, value in variables.items()}
    arrays['desired_feet'] = feet
    np.savez_compressed(output/'trajectory.npz', **arrays)
    (output/'SOLVER.json').write_text(json.dumps({'status': status,
        'return_status': opt.stats()['return_status'], 'iterations': opt.stats()['iter_count']}, indent=2)+'\n')
    try:
        checked = audit(model, config, arrays, feet)
    except ValueError as error:
        checked = {'passed': False, 'error': str(error)}
    result = {'status': status, 'solver_status': opt.stats()['return_status'],
              'solver_iterations': opt.stats()['iter_count'],
              'wall_seconds': time.monotonic()-started, 'audit': checked,
              'trajectory_sha256': digest(output/'trajectory.npz'), 'stage2_complete': False}
    (output/'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'SHA256.json').write_text(json.dumps({str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*'))
                                                if p.is_file()}, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', action='store_true', help='Solve the frozen 20-command coverage bank')
    parser.add_argument('--initial-trajectory', type=Path,
                        help='Restart one command from a saved primal iterate of the same problem')
    parser.add_argument('--mu-strategy', choices=['monotone', 'adaptive'], default='monotone',
                        help='Ipopt barrier update; retain the objective and convergence tolerances')
    parser.add_argument('--forward-mps', type=float, default=.05)
    parser.add_argument('--left-mps', type=float, default=0.)
    parser.add_argument('--yaw-rate-rad-s', type=float, default=0.)
    parser.add_argument('--period-s', type=float, default=1.2)
    parser.add_argument('--max-iterations', type=int, default=600)
    parser.add_argument('--target-curvature-weight', type=float, default=0.)
    parser.add_argument('--root-velocity-weight', type=float, default=0.)
    args = parser.parse_args()
    if args.bank and args.initial_trajectory is not None:
        parser.error('A saved primal iterate applies to one command, not a bank')
    config = Config(forward_mps=args.forward_mps,
        left_mps=args.left_mps, yaw_rate_rad_s=args.yaw_rate_rad_s,
        period_s=args.period_s, max_iterations=args.max_iterations,
        target_curvature_weight=args.target_curvature_weight, root_velocity_weight=args.root_velocity_weight)
    if args.bank:
        from .commands import motion_cases
        args.output.mkdir(parents=True, exist_ok=False)
        results = []
        for case in motion_cases():
            forward, left, yaw = case['command']
            directory = f'command_{len(results):02d}'
            result = run(args.output/directory, replace(config, forward_mps=forward,
                         left_mps=left, yaw_rate_rad_s=yaw), mu_strategy=args.mu_strategy)
            results.append({**case, 'path': directory, 'result': result})
            (args.output/'coverage.json').write_text(json.dumps({
                'schema': 'hexapod_amp_optimization_bank_v1', 'cases': results,
                'native_admitted': False}, indent=2, allow_nan=False)+'\n')
        return 0 if all(r['result']['status'] == 'solved'
                        and r['result']['audit']['passed'] for r in results) else 1
    result = run(args.output, config, initial=args.initial_trajectory, mu_strategy=args.mu_strategy)
    return 0 if result['status'] == 'solved' and result['audit']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
