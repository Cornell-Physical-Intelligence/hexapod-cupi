"""Read-only CAD inertia and sampled explicit-PD diagnostic; not a PhysX solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import mkii_fourbar_kinematics as kin

MOTOR = ROOT / "packages/hexapod_core/hexapod_core/rs05_v2.json"


def mm(a, b):
    # Avoid platform BLAS status-flag artefacts; explicitly validate results too.
    result = np.einsum("ij,jk->ik", a, b)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite matrix product")
    return result


def skew(p):
    return np.array([[0., -p[2], p[1]], [p[2], 0., -p[0]], [-p[1], p[0], 0.]])


def inertials(root):
    result = {}
    for link in root.findall("link"):
        element = link.find("inertial")
        origin = kin._origin(element)
        data = element.find("inertia").attrib
        tensor = np.array([[float(data["ixx"]), float(data["ixy"]), float(data["ixz"])],
                           [float(data["ixy"]), float(data["iyy"]), float(data["iyz"])],
                           [float(data["ixz"]), float(data["iyz"]), float(data["izz"])]])
        result[link.get("name")] = (float(element.find("mass").get("value")),
                                    origin, tensor)
    return result


def mass_matrix(root, contract, armature, epsilon, closed):
    """Kinetic-energy matrix in [base linear, base angular, joint] velocities.

    Base velocities are expressed in the nominal body frame. Joint Jacobians
    use centred finite differences about the nominal reset angles. The closed
    case eliminates passive coordinates using the ideal parallelogram branch;
    the open case deliberately omits the six external C-pin constraints.
    """
    names = contract["active_joint_names"] if closed else contract["tree_joint_names"]
    q = {name: contract["default_joint_positions_rad"][name] for name in names}
    frames = contract["joint_frames"]

    def poses(values):
        return kin.forward_kinematics(frames, kin.expand_active(values) if closed else values)

    properties = inertials(root)
    nominal = poses(q)
    jacobians = {name: (np.zeros((3, len(names))), np.zeros((3, len(names))))
                 for name in properties}
    for column, joint in enumerate(names):
        plus, minus = dict(q), dict(q)
        plus[joint] += epsilon
        minus[joint] -= epsilon
        positive, negative = poses(plus), poses(minus)
        for name, (_, origin, _) in properties.items():
            p = origin[:3, 3]
            jacobians[name][0][:, column] = (
                (positive[name][:3, :3] @ p + positive[name][:3, 3])
                - (negative[name][:3, :3] @ p + negative[name][:3, 3])) / (2 * epsilon)
            rate = mm((positive[name][:3, :3] - negative[name][:3, :3]) / (2 * epsilon),
                      nominal[name][:3, :3].T)
            jacobians[name][1][:, column] = np.array([
                rate[2, 1] - rate[1, 2], rate[0, 2] - rate[2, 0],
                rate[1, 0] - rate[0, 1]]) / 2

    mass = np.zeros((6 + len(names), 6 + len(names)))
    for name, (value, origin, tensor) in properties.items():
        pose = nominal[name]
        com = pose[:3, :3] @ origin[:3, 3] + pose[:3, 3]
        rotation = mm(pose[:3, :3], origin[:3, :3])
        jv = np.hstack((np.eye(3), -skew(com), jacobians[name][0]))
        jw = np.hstack((np.zeros((3, 3)), np.eye(3), jacobians[name][1]))
        tensor_world = mm(mm(rotation, tensor), rotation.T)
        mass += value * mm(jv.T, jv) + mm(mm(jw.T, tensor_world), jw)
    for name in contract["active_joint_names"]:
        index = 6 + names.index(name)
        mass[index, index] += armature
    if not np.isfinite(mass).all() or np.linalg.eigvalsh(mass)[0] <= 0:
        raise ValueError("Nonfinite or nonpositive kinetic-energy matrix")
    return mass


def effective_motor_inertia(mass, motor_indices):
    """Free passive/base response: invert the selected motor mobility block."""
    mobility = np.linalg.inv(mass)[np.ix_(motor_indices, motor_indices)]
    return np.linalg.inv(mobility), 1 / np.diag(mobility)


def step_matrix(inertia, stiffness, damping, dt, integration):
    factor = {"semi_implicit_euler": 1., "constant_acceleration_zoh": .5}[integration]
    return np.array([[1 - factor * dt * dt * stiffness / inertia,
                      dt - factor * dt * dt * damping / inertia],
                     [-dt * stiffness / inertia, 1 - dt * damping / inertia]])


def critical_timestep(inertia, stiffness, damping, integration):
    if integration == "semi_implicit_euler":
        return (-damping + np.sqrt(damping * damping + 4 * stiffness * inertia)) / stiffness
    if integration == "constant_acceleration_zoh":
        return min(2 * inertia / damping, 2 * damping / stiffness)
    raise ValueError(integration)


def scenario(mass, indices, active_names, kp, kd):
    effective, individual = effective_motor_inertia(mass, indices)
    values, vectors = np.linalg.eigh(effective)
    smallest = float(values[0])
    diagnostics = {}
    for method in ("semi_implicit_euler", "constant_acceleration_zoh"):
        samples = []
        for dt in (.005, .0025, .00125):
            poles = np.linalg.eigvals(step_matrix(smallest, kp, kd, dt, method))
            samples.append({"dt_s": dt, "poles_real_imag": [[float(x.real), float(x.imag)] for x in poles],
                            "spectral_radius": float(np.max(np.abs(poles))),
                            "linear_mode_stable": bool(np.max(np.abs(poles)) < 1)})
        diagnostics[method] = {
            "critical_dt_s": float(critical_timestep(smallest, kp, kd, method)), "samples": samples}
    return {"minimum_motor_mode_inertia_kg_m2": smallest,
            "maximum_motor_mode_inertia_kg_m2": float(values[-1]),
            "individual_effective_inertia_kg_m2": dict(zip(active_names, individual.tolist())),
            "lightest_mode_components": dict(zip(active_names, vectors[:, 0].tolist())),
            "integration_diagnostics": diagnostics}


def run():
    np.seterr(all="raise")
    sources = [kin.URDF, kin.PINS, kin.CONTRACT, MOTOR,
               ROOT / "tools/mkii_fourbar_kinematics.py", ROOT / "tools/audit_mkii_stance.py",
               Path(__file__).resolve()]
    before = {str(path.relative_to(ROOT)): kin.sha256(path) for path in sources}
    root, _, _ = kin.load_model()
    contract = json.loads(kin.CONTRACT.read_text())
    motor = json.loads(MOTOR.read_text())["provisional"]
    kp, kd, armature = (motor[key] for key in
        ("stiffness_nm_per_rad", "damping_nm_s_per_rad", "armature_kg_m2"))
    matrices, fd_errors = {}, {}
    for closed in (True, False):
        tag = "ideal_closed" if closed else "unclosed_tree_limiting_model"
        matrices[tag] = mass_matrix(root, contract, armature, 1e-6, closed)
        alternative = mass_matrix(root, contract, armature, 1e-5, closed)
        fd_errors[tag] = float(np.max(np.abs(matrices[tag] - alternative)))
        if fd_errors[tag] > 1e-8:
            raise ValueError("Finite-difference step check failed")

    active = contract["active_joint_names"]
    tree = contract["tree_joint_names"]
    mapping = np.zeros((36, 24))
    mapping[:6, :6] = np.eye(6)
    for name in active:
        mapping[6 + tree.index(name), 6 + active.index(name)] = 1
    for name, relation in contract["passive_relations"].items():
        mapping[6 + tree.index(name), 6 + active.index(relation["source_joint"])] = relation["multiplier"]
    projected = mm(mm(mapping.T, matrices["unclosed_tree_limiting_model"]), mapping)
    projection_error = float(np.max(np.abs(projected - matrices["ideal_closed"])))
    if projection_error > 1e-8:
        raise ValueError("Independent closed-coordinate projection check failed")

    results = {}
    for tag, matrix in matrices.items():
        names = active if tag == "ideal_closed" else tree
        indices = [names.index(name) for name in active]
        results[tag] = {
            "fixed_body": scenario(matrix[6:, 6:], indices, active, kp, kd),
            "free_body": scenario(matrix, [i + 6 for i in indices], active, kp, kd)}
    if any(kin.sha256(path) != before[str(path.relative_to(ROOT))] for path in sources):
        raise ValueError("An input changed during this audit; rerun against a stable source snapshot")
    return {"schema": "hexapod.fourbar.explicit_pd_inertia_diagnostic.v1",
            "scope": "Nominal CAD pose, local linearization; no live PhysX or hardware stability proof",
            "source_sha256": before, "body_count": len(root.findall("link")),
            "total_mass_kg": sum(x[0] for x in inertials(root).values()),
            "motor_parameters": {"kp_nm_per_rad": kp, "kd_nm_s_per_rad": kd,
                                 "active_armature_kg_m2": armature},
            "finite_difference_max_matrix_difference": fd_errors,
            "closed_vs_projected_tree_max_matrix_difference": projection_error,
            "scenarios": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write a NEW report; omit for stdout")
    args = parser.parse_args()
    payload = json.dumps(run(), indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as stream:
            stream.write(payload)
        print(args.out)
    else:
        print(payload, end="")
