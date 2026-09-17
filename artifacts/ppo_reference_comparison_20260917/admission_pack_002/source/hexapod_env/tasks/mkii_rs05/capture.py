"""Per-substep diagnostic capture in the layout the frozen scorer reads.

``experiments/paper_walk/env.py:score_diagnostic`` is unchanged and grades a
directory that holds ``session.json``, ``initial_reset.json``,
``native_readback.json``, one ``control_trace.npz`` and the ``substeps_XXX.npz``
files. This module writes that layout from the Isaac Lab task. It stores the
joint channels in canonical per-leg order, because the scorer reindexes the
damping by the names in ``session.json``.

The capture aborts on the first failed row, keeps the partial evidence and
never resets or drops a row. Writing this layout grades nothing by itself: the
scorer recomputes every channel and decides.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "CAPTURE_ROW_KEYS",
    "CONTACT_CATEGORIES",
    "DiagnosticGeometry",
    "Rs05DiagnosticCapture",
    "capture_row",
    "rotation_matrix_xyzw",
]

#: Every key one substep row carries, in the order the writer stacks them.
CAPTURE_ROW_KEYS = (
    "pre_joint_position_rad", "pre_joint_velocity_rad_s",
    "joint_position_rad", "joint_velocity_rad_s",
    "root_pose_xyzw", "link_pose_xyzw", "root_com_velocity",
    "joint_target_rad", "computed_torque_nm", "applied_torque_nm", "effort_ceiling_nm",
    "native_input_pre_nm", "native_input_post_nm",
    "sequence", "control_index", "substep_index", "explicit_counter", "time_s",
    "contact_valid", "interval_valid", "truncated",
    "distal_contact", "distal_force_world_n", "nonfoot_contact", "nonfoot_force_world_n",
    "minimum_mesh_floor_m", "minimum_non_toe_floor_m", "interval_angle_rate_rad_s",
    "terminated",
)

#: Non-toe contact categories, in the column order of ``nonfoot_force_world_n``.
CONTACT_CATEGORIES = ("body", "coxa", "femur", "shaft")

#: Row gates the capture applies before it stores a row. They repeat the
#: prototype's abort conditions; the scorer applies its own gates again.
MAX_APPLIED_NM = 1.60001
MIN_PLATE_HEIGHT_M = 0.055
MIN_NON_TOE_CLEARANCE_M = -0.001
ROWS_PER_FILE = 800


def rotation_matrix_xyzw(quaternion):
    """Rotation matrix of a normalized XYZW quaternion."""

    x, y, z, w = np.asarray(quaternion, dtype=float)
    if not np.isfinite([x, y, z, w]).all() or abs(x * x + y * y + z * z + w * w - 1.0) > 2e-5:
        raise ValueError("Quaternion is not normalized XYZW")
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_matrices_xyzw(quaternions):
    """Rotation matrices of a batch of normalized XYZW quaternions."""

    values = np.asarray(quaternions, dtype=float)
    if values.ndim != 2 or values.shape[-1] != 4:
        raise ValueError("A quaternion batch has shape (rows, 4)")
    x, y, z, w = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    if not np.isfinite(values).all() or np.max(np.abs(x * x + y * y + z * z + w * w - 1.0)) > 2e-5:
        raise ValueError("Quaternion is not normalized XYZW")
    matrices = np.empty((len(values), 3, 3))
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


class DiagnosticGeometry:
    """Toe-cap membership and mesh floor clearance of the prepared asset.

    ``meta`` is the prepared ``geometry.json`` and ``clouds`` its
    ``geometry_extrema.npz`` arrays. The mathematics repeats the accepted
    definition: a contact point counts as toe contact when it lies inside the
    tibia cap box grown by the contact offset, and clearance is the lowest mesh
    extremum above the floor plane.
    """

    def __init__(self, meta, clouds, body_names, *, expected_usd_sha256=None):
        self.meta = meta
        self.clouds = {key: np.asarray(value) for key, value in clouds.items()}
        self.body_names = list(body_names)
        if set(self.body_names) != set(meta["body_names"]):
            raise ValueError("The geometry body mapping differs from the articulation")
        if expected_usd_sha256 is not None and meta.get("asset_robot_sha256") != expected_usd_sha256:
            raise ValueError("The diagnostic geometry belongs to another asset")
        self.shapes = {shape["body"]: shape for shape in meta["shapes"]}

    @classmethod
    def load(cls, geometry_path, extrema_path, body_names, *, expected_usd_sha256=None):
        meta = json.loads(Path(geometry_path).read_text())
        with np.load(extrema_path, allow_pickle=False) as data:
            clouds = {key: data[key] for key in data.files}
        return cls(meta, clouds, body_names, expected_usd_sha256=expected_usd_sha256)

    def cap_contains(self, body, point_world, link_pose_xyzw):
        """True when a world contact point lies inside that tibia's toe cap."""

        shape = self.shapes[body]
        transform = np.asarray(shape["shape_to_link"])
        rotation = rotation_matrix_xyzw(np.asarray(link_pose_xyzw)[3:])
        local = (np.asarray(point_world) - np.asarray(link_pose_xyzw)[:3]) @ rotation
        point = (local - transform[:3, 3]) @ transform[:3, :3]
        bounds = np.asarray(shape["cap_bounds_m"])
        skin = shape["contact_offset_m"]
        inside = (point[0] >= shape["cap_lower_x_m"] - 1e-12 and point[0] <= bounds[1, 0] + skin
                  and np.all(point[1:] >= bounds[0, 1:] - skin)
                  and np.all(point[1:] <= bounds[1, 1:] + skin))
        return bool(inside), point

    def cap_contains_batch(self, body, points_world, link_poses):
        """Toe-cap membership for one body across every replica."""

        shape = self.shapes[body]
        transform = np.asarray(shape["shape_to_link"])
        poses = np.asarray(link_poses, dtype=float)
        points = np.asarray(points_world, dtype=float)
        rotations = rotation_matrices_xyzw(poses[:, 3:])
        local = np.einsum("ni,nij->nj", points - poses[:, :3], rotations)
        shape_point = np.einsum("ni,ij->nj", local - transform[:3, 3], transform[:3, :3])
        bounds = np.asarray(shape["cap_bounds_m"])
        skin = shape["contact_offset_m"]
        inside = (
            (shape_point[:, 0] >= shape["cap_lower_x_m"] - 1e-12)
            & (shape_point[:, 0] <= bounds[1, 0] + skin)
            & np.all(shape_point[:, 1:] >= bounds[0, 1:] - skin, axis=-1)
            & np.all(shape_point[:, 1:] <= bounds[1, 1:] + skin, axis=-1)
        )
        return inside, shape_point

    def clearance(self, link_poses):
        """Lowest mesh point above the floor, for all bodies and for non-toes."""

        poses = np.asarray(link_poses, dtype=float)
        count = len(poses)
        all_min = np.full(count, np.inf)
        non_toe_min = np.full(count, np.inf)
        for index, name in enumerate(self.body_names):
            pose = poses[:, index]
            up = rotation_matrices_xyzw(pose[:, 3:])[:, 2, :]
            height = pose[:, 2]
            all_min = np.minimum(all_min, (self.clouds["all__" + name] @ up.T).min(0) + height)
            non_toe_min = np.minimum(non_toe_min, (self.clouds["non_toe__" + name] @ up.T).min(0) + height)
        return all_min, non_toe_min


def capture_row(*, num_envs, sequence, substep_index, explicit_counter, physics_dt_s,
                pre_joint_position_rad, pre_joint_velocity_rad_s, joint_position_rad,
                joint_velocity_rad_s, root_pose_xyzw, link_pose_xyzw, root_com_velocity,
                joint_target_rad, computed_torque_nm, applied_torque_nm, effort_ceiling_nm,
                native_input_pre_nm, native_input_post_nm, distal_contact, distal_force_world_n,
                nonfoot_contact, nonfoot_force_world_n, minimum_mesh_floor_m,
                minimum_non_toe_floor_m, previous_joint_position_rad, terminated, truncated):
    """Assemble one substep row with the shapes the frozen scorer reads."""

    joints = (num_envs, 18)
    row = {
        "pre_joint_position_rad": np.asarray(pre_joint_position_rad, dtype=np.float32),
        "pre_joint_velocity_rad_s": np.asarray(pre_joint_velocity_rad_s, dtype=np.float32),
        "joint_position_rad": np.asarray(joint_position_rad, dtype=np.float32),
        "joint_velocity_rad_s": np.asarray(joint_velocity_rad_s, dtype=np.float32),
        "root_pose_xyzw": np.asarray(root_pose_xyzw, dtype=np.float32),
        "link_pose_xyzw": np.asarray(link_pose_xyzw, dtype=np.float32),
        "root_com_velocity": np.asarray(root_com_velocity, dtype=np.float32),
        "joint_target_rad": np.asarray(joint_target_rad, dtype=np.float32),
        "computed_torque_nm": np.asarray(computed_torque_nm, dtype=np.float32),
        "applied_torque_nm": np.asarray(applied_torque_nm, dtype=np.float32),
        "effort_ceiling_nm": np.asarray(effort_ceiling_nm, dtype=np.float32),
        "native_input_pre_nm": np.asarray(native_input_pre_nm, dtype=np.float32),
        "native_input_post_nm": np.asarray(native_input_post_nm, dtype=np.float32),
        "sequence": np.array(sequence),
        "control_index": np.array(sequence // 8),
        "substep_index": np.array(substep_index),
        "explicit_counter": np.array(explicit_counter),
        "time_s": np.array((sequence + 1) * physics_dt_s),
        "contact_valid": np.ones(num_envs, bool),
        "interval_valid": np.ones(num_envs, bool),
        "truncated": np.asarray(truncated, dtype=bool),
        "distal_contact": np.asarray(distal_contact, dtype=bool),
        "distal_force_world_n": np.asarray(distal_force_world_n, dtype=float),
        "nonfoot_contact": np.asarray(nonfoot_contact, dtype=bool),
        "nonfoot_force_world_n": np.asarray(nonfoot_force_world_n, dtype=float),
        "minimum_mesh_floor_m": np.asarray(minimum_mesh_floor_m, dtype=float),
        "minimum_non_toe_floor_m": np.asarray(minimum_non_toe_floor_m, dtype=float),
        "interval_angle_rate_rad_s": (
            np.asarray(joint_position_rad, dtype=np.float32).astype(float)
            - np.asarray(previous_joint_position_rad, dtype=float)
        ) / physics_dt_s,
        "terminated": np.asarray(terminated, dtype=bool),
    }
    expected = {
        "pre_joint_position_rad": joints, "pre_joint_velocity_rad_s": joints,
        "joint_position_rad": joints, "joint_velocity_rad_s": joints,
        "joint_target_rad": joints, "computed_torque_nm": joints, "applied_torque_nm": joints,
        "effort_ceiling_nm": joints, "native_input_pre_nm": joints, "native_input_post_nm": joints,
        "interval_angle_rate_rad_s": joints,
        "root_pose_xyzw": (num_envs, 7), "link_pose_xyzw": (num_envs, 19, 7),
        "root_com_velocity": (num_envs, 6),
        "contact_valid": (num_envs,), "interval_valid": (num_envs,), "truncated": (num_envs,),
        "terminated": (num_envs,), "nonfoot_contact": (num_envs,),
        "minimum_mesh_floor_m": (num_envs,), "minimum_non_toe_floor_m": (num_envs,),
        "distal_contact": (num_envs, 6), "distal_force_world_n": (num_envs, 6, 3),
        "nonfoot_force_world_n": (num_envs, len(CONTACT_CATEGORIES), 3),
    }
    if set(row) != set(CAPTURE_ROW_KEYS):
        raise ValueError("The capture row keys differ from the declared layout")
    for key, shape in expected.items():
        if row[key].shape != shape:
            raise ValueError(f"Capture channel {key} has shape {row[key].shape}, expected {shape}")
    return {key: row[key] for key in CAPTURE_ROW_KEYS}


class Rs05DiagnosticCapture:
    """Write the substep, control and session files of one capture."""

    def __init__(self, output, *, joint_names, body_names, root_paths, neutral_joint_position_rad,
                 native_readback, initial_state, physics_dt_s, strict=True):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.root_paths = list(root_paths)
        self.neutral = [float(value) for value in neutral_joint_position_rad]
        self.physics_dt_s = float(physics_dt_s)
        self.strict = bool(strict)
        self.rows, self.controls, self.files = [], [], []
        self.count = 0
        self.failure = None
        self.counter = int(initial_state["counter_after"])
        self.contacts = (self.output / "contacts.jsonl").open("w")
        self.save("native_readback.json", native_readback)
        self.save("initial_reset.json", initial_state)

    def save(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def record(self, row, *, patches=None):
        """Store one substep row and keep the evidence of a failed one."""

        try:
            self.rows.append(row)
            if int(row["substep_index"]) == 7:
                self.controls.append(row)
            self.contacts.write(json.dumps(
                {"sequence": int(row["sequence"]), "explicit_counter": int(row["explicit_counter"]),
                 "patches": patches or []}, allow_nan=False) + "\n")
            self.count += 1
            if int(row["explicit_counter"]) != self.counter + 1:
                raise RuntimeError("The diagnostic physics counter differs")
            self.counter += 1
            if not all(np.isfinite(value).all() for value in row.values()):
                raise ValueError("Nonfinite diagnostic sample")
            if not np.array_equal(row["native_input_pre_nm"], row["applied_torque_nm"]):
                raise ValueError("The native effort input differs from the applied effort")
            if self.strict and (np.max(abs(row["applied_torque_nm"])) > MAX_APPLIED_NM
                                or bool(row["terminated"].any())):
                raise ValueError("The cap, plate, joint, nonfoot or clearance gate failed")
            if len(self.rows) >= ROWS_PER_FILE:
                self.flush()
        except BaseException as error:
            self.failure = repr(error)
            self.save("failed_partial_step.json", {key: np.asarray(value).tolist() for key, value in row.items()})
            self.flush()
            raise

    def flush(self):
        if self.rows:
            name = f"substeps_{len(self.files):03d}.npz"
            np.savez_compressed(
                self.output / name,
                **{key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]},
            )
            self.files.append(name)
            self.rows = []
        self.contacts.flush()

    def close(self, *, scope):
        self.flush()
        self.contacts.close()
        if self.controls:
            np.savez_compressed(
                self.output / "control_trace.npz",
                **{key: np.stack([row[key] for row in self.controls]) for key in self.controls[0]},
            )
        self.save("session.json", {
            "steps": self.count, "controls": len(self.controls), "reset_count": 1,
            "failure": self.failure, "substep_files": self.files, "joint_names": self.joint_names,
            "body_names": self.body_names, "root_paths": self.root_paths,
            "all_rows_recorded": self.count == len(self.controls) * 8,
            "contact_evidence_complete": self.failure is None,
            "neutral_joint_position_rad": self.neutral,
            "physical_admission": False, "scope": scope,
        })
