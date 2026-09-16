"""The RS05 capture layout against the frozen admission scorer.

``experiments/paper_walk/env.py:score_diagnostic`` is unchanged. This file
writes a synthetic capture with the new writer and lets that scorer read it, so
the key names, shapes, counters and servo recurrence are compared with the real
consumer instead of a copy of it.

The synthetic capture holds one replica at its neutral stance with no motion.
It is a layout fixture. It measures no robot, and it establishes no standing
admission: a real capture comes from the simulator on the Spark host.

The geometry helpers are compared with the frozen ``_DiagnosticGeometry`` on the
prepared geometry of this exact asset.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "packages/hexapod_env"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from experiments.paper_walk.env import _DiagnosticGeometry, score_diagnostic  # noqa: E402
from experiments.paper_walk.env_config import JOINT_NAMES  # noqa: E402
from hexapod_env.actuators.rs05_paper_walk_model import damping_vector, motor_effort  # noqa: E402
from hexapod_env.assets.mkii_rs05 import (  # noqa: E402
    MKII_RS05_ASSET,
    MKII_RS05_USD_SHA256,
)
from hexapod_env.tasks.mkii_rs05.capture import (  # noqa: E402
    CAPTURE_ROW_KEYS,
    CONTACT_CATEGORIES,
    DiagnosticGeometry,
    Rs05DiagnosticCapture,
    capture_row,
    rotation_matrices_xyzw,
    rotation_matrix_xyzw,
)

GEOMETRY_DIR = ROOT / "artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry"
BODY_NAMES = (MKII_RS05_ASSET.root_link,) + tuple(
    name for group in MKII_RS05_ASSET.leg_link_names for name in group
)
CONTROLS = 1000
SUBSTEPS = 8


def neutral_float32():
    stance = MKII_RS05_ASSET.default_joint_positions()
    return np.asarray([stance[name] for name in JOINT_NAMES], dtype=np.float32)


def write_synthetic_capture(directory):
    """One replica holding its neutral stance for 1000 control steps."""

    neutral = neutral_float32()
    joint_pos = neutral[None].copy()
    joint_vel = np.zeros_like(joint_pos)
    requested, applied, ceiling = motor_effort(
        torch.from_numpy(joint_pos), torch.from_numpy(joint_vel),
        torch.from_numpy(joint_pos), damping_vector(JOINT_NAMES),
    )
    limits = MKII_RS05_ASSET.joint_limits()
    root_pose = np.asarray([[0.0, 0.0, 0.0978, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    link_pose = np.zeros((1, 19, 7), dtype=np.float32)
    link_pose[:, :, 2] = 0.05
    link_pose[:, :, 6] = 1.0
    native_readback = {
        "limits": [[[limits[name][0], limits[name][1]] for name in JOINT_NAMES]],
        "native_max_velocity": [[50.26548385620117] * 18],
    }
    initial_state = {
        "counter_after": 0,
        "post_reset": {
            "joint_position_rad": joint_pos.tolist(),
            "joint_velocity_rad_s": joint_vel.tolist(),
            "root_pose_xyzw": root_pose.tolist(),
        },
        "scope": "Synthetic constant state; it implies no admission.",
    }
    capture = Rs05DiagnosticCapture(
        directory,
        joint_names=list(JOINT_NAMES),
        body_names=list(BODY_NAMES),
        root_paths=["/World/envs/env_0"],
        neutral_joint_position_rad=neutral.tolist(),
        native_readback=native_readback,
        initial_state=initial_state,
        physics_dt_s=0.0025,
    )
    for sequence in range(CONTROLS * SUBSTEPS):
        row = capture_row(
            num_envs=1,
            sequence=sequence,
            substep_index=sequence % SUBSTEPS,
            explicit_counter=sequence + 1,
            physics_dt_s=0.0025,
            pre_joint_position_rad=joint_pos,
            pre_joint_velocity_rad_s=joint_vel,
            joint_position_rad=joint_pos,
            joint_velocity_rad_s=joint_vel,
            root_pose_xyzw=root_pose,
            link_pose_xyzw=link_pose,
            root_com_velocity=np.zeros((1, 6), dtype=np.float32),
            joint_target_rad=joint_pos,
            computed_torque_nm=requested.numpy(),
            applied_torque_nm=applied.numpy(),
            effort_ceiling_nm=ceiling.numpy(),
            native_input_pre_nm=applied.numpy(),
            native_input_post_nm=applied.numpy(),
            distal_contact=np.ones((1, 6), dtype=bool),
            distal_force_world_n=np.tile([0.0, 0.0, 12.2], (1, 6, 1)),
            nonfoot_contact=np.zeros(1, dtype=bool),
            nonfoot_force_world_n=np.zeros((1, len(CONTACT_CATEGORIES), 3)),
            minimum_mesh_floor_m=np.zeros(1),
            minimum_non_toe_floor_m=np.full(1, 0.02),
            previous_joint_position_rad=joint_pos.astype(float),
            terminated=np.zeros(1, dtype=bool),
            truncated=np.zeros(1, dtype=bool),
        )
        capture.record(row)
    capture.close(scope="Synthetic layout fixture; the scorer recomputes every channel.")
    return capture


class ScorerLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.capture = write_synthetic_capture(Path(cls.directory.name) / "standing")
        cls.result = score_diagnostic(Path(cls.directory.name) / "standing")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_the_frozen_scorer_reads_the_written_layout(self):
        self.assertEqual(self.result["controls"], CONTROLS)
        self.assertEqual(self.result["substeps"], CONTROLS * SUBSTEPS)
        self.assertEqual(self.result["num_envs"], 1)
        replica = self.result["replicas"][0]
        self.assertEqual(replica["failed_physical_bounds"], [])
        self.assertEqual(replica["quiet"]["failed_bounds"], [])
        self.assertIs(self.result["all_pass"], True)

    def test_the_session_records_the_canonical_order_and_no_admission(self):
        session = json.loads((Path(self.directory.name) / "standing/session.json").read_text())
        self.assertEqual(session["joint_names"], list(JOINT_NAMES))
        self.assertEqual(session["body_names"], list(BODY_NAMES))
        self.assertIs(session["physical_admission"], False)
        self.assertIsNone(session["failure"])
        self.assertIs(session["all_rows_recorded"], True)
        self.assertEqual(len(session["substep_files"]), 10)

    def test_every_substep_file_carries_the_declared_keys(self):
        session = json.loads((Path(self.directory.name) / "standing/session.json").read_text())
        for name in session["substep_files"]:
            with np.load(Path(self.directory.name) / "standing" / name) as data:
                self.assertEqual(set(data.files), set(CAPTURE_ROW_KEYS))
                self.assertEqual(data["joint_position_rad"].shape, (800, 1, 18))
                self.assertEqual(data["link_pose_xyzw"].shape, (800, 1, 19, 7))

    def test_a_wrong_channel_shape_is_refused(self):
        with self.assertRaisesRegex(ValueError, "distal_contact"):
            capture_row(
                num_envs=1, sequence=0, substep_index=0, explicit_counter=1, physics_dt_s=0.0025,
                pre_joint_position_rad=np.zeros((1, 18)), pre_joint_velocity_rad_s=np.zeros((1, 18)),
                joint_position_rad=np.zeros((1, 18)), joint_velocity_rad_s=np.zeros((1, 18)),
                root_pose_xyzw=np.zeros((1, 7)), link_pose_xyzw=np.zeros((1, 19, 7)),
                root_com_velocity=np.zeros((1, 6)), joint_target_rad=np.zeros((1, 18)),
                computed_torque_nm=np.zeros((1, 18)), applied_torque_nm=np.zeros((1, 18)),
                effort_ceiling_nm=np.zeros((1, 18)), native_input_pre_nm=np.zeros((1, 18)),
                native_input_post_nm=np.zeros((1, 18)), distal_contact=np.zeros((1, 5), dtype=bool),
                distal_force_world_n=np.zeros((1, 6, 3)), nonfoot_contact=np.zeros(1, dtype=bool),
                nonfoot_force_world_n=np.zeros((1, 4, 3)), minimum_mesh_floor_m=np.zeros(1),
                minimum_non_toe_floor_m=np.zeros(1), previous_joint_position_rad=np.zeros((1, 18)),
                terminated=np.zeros(1, dtype=bool), truncated=np.zeros(1, dtype=bool),
            )


class CaptureGateTests(unittest.TestCase):
    def row(self, **changes):
        neutral = neutral_float32()[None]
        values = dict(
            num_envs=1, sequence=0, substep_index=0, explicit_counter=1, physics_dt_s=0.0025,
            pre_joint_position_rad=neutral, pre_joint_velocity_rad_s=np.zeros((1, 18)),
            joint_position_rad=neutral, joint_velocity_rad_s=np.zeros((1, 18)),
            root_pose_xyzw=np.asarray([[0.0, 0.0, 0.0978, 0.0, 0.0, 0.0, 1.0]]),
            link_pose_xyzw=np.zeros((1, 19, 7)), root_com_velocity=np.zeros((1, 6)),
            joint_target_rad=neutral, computed_torque_nm=np.zeros((1, 18)),
            applied_torque_nm=np.zeros((1, 18)), effort_ceiling_nm=np.full((1, 18), 1.6),
            native_input_pre_nm=np.zeros((1, 18)), native_input_post_nm=np.zeros((1, 18)),
            distal_contact=np.ones((1, 6), dtype=bool), distal_force_world_n=np.zeros((1, 6, 3)),
            nonfoot_contact=np.zeros(1, dtype=bool), nonfoot_force_world_n=np.zeros((1, 4, 3)),
            minimum_mesh_floor_m=np.zeros(1), minimum_non_toe_floor_m=np.full(1, 0.02),
            previous_joint_position_rad=neutral.astype(float),
            terminated=np.zeros(1, dtype=bool), truncated=np.zeros(1, dtype=bool),
        )
        values.update(changes)
        return capture_row(**values)

    def capture(self, directory):
        return Rs05DiagnosticCapture(
            Path(directory) / "standing", joint_names=list(JOINT_NAMES), body_names=list(BODY_NAMES),
            root_paths=["/World/envs/env_0"], neutral_joint_position_rad=neutral_float32().tolist(),
            native_readback={}, initial_state={"counter_after": 0}, physics_dt_s=0.0025,
        )

    def test_an_over_cap_effort_aborts_and_keeps_the_partial_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.capture(directory)
            with self.assertRaisesRegex(ValueError, "gate failed"):
                capture.record(self.row(applied_torque_nm=np.full((1, 18), 1.7),
                                        native_input_pre_nm=np.full((1, 18), 1.7)))
            self.assertTrue((Path(directory) / "standing/failed_partial_step.json").is_file())
            capture.close(scope="Aborted fixture")
            session = json.loads((Path(directory) / "standing/session.json").read_text())
            self.assertIn("gate failed", session["failure"])
            self.assertIs(session["contact_evidence_complete"], False)

    def test_a_native_effort_mismatch_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.capture(directory)
            with self.assertRaisesRegex(ValueError, "native effort input"):
                capture.record(self.row(native_input_pre_nm=np.full((1, 18), 0.5)))
            capture.close(scope="Aborted fixture")

    def test_a_terminated_row_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.capture(directory)
            with self.assertRaisesRegex(ValueError, "gate failed"):
                capture.record(self.row(terminated=np.ones(1, dtype=bool)))
            capture.close(scope="Aborted fixture")

    def test_a_counter_gap_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.capture(directory)
            with self.assertRaisesRegex(RuntimeError, "counter"):
                capture.record(self.row(explicit_counter=5))
            capture.close(scope="Aborted fixture")


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        meta = json.loads((GEOMETRY_DIR / "geometry.json").read_text())
        with np.load(GEOMETRY_DIR / "geometry_extrema.npz", allow_pickle=False) as data:
            clouds = {key: data[key] for key in data.files}
        cls.frozen = _DiagnosticGeometry(meta, clouds, list(BODY_NAMES))
        cls.mine = DiagnosticGeometry(meta, clouds, list(BODY_NAMES),
                                      expected_usd_sha256=MKII_RS05_USD_SHA256)
        generator = np.random.default_rng(31)
        poses = np.zeros((7, 19, 7))
        poses[:, :, :3] = generator.uniform(-0.2, 0.2, (7, 19, 3))
        poses[:, :, 2] = generator.uniform(0.02, 0.2, (7, 19))
        quaternion = generator.normal(size=(7, 19, 4))
        poses[:, :, 3:] = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
        cls.poses = poses

    def test_the_geometry_belongs_to_the_selected_asset(self):
        self.assertEqual(self.mine.meta["asset_robot_sha256"], MKII_RS05_USD_SHA256)
        meta = dict(self.mine.meta, asset_robot_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "another asset"):
            DiagnosticGeometry(meta, self.mine.clouds, list(BODY_NAMES),
                               expected_usd_sha256=MKII_RS05_USD_SHA256)

    def test_clearance_matches_the_frozen_geometry(self):
        expected = self.frozen.clearance(self.poses)
        actual = self.mine.clearance(self.poses)
        for left, right in zip(actual, expected):
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-12)

    def test_the_toe_cap_test_matches_the_frozen_geometry(self):
        generator = np.random.default_rng(5)
        points = generator.uniform(-0.3, 0.3, (7, 3))
        for body in ("lf_tibia", "rm_tibia"):
            poses = self.poses[:, BODY_NAMES.index(body)]
            expected = [self.frozen.cap(body, points[row], poses[row]) for row in range(len(points))]
            inside, shape_point = self.mine.cap_contains_batch(body, points, poses)
            np.testing.assert_array_equal(inside, [value for value, _ in expected])
            np.testing.assert_allclose(
                shape_point, np.stack([point for _, point in expected]), rtol=0, atol=1e-12
            )

    def test_the_batch_rotation_matches_the_single_rotation(self):
        quaternions = self.poses[:, 0, 3:]
        expected = np.stack([rotation_matrix_xyzw(row) for row in quaternions])
        np.testing.assert_allclose(rotation_matrices_xyzw(quaternions), expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            rotation_matrices_xyzw(np.zeros((3, 4)))


if __name__ == "__main__":
    unittest.main()
