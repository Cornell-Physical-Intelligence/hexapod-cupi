"""Check the Table II declaration without a simulator: hash, bounds, determinism, isolation and the helpers."""

import json
import math
import unittest

import numpy as np
import torch

from locomotion import randomization as rnd
from locomotion.robots.hexapod import JOINT_NAMES, KD, MASS_KG
from locomotion.robots.rs05 import motor_force

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
MODEL = ROOT / "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json"
STANCE = ROOT / "robot/hexapod_mkii_updated_v1/stance.json"


def limits():
    joints = {j["name"]: j for j in json.loads(MODEL.read_text())["joints"]}
    return (
        np.array([joints[n]["lower"] for n in JOINT_NAMES]),
        np.array([joints[n]["upper"] for n in JOINT_NAMES]),
    )


class DeclarationTests(unittest.TestCase):
    def test_canonical_json_is_sorted_and_hashes_stably(self):
        for name in rnd.NAMES:
            record = rnd.declaration(name)
            value = json.loads(record["json"])
            self.assertEqual(record["json"], rnd.canonical(value))
            self.assertEqual(rnd.declaration(name)["sha256"], record["sha256"])
            self.assertEqual((value["schema"], value["name"]), (rnd.SCHEMA, name))
            self.assertEqual(value["base"]["kd"], list(KD))
            self.assertEqual(value["base"]["mass_kg"], MASS_KG)
        self.assertNotEqual(
            rnd.declaration("none")["sha256"], rnd.declaration("table2_v1")["sha256"]
        )
        with self.assertRaises(ValueError):
            rnd.declaration("table2_v2")

    def test_every_row_records_a_status_range_granularity_and_reason(self):
        table = json.loads(rnd.declaration("table2_v1")["json"])["table"]
        rows = (
            "joint_stiffness",
            "joint_damping",
            "joint_position",
            "link_mass",
            "payload_mass",
            "payload_position",
            "foot_friction",
            "motor_strength",
        )
        for row in rows:
            self.assertIn(table[row]["status"], ("kept", "adapted", "excluded"), row)
            self.assertIn(table[row]["granularity"], ("shared", "per_joint", "per_link"), row)
            self.assertTrue(table[row]["paper"] and table[row]["reason"], row)
            if row != "joint_position":
                self.assertLess(table[row]["low"], table[row]["high"], row)
        self.assertEqual(table["joint_position"]["status"], "adapted")
        self.assertEqual(table["push"]["status"], "adapted")
        self.assertAlmostEqual(table["payload_mass"]["high"], MASS_KG * 5 / 25.5)
        self.assertEqual(
            rnd.identity("table2_v1")["declaration_sha256"], rnd.declaration("table2_v1")["sha256"]
        )

    def test_nominal_pose_matches_the_stance_file(self):
        stance = json.loads(STANCE.read_text())["nominal_joint_position_rad"]
        expected = [
            rnd.NOMINAL_POSE_RAD[rnd.JOINT_GROUPS.index(n.split("_", 1)[1])] for n in JOINT_NAMES
        ]
        self.assertEqual(stance, expected)

    def test_training_guide_names_the_current_hash(self):
        digest = rnd.declaration("table2_v1")["sha256"]
        guide = (ROOT / "src/locomotion/README.md").read_text()
        self.assertIn(digest, guide)
        self.assertIn("## Step 7: domain randomization declaration", guide)
        self.assertIn(digest, (ROOT / "docs/TRAINING.md").read_text())

    def test_kp_matches_the_motor_law(self):
        q, dq = torch.zeros(1, 18), torch.zeros(1, 18)
        requested, _, _ = motor_force(q, dq, torch.full((1, 18), 0.01), torch.tensor(KD))
        self.assertTrue(torch.allclose(requested, torch.full((1, 18), 0.01 * rnd.KP)))


class DrawTests(unittest.TestCase):
    def test_none_draws_the_nominal_properties_and_no_push(self):
        generator = rnd.replica_generator(20260917, 3, "none")
        before = generator.bit_generator.state
        d = rnd.draw("none", generator)
        self.assertEqual(
            (d.kp_scale, d.kd_scale, d.payload_mass_kg, d.foot_friction), (1.0, 1.0, 0.0, 1.0)
        )
        self.assertTrue(
            np.all(d.joint_offset_rad == 0)
            and np.all(d.link_mass_scale == 1)
            and np.all(d.motor_strength == 1)
        )
        self.assertIsNone(rnd.next_push("none", generator, first=True))
        self.assertEqual(generator.bit_generator.state, before)

    def test_draws_stay_inside_the_declared_ranges_and_the_joint_limits(self):
        lower, upper = limits()
        t = rnd.TABLE2_V1
        half = np.array(
            [
                t.joint_position.half_width_rad[rnd.JOINT_GROUPS.index(n.split("_", 1)[1])]
                for n in JOINT_NAMES
            ]
        )
        neutral = np.array(
            [rnd.NOMINAL_POSE_RAD[rnd.JOINT_GROUPS.index(n.split("_", 1)[1])] for n in JOINT_NAMES]
        )
        for replica in range(64):
            generator = rnd.replica_generator(20260917, replica, "table2_v1")
            d = rnd.draw("table2_v1", generator)
            self.assertTrue(t.joint_stiffness.low <= d.kp_scale <= t.joint_stiffness.high)
            self.assertTrue(t.joint_damping.low <= d.kd_scale <= t.joint_damping.high)
            self.assertTrue(np.all(np.abs(d.joint_offset_rad) <= half))
            self.assertEqual(d.link_mass_scale.shape, (19,))
            self.assertTrue(np.all((d.link_mass_scale >= 0.9) & (d.link_mass_scale <= 1.1)))
            self.assertTrue(0.0 <= d.payload_mass_kg <= rnd.PAYLOAD_MAX_KG)
            self.assertTrue(
                np.all(np.abs(d.payload_position_m[:2]) <= 0.15) and d.payload_position_m[2] == 0.04
            )
            self.assertTrue(0.1 <= d.foot_friction <= 2.5)
            self.assertTrue(
                np.all((d.motor_strength >= 0.8) & (d.motor_strength <= 1.2))
                and d.motor_strength.shape == (18,)
            )
            pose = rnd.reset_pose(neutral, d.joint_offset_rad, lower, upper)
            self.assertTrue(np.all(pose >= lower) and np.all(pose <= upper))
        # The extreme offsets also stay inside the limits, so the draw never needs clipping.
        for sign in (-1.0, 1.0):
            rnd.reset_pose(neutral, sign * half, lower, upper)
        with self.assertRaises(ValueError):
            rnd.reset_pose(neutral, np.full(18, 3.0), lower, upper)

    def test_pushes_follow_the_schedule(self):
        p = rnd.TABLE2_V1.push
        generator = rnd.replica_generator(20260917, 0, "table2_v1")
        rnd.draw("table2_v1", generator)
        first = rnd.next_push("table2_v1", generator, first=True)
        later = [rnd.next_push("table2_v1", generator, first=False) for _ in range(50)]
        self.assertTrue(p.first_delay_s[0] <= first.delay_s <= p.first_delay_s[1])
        for push in later:
            self.assertTrue(p.interval_s[0] <= push.delay_s <= p.interval_s[1])
            self.assertEqual(push.duration_s, p.duration_s)
            self.assertEqual(push.force_body_n[2], 0.0)
            self.assertLessEqual(float(np.linalg.norm(push.force_body_n)), p.magnitude_n[1] + 1e-12)
        self.assertEqual(round(p.duration_s / rnd.CONTROL_DT_S), 10)

    def test_generators_are_deterministic_and_isolated(self):
        a = rnd.draw("table2_v1", rnd.replica_generator(20260917, 5, "table2_v1"))
        b = rnd.draw("table2_v1", rnd.replica_generator(20260917, 5, "table2_v1"))
        self.assertEqual(a.as_json(), b.as_json())
        other = rnd.draw("table2_v1", rnd.replica_generator(20260917, 6, "table2_v1"))
        seed = rnd.draw("table2_v1", rnd.replica_generator(20260918, 5, "table2_v1"))
        self.assertNotEqual(a.as_json(), other.as_json())
        self.assertNotEqual(a.as_json(), seed.as_json())
        # A replica's generator is independent of torch's global state and of the other replicas.
        torch.manual_seed(1)
        generators = [rnd.replica_generator(20260917, i, "table2_v1") for i in range(3)]
        rnd.draw("table2_v1", generators[1])
        self.assertEqual(
            rnd.draw("table2_v1", generators[0]).as_json(),
            rnd.draw("table2_v1", rnd.replica_generator(20260917, 0, "table2_v1")).as_json(),
        )
        self.assertEqual(json.loads(rnd.canonical(a.as_json())), a.as_json())


class HelperTests(unittest.TestCase):
    def test_scaled_link_keeps_the_shape(self):
        inertia = np.diag([1.0, 2.0, 3.0])
        mass, tensor = rnd.scaled_link(2.0, inertia, 1.1)
        self.assertAlmostEqual(mass, 2.2)
        self.assertTrue(np.allclose(tensor, 1.1 * inertia))

    def test_payload_combination_follows_the_parallel_axis_theorem(self):
        base = json.loads(MODEL.read_text())["links"][0]
        mass, com, inertia = base["mass"], np.array(base["com"]), np.array(base["inertia"])
        total, centre, tensor = rnd.combined_base_inertial(
            mass, com, inertia, 0.0, [0.1, 0.1, 0.04]
        )
        self.assertEqual(total, mass)
        self.assertTrue(np.allclose(centre, com) and np.allclose(tensor, inertia))
        payload, position = rnd.PAYLOAD_MAX_KG, np.array([0.15, -0.15, 0.04])
        total, centre, tensor = rnd.combined_base_inertial(mass, com, inertia, payload, position)
        self.assertAlmostEqual(total, mass + payload)
        self.assertTrue(np.allclose(centre, (mass * com + payload * position) / total))
        # The tensor about the new centre: move the base tensor there, then add the point mass.
        d_base, d_payload = com - centre, position - centre
        expected = (
            inertia
            + mass * (d_base @ d_base * np.eye(3) - np.outer(d_base, d_base))
            + payload * (d_payload @ d_payload * np.eye(3) - np.outer(d_payload, d_payload))
        )
        self.assertTrue(np.allclose(tensor, expected))
        self.assertTrue(np.allclose(tensor, tensor.T) and np.all(np.linalg.eigvalsh(tensor) > 0))

    def test_strength_scales_the_curve_under_the_fixed_cap(self):
        rpm = torch.tensor([0.0, 300.0, 450.0, 465.0, 477.0], dtype=torch.float64)
        dq = (rpm * 2 * math.pi / 60).to(torch.float32)[None]
        _, _, ceiling = motor_force(torch.zeros(1, 5), dq, torch.zeros(1, 5), torch.zeros(5))
        strong = rnd.strength_ceiling(ceiling, torch.full((1, 5), 1.2))
        weak = rnd.strength_ceiling(ceiling, torch.full((1, 5), 0.8))
        self.assertTrue(torch.all(strong <= rnd.TORQUE_CAP_NM))
        # Below the cap region the factor above one changes nothing; the factor below one lowers it everywhere.
        self.assertTrue(torch.allclose(strong[0, :3], ceiling[0, :3]))
        self.assertGreater(float(strong[0, 4]), float(ceiling[0, 4]))
        self.assertTrue(torch.allclose(weak, 0.8 * ceiling))
        self.assertTrue(
            np.allclose(
                rnd.strength_ceiling(np.array([5.0, 1.0]), np.array([1.2, 1.2])), [1.6, 1.2]
            )
        )

    def test_wrench_mean_reads_eight_substeps(self):
        forces = np.zeros((8, 3))
        forces[:4, 0] = 15.0
        wrench = rnd.wrench_mean(forces, np.zeros((8, 3)))
        self.assertTrue(np.allclose(wrench, [7.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
        with self.assertRaises(ValueError):
            rnd.wrench_mean(np.zeros((7, 3)), np.zeros((8, 3)))


if __name__ == "__main__":
    unittest.main()
