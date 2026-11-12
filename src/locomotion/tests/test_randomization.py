"""Check the Table II declaration without a simulator: hash, bounds, determinism, isolation and the helpers."""

import json
import math
import unittest

import numpy as np
import torch

from locomotion.robots import randomization as rnd
from locomotion.robots.hexapod import BODY_NAMES, JOINT_NAMES, KD, LEGS, MASS_KG
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

    def test_table2_v1_and_its_grid_are_frozen(self):
        # Frozen 2026-11-08 under #67. A range or text change takes a new declaration name.
        self.assertEqual(
            rnd.declaration("table2_v1")["sha256"],
            "730cad982ddac064cdd36090fd3d770a1b38a6f81aa03bba0ce792bba84b6a5c",
        )
        self.assertEqual(
            rnd.grid_declaration("table2_v1_grid")["sha256"],
            "7d631f12e8abefd8141f0dc515eaa446b90b5563315fe41ea608db0ab2822882",
        )

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


class GridTests(unittest.TestCase):
    def test_the_grid_crosses_the_issue_values_in_a_fixed_order(self):
        cells = rnd.grid_cells("table2_v1_grid")
        self.assertEqual(len(cells), 27)
        self.assertEqual(len({cell for cell, _ in cells}), 27)
        values = [value for _, value in cells]
        self.assertEqual(sorted({v[0] for v in values}), [0.1, 1.0, 2.5])
        self.assertEqual(sorted({v[2] for v in values}), [0.8, 1.0, 1.2])
        payloads = sorted({v[1] for v in values})
        self.assertEqual(payloads, [0.0, rnd.PAYLOAD_MAX_KG / 2, rnd.PAYLOAD_MAX_KG])
        self.assertAlmostEqual(payloads[-1], MASS_KG * 5 / 25.5)
        self.assertEqual(values[0], (0.1, 0.0, 0.8))
        self.assertEqual(values[-1], (2.5, rnd.PAYLOAD_MAX_KG, 1.2))

    def test_a_cell_changes_its_three_conditions_and_nothing_else(self):
        draw = rnd.cell_draw("table2_v1_grid", "mu0.1_payload1.464kg_strength0.8")
        nominal = rnd.nominal_draw()
        self.assertEqual((draw.foot_friction, draw.payload_mass_kg), (0.1, rnd.PAYLOAD_MAX_KG))
        self.assertTrue(np.array_equal(draw.motor_strength, np.full(18, 0.8)))
        self.assertTrue(np.array_equal(draw.payload_position_m, [0.0, 0.0, 0.04]))
        self.assertEqual((draw.kp_scale, draw.kd_scale), (1.0, 1.0))
        self.assertTrue(np.array_equal(draw.joint_offset_rad, nominal.joint_offset_rad))
        self.assertTrue(np.array_equal(draw.link_mass_scale, nominal.link_mass_scale))
        middle = rnd.cell_draw("table2_v1_grid", "mu1_payload0.000kg_strength1")
        self.assertEqual(middle.as_json(), nominal.as_json())
        with self.assertRaises(ValueError):
            rnd.cell_draw("table2_v1_grid", "mu0.5_payload0.000kg_strength1")

    def test_the_grid_hash_binds_the_screened_declaration_and_leaves_it_unchanged(self):
        grid = rnd.grid_declaration("table2_v1_grid")
        value = json.loads(grid["json"])
        self.assertEqual(value["randomization_sha256"], rnd.declaration("table2_v1")["sha256"])
        self.assertEqual(
            rnd.declaration("table2_v1")["sha256"],
            "730cad982ddac064cdd36090fd3d770a1b38a6f81aa03bba0ce792bba84b6a5c",
        )
        self.assertFalse(value["grid"]["pushes"])
        self.assertEqual(len(value["cells"]), 27)
        guide = (ROOT / "src/locomotion/README.md").read_text()
        self.assertIn(grid["sha256"], guide)


class ProbeMotorLawTests(unittest.TestCase):
    """The probe runner's exact torque check follows the applied draw, not only the nominal law."""

    def batch(self, strength):
        generator = torch.Generator().manual_seed(5)
        q = 0.4 * torch.randn(64, 18, generator=generator)
        # Joint speeds from rest through the 450-480 rpm tail, where strength 1.2 lifts the ceiling.
        dq = torch.linspace(0.0, 500.0, 64 * 18).reshape(64, 18) * (2 * math.pi / 60)
        dq = dq * torch.where(torch.rand(64, 18, generator=generator) < 0.5, -1.0, 1.0)
        target = q + 0.3 * torch.randn(64, 18, generator=generator)
        kp = rnd.KP * (0.8 + 0.4 * torch.rand(64, 1, generator=generator))
        kd = torch.tensor(KD) * (0.8 + 0.4 * torch.rand(64, 1, generator=generator))
        return q, dq, target, kp, kd, torch.full((64, 18), strength)

    def test_the_expected_law_reproduces_the_randomized_motor_bit_for_bit(self):
        from locomotion.evaluation.runner import expected_motor_law

        for strength in (0.8, 1.0, 1.2):
            q, dq, target, kp, kd, factor = self.batch(strength)
            torques = rnd.motor_force(q, dq, target, kd, kp, factor)
            law = {
                "kp": kp.expand(64, 18).numpy(),
                "kd": kd.numpy(),
                "strength": factor.numpy(),
            }
            wanted = expected_motor_law(q.numpy(), dq.numpy(), target.numpy(), law)
            for name, actual, expected in zip(("requested", "applied", "ceiling"), torques, wanted):
                with self.subTest(strength=strength, name=name):
                    self.assertTrue(np.array_equal(actual.numpy(), expected))
                    self.assertEqual(expected.dtype, np.float32)
            self.assertLessEqual(float(wanted[2].max()), np.float32(rnd.TORQUE_CAP_NM))

    def test_the_nominal_law_still_rejects_a_scaled_ceiling(self):
        from locomotion.evaluation.runner import expected_motor_law

        for strength in (0.8, 1.2):
            q, dq, target, _, _, factor = self.batch(strength)
            kp = torch.full((64, 18), rnd.KP)
            kd = torch.tensor(KD).expand(64, 18)
            _, _, ceiling = rnd.motor_force(q, dq, target, kd, kp, factor)
            nominal = expected_motor_law(q.numpy(), dq.numpy(), target.numpy())
            with self.subTest(strength=strength):
                self.assertFalse(np.array_equal(ceiling.numpy(), nominal[2]))

    def test_a_nominal_draw_gives_the_nominal_check(self):
        from locomotion.evaluation.runner import expected_motor_law

        q, dq, target, _, _, factor = self.batch(1.0)
        law = {
            "kp": np.full((64, 18), rnd.KP, np.float32),
            "kd": np.tile(np.asarray(KD, np.float32), (64, 1)),
            "strength": factor.numpy(),
        }
        args = (q.numpy(), dq.numpy(), target.numpy())
        for drawn, nominal in zip(expected_motor_law(*args, law), expected_motor_law(*args)):
            self.assertTrue(np.array_equal(drawn, nominal))

    def test_the_native_capture_source_checks_the_reset_draw(self):
        from types import SimpleNamespace

        from locomotion.evaluation.runner import NativeCaptureSource, expected_motor_law

        q, dq, target, kp, kd, factor = self.batch(1.2)
        law = {"kp": kp.expand(64, 18).numpy(), "kd": kd.numpy(), "strength": factor.numpy()}
        env = SimpleNamespace(motor_law={k: v.copy() for k, v in law.items()})
        source = NativeCaptureSource(env)
        env.motor_law["strength"][:] = 0.8  # a later redraw must not move the snapshot
        args = (q.numpy(), dq.numpy(), target.numpy())
        torques = rnd.motor_force(q, dq, target, kd, kp, factor)
        for actual, wanted in zip(torques, source.servo(*args)):
            self.assertTrue(np.array_equal(actual.numpy(), wanted))
        self.assertEqual(
            source.metadata["motor_law"], "canonical_rs05_with_reset_gains_and_strength"
        )
        self.assertEqual(source.metadata["motor_snapshot"]["strength"], law["strength"].tolist())
        nominal = NativeCaptureSource(SimpleNamespace(motor_law=None))
        self.assertEqual(nominal.metadata["motor_law"], "canonical_rs05")
        self.assertNotIn("motor_snapshot", nominal.metadata)
        for drawn, wanted in zip(nominal.servo(*args), expected_motor_law(*args)):
            self.assertTrue(np.array_equal(drawn, wanted))


def load_actuator():
    """Import robots/actuator.py over stand-ins for the Isaac Lab actuator base and configclass."""
    import importlib
    import sys
    from unittest import mock

    stubs = {
        "isaaclab": mock.MagicMock(),
        "isaaclab.actuators": mock.MagicMock(ActuatorBase=object, ActuatorBaseCfg=object),
        "isaaclab.utils": mock.MagicMock(configclass=lambda cls: cls),
    }
    saved = {name: sys.modules.get(name) for name in (*stubs, "locomotion.robots.actuator")}
    sys.modules.update(stubs)
    sys.modules.pop("locomotion.robots.actuator", None)
    try:
        return importlib.import_module("locomotion.robots.actuator")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class NativeMotorDrawTests(unittest.TestCase):
    """The native actuator and environment apply the same motor draw as the surrogate."""

    def setUp(self):
        generator = torch.Generator().manual_seed(9)
        self.q = 0.4 * torch.randn(4, 18, generator=generator)
        self.dq = torch.linspace(-50.0, 50.0, 72).reshape(4, 18)
        self.target = self.q + 0.3 * torch.randn(4, 18, generator=generator)
        self.motor = object.__new__(load_actuator().Motor)
        self.motor.damping = torch.tensor(KD).expand(4, -1).clone()

    def compute(self):
        from types import SimpleNamespace

        action = SimpleNamespace(joint_positions=self.target.clone(), joint_velocities=None)
        self.motor.compute(action, self.q, self.dq)
        return self.motor.computed_effort, self.motor.applied_effort, self.motor.ceiling

    def test_no_draw_keeps_the_nominal_law_and_a_nominal_draw_matches_it(self):
        nominal = motor_force(self.q, self.dq, self.target, self.motor.damping)
        for actual, expected in zip(self.compute(), nominal):
            self.assertTrue(torch.equal(actual, expected))
        self.motor.draw = {
            "kp": torch.full((4, 18), rnd.KP),
            "kd": self.motor.damping.clone(),
            "strength": torch.ones(4, 18),
        }
        for actual, expected in zip(self.compute(), nominal):
            self.assertTrue(torch.equal(actual, expected))

    def test_a_draw_applies_the_randomized_law(self):
        generator = torch.Generator().manual_seed(3)
        draw = {
            "kp": rnd.KP * (0.8 + 0.4 * torch.rand(4, 1, generator=generator)).expand(4, 18),
            "kd": self.motor.damping * (0.8 + 0.4 * torch.rand(4, 1, generator=generator)),
            "strength": 0.8 + 0.4 * torch.rand(4, 18, generator=generator),
        }
        self.motor.draw = draw
        expected = rnd.motor_force(
            self.q, self.dq, self.target, draw["kd"], draw["kp"], draw["strength"]
        )
        for actual, wanted in zip(self.compute(), expected):
            self.assertTrue(torch.equal(actual, wanted))

    def environment(self, n=3, seed=20260917, kinematics=None, init=True):
        """A native environment over stand-in PhysX views; ``kinematics`` (a surrogate) poses its links."""
        from types import SimpleNamespace
        from unittest import mock

        from locomotion.tests.test_direct_environment import load_isaac

        module = load_isaac()
        module.wp = SimpleNamespace(
            to_torch=lambda x: x,
            from_torch=lambda x: x,
            uint32=None,
            array=lambda x, **kwargs: torch.as_tensor(x.astype("int64")),
        )
        env = object.__new__(module.IsaacEnv)
        # Native actuator and link orders differ from the canonical ones; draws must follow names.
        native = list(reversed(JOINT_NAMES))
        links = list(reversed(BODY_NAMES))
        # Two shapes on each tibia, three on the base and one elsewhere.
        counts = [2 if b.endswith("_tibia") else 3 if b == "body" else 1 for b in links]
        written = []
        model = json.loads(MODEL.read_text())
        named = {link["name"]: link for link in model["links"]}
        # PhysX state: the ledger's inertial values and the pose last written for each row.
        physx = {
            "masses": torch.tensor([named[b]["mass"] for b in links]).expand(n, -1).clone(),
            "coms": torch.tensor([[*named[b]["com"], 0.0, 0.0, 0.0, 1.0] for b in links])
            .expand(n, -1, -1)
            .clone(),
            "inertias": torch.tensor([named[b]["inertia"] for b in links])
            .reshape(1, -1, 9)
            .expand(n, -1, -1)
            .clone(),
            "root": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(n, -1).clone(),
            "q": torch.zeros(n, 18),
        }

        def setter(key):
            def write(data, rows):
                physx[key][rows] = data[rows]

            return write

        def link_transforms():
            # The surrogate's MuJoCo kinematics stand in for PhysX: same frames, xyzw quaternions.
            import mujoco

            m = kinematics.mj_model
            d = mujoco.MjData(m)
            result = torch.zeros(n, len(links), 7)
            for row in range(n):
                root = physx["root"][row].double().numpy()
                d.qpos[:3], d.qpos[3:7] = root[:3], root[[6, 3, 4, 5]]
                d.qpos[kinematics._qpos_columns] = (
                    physx["q"][row].double().numpy()[env.joint_index.numpy()]
                )
                mujoco.mj_kinematics(m, d)
                for k, name in enumerate(links):
                    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                    result[row, k] = torch.tensor([*d.xpos[b], *d.xquat[b][[1, 2, 3, 0]]])
            return result

        view = SimpleNamespace(
            link_paths=[[f"/World/envs/env_0/Robot/{b}" for b in links]],
            get_material_properties=lambda: torch.tensor([1.0, 1.0, 0.0]).expand(
                n, sum(counts), -1
            ),
            set_material_properties=lambda data, rows: written.append((data.clone(), rows)),
            get_link_transforms=link_transforms,
            get_root_transforms=lambda: physx["root"].clone(),
            **{"get_" + k: (lambda k=k: physx[k].clone()) for k in ("masses", "coms", "inertias")},
            **{"set_" + k: setter(k) for k in ("masses", "coms", "inertias")},
        )

        def write_root(root_pose, env_ids):
            physx["root"][env_ids.long()] = root_pose

        def write_q(position, env_ids):
            physx["q"][env_ids.long()] = position

        composer = SimpleNamespace(out_force_b=None, out_torque_b=None)

        def set_wrench(forces, torques, body_ids, env_ids):
            force, torque = torch.zeros(n, len(links), 3), torch.zeros(n, len(links), 3)
            force[env_ids.long()[:, None], body_ids.long()] = forces
            torque[env_ids.long()[:, None], body_ids.long()] = torques
            composer.out_force_b = SimpleNamespace(warp=force)
            composer.out_torque_b = SimpleNamespace(warp=torque)

        composer.set_forces_and_torques_index = set_wrench
        env.robot = mock.MagicMock(
            body_names=links,
            permanent_wrench_composer=composer,
            write_root_link_pose_to_sim_index=write_root,
            write_joint_position_to_sim_index=write_q,
        )
        env.config = SimpleNamespace(num_envs=n, seed=seed, physics_dt=0.0025, decimation=8)
        env.randomization = "table2_v1"
        env.device = "cpu"
        env.model = model
        env.geometry_path = ROOT / "robot/hexapod_mkii_updated_v1/geometry/geometry.json"
        env.geometry_meta = json.loads(env.geometry_path.read_text())
        env.native_body_names = links
        env.native_joint_names = native
        env.joint_index = torch.tensor([native.index(j) for j in JOINT_NAMES])
        env.inverse_joint_index = torch.argsort(env.joint_index)
        env.body_index = links.index("body")
        env.toe_index = torch.tensor([links.index(leg + "_tibia") for leg in LEGS])
        env.root_com_local = physx["coms"][:, env.body_index, :3].clone()
        surrogate = kinematics
        env.neutral = surrogate.neutral.clone() if surrogate else torch.zeros(18)
        env.lower, env.upper = (
            (surrogate.lower.clone(), surrogate.upper.clone())
            if surrogate
            else (torch.full((18,), -3.0), torch.full((18,), 3.0))
        )
        env.reset_height = surrogate.reset_height if surrogate else 0.1
        env.origins = torch.zeros(n, 3)
        env.view = view
        env.sim = SimpleNamespace(
            physics_sim_view=SimpleNamespace(
                create_rigid_body_view=lambda path: SimpleNamespace(
                    max_shapes=counts[links.index(path.rsplit("/", 1)[1])]
                ),
                update_articulations_kinematic=lambda: None,
            )
        )
        env.torch = lambda x: torch.as_tensor(x, dtype=torch.float32)
        env.motor = SimpleNamespace(
            joint_names=native,
            damping=torch.tensor([KD[JOINT_NAMES.index(j)] for j in native]).expand(n, -1).clone(),
            draw=None,
        )
        env.motor_order = torch.tensor([native.index(j) for j in JOINT_NAMES])
        env.written, env.counts, env.links, env.physx = written, counts, links, physx
        if init:
            env._init_randomization()
        return env

    def test_friction_covers_every_tibia_shape_of_the_reset_row_only(self):
        env = self.environment()
        env.redraw(torch.tensor([2]))
        expected = rnd.draw("table2_v1", rnd.replica_generator(20260917, 2, "table2_v1"))
        data, rows = env.written[-1]
        self.assertEqual(rows.tolist(), [2])
        starts = np.cumsum([0, *env.counts])
        tibia = [
            k
            for name, start, count in zip(env.links, starts, env.counts)
            if name.endswith("_tibia")
            for k in range(start, start + count)
        ]
        self.assertEqual(sorted(env.tibia_shapes.tolist()), tibia)
        self.assertEqual(len(tibia), 12)
        friction = np.float32(expected.foot_friction)
        self.assertTrue(torch.all(data[2, tibia, :2] == friction))
        self.assertTrue(torch.all(data[2, tibia, 2] == 0.0))
        others = [k for k in range(sum(env.counts)) if k not in tibia]
        self.assertTrue(torch.all(data[2, others] == torch.tensor([1.0, 1.0, 0.0])))
        self.assertTrue(torch.all(data[[0, 1]] == torch.tensor([1.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(env.expected_materials, data))
        self.assertEqual(env.ground_friction.tolist()[2], float(friction))
        self.assertEqual(env.ground_friction.tolist()[:2], [1.0, 1.0])

    def test_a_link_layout_that_differs_from_the_body_order_is_refused(self):
        env = self.environment()
        env.native_body_names = list(BODY_NAMES)
        with self.assertRaisesRegex(ValueError, "Native shape layout differs from the link order"):
            env._init_randomization()

    def test_each_reset_row_draws_from_its_own_generator_in_joint_name_order(self):
        env = self.environment()
        before = env.motor_law
        self.assertTrue(np.array_equal(before["kd"], np.tile(np.float32(KD), (3, 1))))
        env.redraw(torch.tensor([1]))
        after = env.motor_law
        for name in ("kp", "kd", "strength"):
            self.assertTrue(np.array_equal(after[name][[0, 2]], before[name][[0, 2]]))
        expected = rnd.draw("table2_v1", rnd.replica_generator(20260917, 1, "table2_v1"))
        self.assertTrue(
            np.array_equal(after["kp"][1], np.full(18, np.float32(rnd.KP * expected.kp_scale)))
        )
        self.assertTrue(
            np.array_equal(after["kd"][1], (torch.tensor(KD) * expected.kd_scale).numpy())
        )
        self.assertTrue(np.array_equal(after["strength"][1], np.float32(expected.motor_strength)))

    def test_the_native_motor_law_matches_the_surrogate_first_draw(self):
        from locomotion.surrogate.env import make_env

        surrogate = make_env(3, threads=0, randomization="table2_v1", seed=20260917)
        try:
            wanted = surrogate.motor_law
        finally:
            surrogate.close()
        env = self.environment()
        env.redraw(torch.arange(3))
        for name in ("kp", "kd", "strength"):
            self.assertTrue(np.array_equal(env.motor_law[name], wanted[name]), name)

    def test_pushes_keep_each_row_stream_in_step_with_the_surrogate(self):
        """Over 30 s with mid-run resets, both engines time the same pushes and draw the same streams."""
        from locomotion.surrogate.env import make_env

        surrogate = make_env(3, threads=0, randomization="table2_v1", seed=20260917)
        try:
            env = self.environment()
            env.redraw(torch.arange(3))
            resets = {400: [1], 900: [0, 2], 1300: [1]}
            ended = pushed = 0
            for control in range(1500):
                for substep in range(8):
                    surrogate._apply_pushes(substep)
                    env.push_window(substep)
                    self.assertTrue(
                        np.array_equal(env.push_body[:, substep], surrogate._push_body[:, substep])
                    )
                    before = env.push_end.copy()
                    surrogate._advance_pushes()
                    env.advance_pushes()
                    ended += int(np.sum(env.push_end != before))
                    pushed += int(np.any(env.push_body[:, substep] != 0))
                self.assertTrue(
                    torch.equal(env.external_wrench_body, surrogate_wrench(surrogate)), control
                )
                for row in resets.get(control, []):
                    surrogate._redraw(row)
                    env.redraw(torch.tensor([row]))
                for name in ("since_reset", "push_start", "push_end", "push_force"):
                    self.assertTrue(
                        np.array_equal(getattr(env, name), getattr(surrogate, "_" + name)), name
                    )
            self.assertGreater(ended, 6)
            self.assertGreater(pushed, 0)
            for row in range(3):
                self.assertEqual(
                    env.generators[row].bit_generator.state,
                    surrogate._generators[row].bit_generator.state,
                )
            for name in ("kp", "kd", "strength"):
                self.assertTrue(
                    np.array_equal(env.motor_law[name], surrogate.motor_law[name]), name
                )
            self.assertTrue(torch.equal(env.ground_friction, surrogate.ground_friction))
        finally:
            surrogate.close()

    def test_the_native_link_masses_and_payload_match_the_surrogate_models(self):
        """Each row's 19 masses, centres of mass and link-axis tensors equal the surrogate's draw."""
        import mujoco

        from locomotion.surrogate.env import make_env

        surrogate = make_env(3, threads=0, randomization="table2_v1", seed=20260917)
        try:
            env = self.environment()
            env.redraw(torch.arange(3))
            order = [env.links.index(b) for b in BODY_NAMES]
            payloads = 0
            for row in range(3):
                model = surrogate._models[row]
                ids = surrogate._body_ids
                tensors = []
                for b in ids:
                    rotation = np.zeros(9)
                    mujoco.mju_quat2Mat(rotation, model.body_iquat[b])
                    rotation = rotation.reshape(3, 3)
                    tensors.append(rotation @ np.diag(model.body_inertia[b]) @ rotation.T)
                tensors = torch.tensor(np.array(tensors))
                native = env.physx["inertias"][row, order].double().reshape(-1, 3, 3)
                scale = tensors.abs().amax((-2, -1), keepdim=True)
                self.assertTrue(torch.all((native - tensors).abs() <= 1e-6 * scale), row)
                self.assertTrue(
                    np.allclose(
                        env.physx["masses"][row, order].numpy(),
                        model.body_mass[ids],
                        rtol=1e-6,
                        atol=0,
                    ),
                    row,
                )
                self.assertTrue(
                    np.allclose(
                        env.physx["coms"][row, order, :3].numpy(), model.body_ipos[ids], atol=1e-7
                    ),
                    row,
                )
                self.assertTrue(
                    torch.allclose(
                        env.root_com_local[row],
                        torch.tensor(model.body_ipos[surrogate._base_id], dtype=torch.float32),
                    )
                )
                payloads += int(model.body_mass[surrogate._base_id] > 2.5)
            self.assertGreater(payloads, 0)
        finally:
            surrogate.close()

    def test_the_native_reset_pose_and_root_lift_match_the_surrogate(self):
        """The drawn offsets reach the joints and the root rises to the nominal toe clearance."""
        from locomotion.surrogate.env import make_env

        surrogate = make_env(3, threads=0, randomization="table2_v1", seed=20260917)
        try:
            env = self.environment(kinematics=surrogate)
            env.view.get_dof_actuation_forces = lambda: torch.zeros(3, 18)
            env.view.set_dof_actuation_forces = lambda *args: None
            env.held, env.previous_action = torch.zeros(3, 18), torch.zeros(3, 18)
            env.write_reset(torch.arange(3))
            wanted = surrogate._qpos0[:, surrogate._qpos_columns]
            self.assertTrue(np.allclose(env.reset_q.numpy(), wanted, atol=1e-6))
            canonical = env.physx["q"][:, env.joint_index]
            self.assertTrue(torch.equal(canonical, env.reset_q))
            self.assertTrue(
                torch.allclose(env.reset_heights, surrogate.reset_heights, atol=2e-6, rtol=0)
            )
            self.assertTrue(torch.equal(env.physx["root"][:, 2], env.reset_heights))
            # The draws lift every row; the held target keeps the nominal pose.
            self.assertTrue(torch.all(env.reset_heights > env.reset_height))
            target = env.robot.set_joint_position_target_index.call_args.kwargs["target"]
            self.assertTrue(torch.equal(target, env.neutral[env.inverse_joint_index].expand(3, -1)))
        finally:
            surrogate.close()

    def test_a_drawn_pose_outside_the_named_limits_is_refused(self):
        env = self.environment()
        env.redraw(torch.arange(3))
        env.nominal_toe_low = 0.0
        env.upper = env.neutral.clone()
        with self.assertRaisesRegex(ValueError, "named joint limits"):
            env.drawn_pose_lift(torch.arange(3), torch.arange(3, dtype=torch.int32))

    def test_the_push_reaches_the_base_wrench_and_is_read_back(self):
        env = self.environment()
        env.redraw(torch.arange(3))
        env.since_reset[:] = env.push_start
        env.push_window(0)
        env.verify_push(0)
        force = env.robot.permanent_wrench_composer.out_force_b.warp
        self.assertTrue(
            torch.equal(force[:, env.body_index], torch.tensor(env.push_force, dtype=torch.float32))
        )
        self.assertTrue(torch.any(force[:, env.body_index] != 0))
        others = [k for k in range(len(env.links)) if k != env.body_index]
        self.assertTrue(torch.all(force[:, others] == 0))
        force[0, env.body_index, 0] += 1.0
        with self.assertRaisesRegex(RuntimeError, "push wrench"):
            env.verify_push(0)

    def test_an_inertial_write_that_does_not_read_back_is_refused(self):
        env = self.environment()
        env.view.set_masses = lambda data, rows: None
        with self.assertRaisesRegex(ValueError, "drawn masses"):
            env.redraw(torch.arange(3))

    def test_a_nominal_tensor_off_the_ledger_axes_is_refused(self):
        env = self.environment(init=False)
        env.physx["inertias"][:, env.links.index("body"), 1] = 1e-3
        with self.assertRaisesRegex(ValueError, "ledger"):
            env._init_randomization()

    def test_a_reset_clears_the_row_wrench_and_restarts_its_clock(self):
        env = self.environment()
        env.redraw(torch.arange(3))
        env.external_wrench_body[:] = 1.0
        env.since_reset[:] = 50
        env.redraw(torch.tensor([1]))
        self.assertEqual(env.since_reset.tolist(), [50, 0, 50])
        self.assertTrue(torch.all(env.external_wrench_body[1] == 0))
        self.assertTrue(torch.all(env.external_wrench_body[[0, 2]] == 1))


def surrogate_wrench(surrogate):
    """The surrogate's privileged push wrench as its step computes it from the substep pushes."""
    mean = torch.as_tensor(surrogate._push_body.mean(1), dtype=torch.float32)
    return torch.cat((mean, torch.zeros((surrogate.num_envs, 3))), dim=-1)


if __name__ == "__main__":
    unittest.main()
