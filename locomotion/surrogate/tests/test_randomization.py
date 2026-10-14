"""Check that the surrogate applies Table II draws per replica, isolates replicas and keeps ``none`` unchanged."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from locomotion import randomization as rnd
from locomotion.env import motor_force
from locomotion.env_config import BODY_NAMES, KD, MASS_KG
from locomotion.surrogate import evaluate, train
from locomotion.surrogate.env import make_env


def first_draws(env):
    """The draw each replica's constructor reset took: a fresh generator repeats it."""
    return [rnd.draw(env.randomization, rnd.replica_generator(env.cfg.seed, i, env.randomization))
            for i in range(env.num_envs)]


def body_tensor(quaternion, principal):
    """A body's inertia tensor in its own frame from MuJoCo's principal values and axes."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(principal) @ rotation.T


class RandomizedEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.env = make_env(3, threads=0, randomization="table2_v1")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_none_shares_the_compiled_model_and_adds_no_telemetry(self):
        env = make_env(2, threads=0)
        try:
            self.assertTrue(all(model is env.mj_model for model in env._models))
            env.step(torch.zeros(2, 18))
            self.assertNotIn("ground_friction", env.telemetry)
            self.assertNotIn("external_wrench_body", env.telemetry)
            self.assertEqual(env.native_readback["randomization"], rnd.identity("none"))
        finally:
            env.close()

    def test_constructor_draws_apply_to_each_replica_model(self):
        env = make_env(3, threads=0, randomization="table2_v1")
        try:
            self.assertEqual(len({id(model) for model in env._models}), 3)
            self.assertAlmostEqual(float(env.mj_model.body_mass.sum()), MASS_KG, places=9)
            for i, draw in enumerate(first_draws(env)):
                model = env._models[i]
                ids = env._body_ids
                expected = env._nominal["body_mass"][ids] * draw.link_mass_scale
                expected[BODY_NAMES.index("body")] += draw.payload_mass_kg
                self.assertTrue(np.allclose(model.body_mass[ids], expected))
                self.assertTrue(np.all(model.geom_friction[env._friction_geoms, 0] == draw.foot_friction))
                self.assertAlmostEqual(float(env.ground_friction[i]), draw.foot_friction, places=6)
                self.assertAlmostEqual(float(env._kp[i]), rnd.KP * draw.kp_scale, places=5)
                self.assertTrue(torch.allclose(env._kd[i], torch.tensor(KD) * draw.kd_scale))
                self.assertTrue(np.allclose(env.root_com_local[i].numpy(), model.body_ipos[env._base_id]))
                pose = env._qpos0[i, env._qpos_columns]
                self.assertTrue(np.allclose(pose, env._neutral64 + draw.joint_offset_rad))
                self.assertAlmostEqual(env._lowest_toe(env._qpos0[i]), env._nominal_toe_low, places=12)
            self.assertEqual(env.native_readback["randomization"], rnd.identity("table2_v1"))
            self.assertEqual(env.native_readback["mass_per_replica_kg"], [float(m.body_mass.sum()) for m in env._models])
        finally:
            env.close()

    def test_payload_keeps_the_base_tensor_about_its_new_centre(self):
        env = self.env
        base, model, nominal = env._base_id, env._models[0], env._nominal
        draw = first_draws(env)[0]
        scale = draw.link_mass_scale[BODY_NAMES.index("body")]
        tensor = body_tensor(nominal["body_iquat"][base], nominal["body_inertia"][base] * scale)
        _, centre, expected = rnd.combined_base_inertial(nominal["body_mass"][base] * scale, nominal["body_ipos"][base],
                                                         tensor, draw.payload_mass_kg, draw.payload_position_m)
        self.assertTrue(np.allclose(body_tensor(model.body_iquat[base], model.body_inertia[base]), expected))
        self.assertTrue(np.allclose(model.body_ipos[base], centre))

    def test_selected_reset_redraws_only_that_replica(self):
        env = self.env
        env.reset()
        for _ in range(3):
            env.step(.2 * torch.ones(3, 18))
        models = [(m.body_mass.copy(), m.body_inertia.copy(), m.geom_friction.copy()) for m in env._models]
        state, kp = env._state.copy(), env._kp.clone()
        env.reset([1])
        for i in (0, 2):
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(models[i], (env._models[i].body_mass,
                                                                                 env._models[i].body_inertia,
                                                                                 env._models[i].geom_friction))))
            self.assertTrue(np.array_equal(state[i], env._state[i]))
            self.assertEqual(float(kp[i]), float(env._kp[i]))
        self.assertFalse(np.array_equal(models[1][0], env._models[1].body_mass))
        self.assertTrue(np.array_equal(env._state[1, 1:26], env._qpos0[1]))
        self.assertEqual(float(env.external_wrench_body[1].abs().sum()), 0.)

    def test_draws_and_steps_are_deterministic(self):
        def run():
            env = make_env(2, threads=0, randomization="table2_v1", seed=11)
            generator = torch.Generator().manual_seed(3)
            for _ in range(10):
                env.step(.3 * torch.randn(2, 18, generator=generator))
            result = env._state.copy(), [m.body_mass.copy() for m in env._models]
            env.close()
            return result
        (a, masses_a), (b, masses_b) = run(), run()
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(masses_a, masses_b)))

    def test_push_reports_the_body_force_and_no_torque(self):
        env = self.env
        env.reset()
        # Schedule a push that covers the whole next control on replica 0 alone.
        env._push_start[:] = 10 ** 9
        env._push_end[:] = 10 ** 9 + 1
        env._push_start[0], env._push_end[0] = env._since_reset[0], env._since_reset[0] + 80
        env._push_force[0] = (6., -8., 0.)
        env.step(torch.zeros(3, 18))
        wrench = env.telemetry["external_wrench_body"]
        self.assertTrue(torch.allclose(wrench[0], torch.tensor([6., -8., 0., 0., 0., 0.])))
        self.assertEqual(float(wrench[1:].abs().sum()), 0.)
        column = env.mj_model.nu + 6 * env._base_id
        self.assertAlmostEqual(float(np.linalg.norm(env._control[0, 0, column:column + 3])), 10., places=9)
        self.assertEqual(float(np.abs(env._control[0, 0, column + 3:column + 6]).sum()), 0.)

    def test_push_end_draws_the_next_push_from_the_interval(self):
        env = self.env
        env.reset()
        end = int(env._push_end[2])
        env._since_reset[2] = end - 1
        env._advance_pushes()
        low, high = (env._steps(x) for x in rnd.TABLE2_V1.push.interval_s)
        self.assertTrue(end + low <= env._push_start[2] <= end + high)
        self.assertEqual(env._push_end[2] - env._push_start[2], 80)


class CommandTests(unittest.TestCase):
    def test_both_commands_offer_the_declared_names_and_default_to_none(self):
        for parser in (train.build_parser(), evaluate.build_parser()):
            action = next(a for a in parser._actions if a.dest == "randomization")
            self.assertEqual((tuple(action.choices), action.default), (rnd.NAMES, "none"))

    def test_training_identity_binds_the_declaration_and_its_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                code = train.main(["--output", str(output), "--num-envs", "2", "--threads", "1", "--updates", "1",
                                   "--randomization", "table2_v1"])
            state = json.loads((output / "state.json").read_text())
            self.assertEqual(code, 0, state.get("traceback"))
            recorded = state["identity"]["randomization"]
            declaration = rnd.declaration("table2_v1")
            self.assertEqual(recorded["declaration_sha256"], declaration["sha256"])
            self.assertEqual(rnd.canonical(recorded["declaration"]), declaration["json"])
            self.assertEqual(recorded["declaration"]["table"]["foot_friction"]["high"], 2.5)
            readback = json.loads((output / "native/native_readback.json").read_text())
            self.assertEqual(readback["randomization"], rnd.identity("table2_v1"))
            # The run records the declaration, never a replica's drawn values.
            for path in output.rglob("*"):
                if path.is_file() and path.suffix in (".json", ".jsonl"):
                    self.assertNotIn("link_mass_scale", path.read_text(), path.name)

    def test_randomized_evaluation_refuses_the_friction_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.with_suffix(".json").write_text(json.dumps({"identity": {"ppo_config": {}}}))
            args = evaluate.build_parser().parse_args(["--checkpoint", str(checkpoint), "--output",
                                                       str(Path(directory) / "out"), "--randomization", "table2_v1"])
            with self.assertRaisesRegex(ValueError, "no-contact-consensus"):
                evaluate.evaluate_checkpoint(args)
            self.assertFalse((Path(directory) / "out").exists())


class MotorDrawTests(unittest.TestCase):
    def test_defaults_are_the_nominal_motor_and_strength_never_lifts_the_cap(self):
        generator = torch.Generator().manual_seed(5)
        q, dq, target = (torch.randn(4, 18, generator=generator) for _ in range(3))
        dq = dq * 40.
        kd = torch.tensor(KD)
        nominal = motor_force(q, dq, target, kd)
        explicit = motor_force(q, dq, target, kd, 12.0, None)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(nominal, explicit)))
        _, applied, ceiling = motor_force(q, dq, target, kd.expand(4, -1), torch.full((4, 1), 14.4),
                                          torch.full((4, 18), 1.2))
        self.assertTrue(torch.all(ceiling <= rnd.TORQUE_CAP_NM))
        self.assertTrue(torch.all(applied.abs() <= ceiling))


if __name__ == "__main__":
    unittest.main()
