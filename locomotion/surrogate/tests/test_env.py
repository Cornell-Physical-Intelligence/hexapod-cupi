"""Check the surrogate environment against the native interface, limiter and termination rules."""
from dataclasses import fields
import unittest
from unittest import mock

import numpy as np
import torch

from locomotion import noise_probe
from locomotion.env import LocomotionEnv, emitted_target
from locomotion.env_config import MASS_KG, EnvConfig
from locomotion.surrogate.env import (MAX_JOINT_SPEED, ContactModel, SurrogateConfig, SurrogateEnv, contact_overrides,
                                      make_env)


def roll(env, controls=30, std=.3, seed=7):
    """Step seeded Gaussian actions and return the physics state after the last control, without the clock column."""
    env.reset()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(controls):
        output = env.step(std * torch.randn(3, 18, generator=generator)[:env.num_envs])
        done = output["terminated"] | output["truncated"]
        if bool(done.any()):
            env.reset(done.nonzero().flatten())
    return env._state[:, 1:].copy()


class EnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.env = make_env(3, threads=0)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset()

    def test_interface_matches_the_native_environment(self):
        env = self.env
        for name in ("num_actions", "observation_width", "critic_width", "amp_width"):
            self.assertEqual(getattr(SurrogateEnv, name), getattr(LocomotionEnv, name))
        output = env.reset()
        self.assertEqual((tuple(output["obs"].shape), tuple(output["critic"].shape)), ((3, 231), (3, 234)))
        self.assertTrue(torch.equal(output["obs"][:, 210:213], env.commands))
        self.assertEqual(float(output["obs"][:, 213:231].abs().max()), 0.)
        native = {field.name: field.default for field in fields(EnvConfig)}
        mine = {field.name: field.default for field in fields(SurrogateConfig)}
        self.assertEqual(list(native), list(mine))
        self.assertEqual({key for key in native if native[key] != mine[key]}, {"num_envs", "device"})
        self.assertEqual(SurrogateConfig().control_dt, .02)
        self.assertAlmostEqual(float(env.mj_model.body_mass.sum()), MASS_KG, delta=1e-9)
        declaration = SurrogateConfig().declaration()
        self.assertTrue(declaration["surrogate"])
        self.assertEqual((declaration["actor_width"], declaration["critic_width"], declaration["amp_width"]), (231, 234, 61))
        env.step(torch.zeros(3, 18))
        self.assertTrue(set(noise_probe.FIELDS) <= set(env.telemetry))

    def test_target_follows_the_native_limiter(self):
        env = self.env
        held = env.held.clone()
        action = torch.full((3, 18), 5.)
        expected = emitted_target(action, held, env.neutral, env.lower, env.upper, .35, .04)
        env.step(action)
        target = env.telemetry["joint_target_rad"]
        self.assertTrue(torch.equal(target, expected))
        self.assertLessEqual(float((target - held).abs().max()), .04 + 1e-6)
        self.assertTrue(bool(((target >= env.lower) & (target <= env.upper)).all()))
        self.assertTrue(torch.allclose(env.previous_action, (target - env.neutral) / .35))
        for bad in (torch.full((3, 18), float("nan")), torch.zeros(3, 17)):
            with self.assertRaises(ValueError):
                env.step(bad)

    def test_joint_speed_clamp_counts_each_clipped_value(self):
        env = self.env
        column = 26 + env._qvel_columns[2]
        before = env.speed_clamp_events
        env._state[0, column] = 80.
        env._advance(torch.zeros(3, 18))
        self.assertEqual(env.speed_clamp_events - before, 1)
        self.assertEqual(env._state[0, column], MAX_JOINT_SPEED)
        self.assertGreaterEqual(env.peak_joint_speed, MAX_JOINT_SPEED)

    def test_termination_reads_tilt_and_joint_stops(self):
        env = self.env
        env._state[2, 4], env._state[2, 5] = np.cos(.6), np.sin(.6)
        output = env.step(torch.zeros(3, 18))
        self.assertEqual(output["terminated"].tolist(), [False, False, True])
        # Hold the written state through the control, so the rule reads the joint position alone.
        env.reset()
        column = 1 + env._qpos_columns[2]
        for excess, expected in ((1e-4, True), (1e-6, False)):
            env._state[1, column] = float(env.upper[2]) + excess
            with mock.patch.object(env, "_advance", return_value=env._forces):
                output = env.step(torch.zeros(3, 18))
            self.assertEqual(output["terminated"].tolist(), [False, expected, False])

    def test_episode_timeout_truncates(self):
        env = SurrogateEnv(SurrogateConfig(num_envs=1, episode_seconds=.1), threads=0)
        flags = [bool(env.step(torch.zeros(1, 18))["truncated"][0]) for _ in range(5)]
        env.close()
        self.assertEqual(flags, [False, False, False, False, True])

    def test_step_is_deterministic_across_batch_and_threads(self):
        first = roll(self.env)
        self.assertTrue(np.array_equal(first, roll(self.env)))
        threaded = make_env(3, threads=2)
        alone = make_env(1, threads=0)
        try:
            self.assertTrue(np.array_equal(first, roll(threaded)))
            self.assertTrue(np.array_equal(first[:1], roll(alone)))
        finally:
            threaded.close()
            alone.close()

    def test_selected_reset_restores_named_rows(self):
        env = self.env
        start = env._state.copy()
        for _ in range(5):
            env.step(.3 * torch.ones(3, 18))
        moved = env._state.copy()
        env.reset([1])
        self.assertTrue(np.array_equal(env._state[1, 1:], start[1, 1:]))
        self.assertTrue(np.array_equal(env._state[[0, 2]], moved[[0, 2]]))
        self.assertEqual(env.episode_steps.tolist(), [5, 0, 5])
        self.assertEqual(float(env.previous_action[1].abs().max()), 0.)
        for rows in ([1, 1], [3], [-1]):
            with self.assertRaises(ValueError):
                env.reset(rows)

    def test_contact_model_overrides(self):
        model = contact_overrides(["noslip_iterations=0", "friction=(0.9, 0.005, 0.0001)"])
        self.assertEqual((model.noslip_iterations, model.friction), (0, (.9, .005, .0001)))
        self.assertEqual(contact_overrides([]), ContactModel())
        for bad in (["unknown_field=1"], ["noslip_iterations"]):
            with self.assertRaises(ValueError):
                contact_overrides(bad)
        with self.assertRaises(ValueError):
            ContactModel(position_blend=.5, integrator="implicitfast")
        with self.assertRaises(ValueError):
            ContactModel(overshoot_phase=(.9, .75))

    def test_touchdown_counters(self):
        env = self.env
        env.touchdown_events = env.overshoot_events = 0
        for _ in range(25):
            env.step(torch.zeros(3, 18))
        self.assertGreaterEqual(env.touchdown_events, 6 * 3)
        plain = make_env(3, threads=0, contact=ContactModel(touchdown_overshoot=False))
        roll(plain)
        plain.close()
        self.assertEqual((plain.touchdown_events, plain.overshoot_events), (0, 0))
        roll(env)
        self.assertGreater(env.overshoot_events, 0)


if __name__ == "__main__":
    unittest.main()
