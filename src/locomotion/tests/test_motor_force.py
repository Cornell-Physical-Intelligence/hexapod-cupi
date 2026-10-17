"""Check motor force arithmetic on the CPU; these tests do not qualify native physics."""
import math
import unittest
import torch
from locomotion.robots.rs05 import motor_force
from locomotion.robots.hexapod import JOINT_NAMES, KD, motor_damping


def named_motor_force(names, q, dq, target):
    return motor_force(q, dq, target, q.new_tensor(motor_damping(names)))


class MotorForceTests(unittest.TestCase):
    def test_permuted_native_order_preserves_exact_named_gains(self):
        generator = torch.Generator().manual_seed(713)
        q, dq, target = [torch.randn((7, 18), generator=generator) for _ in range(3)]
        dq *= 40
        permutation = torch.randperm(18, generator=generator)
        names = [JOINT_NAMES[i] for i in permutation]
        expected = motor_force(q, dq, target, torch.tensor(KD))
        actual = named_motor_force(names, q[:, permutation], dq[:, permutation], target[:, permutation])
        for canonical, native in zip(expected, actual):
            self.assertTrue(torch.equal(canonical[:, permutation], native))

    def test_curve_knots_and_both_speed_signs(self):
        rpm = torch.tensor([0., 70., 275., 340., 450., 477., 480., 500., 479.] * 2)
        velocity = (rpm * (2 * math.pi / 60)).repeat(2, 1)
        velocity[1] *= -1
        q = torch.zeros_like(velocity)
        requested, applied, ceiling = named_motor_force(JOINT_NAMES, q, velocity, torch.full_like(q, 100.))
        expected = motor_force(q, velocity, torch.full_like(q, 100.), torch.tensor(KD))
        for actual, value in zip((requested, applied, ceiling), expected):
            self.assertTrue(torch.equal(actual, value))
        self.assertTrue(torch.equal(ceiling[0], ceiling[1]))
        self.assertTrue(bool((applied.abs() <= 1.6).all()))
        self.assertEqual(float(ceiling[0, 7]), 0.)
        self.assertGreater(float(ceiling[0, 8]), 0.)

    def test_wrong_topology_fails_before_motor_output(self):
        q = torch.zeros((2, 18))
        for names in [JOINT_NAMES[:-1], (*JOINT_NAMES[:-1], JOINT_NAMES[0])]:
            with self.assertRaises(ValueError):
                named_motor_force(names, q, q, q)


if __name__ == "__main__":
    unittest.main()
