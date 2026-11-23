"""Check the stored-cycle controller: source selection, ramps and the fitted joint gain."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomotion.amp.cycle import CycleController, bank_from_replays

NEUTRAL = np.tile([0.0, -0.3, 0.4], 6)
FORWARD, LEFT, STAND = (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.0)


def bank(period=4):
    rng = np.random.default_rng(0)
    targets = NEUTRAL + rng.uniform(-0.1, 0.1, (2, period, 18))
    deflection = rng.uniform(0.02, 0.05, (2, period, 18))
    return np.array([FORWARD, LEFT]), targets, targets - deflection


class CycleControllerTests(unittest.TestCase):
    def controller(self, **options):
        return CycleController(*bank(), NEUTRAL, ramp_controls=2, **options)

    def test_walk_ramps_from_neutral_at_phase_zero_then_repeats_the_cycle(self):
        targets = bank()[1][0]
        controller = self.controller()
        self.assertTrue(np.array_equal(controller(STAND, NEUTRAL), NEUTRAL))
        sent = [controller(FORWARD, NEUTRAL) for _ in range(9)]
        self.assertTrue(np.allclose(sent[0], NEUTRAL + 0.5 * (targets[0] - NEUTRAL)))
        for control in range(1, 9):
            self.assertTrue(np.array_equal(sent[control], targets[control % 4]))

    def test_zero_command_ramps_to_neutral_and_holds(self):
        controller = self.controller()
        last = [controller(FORWARD, NEUTRAL) for _ in range(5)][-1]
        first, *rest = [controller(STAND, NEUTRAL) for _ in range(4)]
        self.assertTrue(np.allclose(first, last + 0.5 * (NEUTRAL - last)))
        for target in rest:
            self.assertTrue(np.array_equal(target, NEUTRAL))

    def test_command_change_ramps_from_the_last_target_and_keeps_the_phase(self):
        targets = bank()[1]
        controller = self.controller()
        last = [controller(FORWARD, NEUTRAL) for _ in range(5)][-1]
        first, second = controller(LEFT, NEUTRAL), controller(LEFT, NEUTRAL)
        self.assertTrue(np.allclose(first, last + 0.5 * (targets[1][1] - last)))
        self.assertTrue(np.array_equal(second, targets[1][2]))

    def test_command_outside_the_bank_raises(self):
        with self.assertRaisesRegex(KeyError, "no cycle"):
            self.controller()((0.1, 0.0, 0.0), NEUTRAL)

    def test_gain_settles_at_the_inverse_of_a_soft_motor_stiffness(self):
        _, targets, references = bank()
        deflection, stiffness = (targets - references)[0], 0.6
        for rate, expected in ((0.0, 1.0), (0.5, 1 / stiffness)):
            controller = self.controller(learning_rate=rate)
            angle = NEUTRAL
            for control in range(200):
                sent = controller(FORWARD, angle)
                # The servo holds the nominal load at a deflection of 1/stiffness times the nominal one.
                angle = sent - deflection[control % 4] / stiffness
            self.assertTrue(np.allclose(controller.gain, expected, atol=1e-3))
        self.assertTrue(np.array_equal(self.controller(learning_rate=0.5).gain, np.ones(18)))


class BankTests(unittest.TestCase):
    def test_bank_rows_follow_the_phase_of_a_phase_zero_replay(self):
        rng = np.random.default_rng(1)
        cycle, settled = rng.normal(size=(60, 18)), rng.normal(size=(60, 18))
        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                Path(directory) / "control_trace.npz",
                command=np.tile(np.float32([0.05, 0.0, 0.0]), (1000, 1, 1)),
                joint_target_rad=np.tile(cycle, (17, 1))[:1000, None],
                joint_position_rad=np.tile(settled, (17, 1))[:1000, None],
            )
            commands, targets, references = bank_from_replays([directory])
        self.assertEqual(
            (commands.shape, targets.shape, references.shape), ((1, 3), (1, 60, 18), (1, 60, 18))
        )
        self.assertTrue(np.allclose(targets[0], cycle) and np.allclose(references[0], settled))


if __name__ == "__main__":
    unittest.main()
