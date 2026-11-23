"""Check the stored-cycle controller: source selection, ramps and the fitted joint gain."""

import hashlib
import json
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from locomotion.amp import cycle as cycle_module
from locomotion.amp.cycle import CycleController, _bank_from_traces, bank_from_replays

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

    def test_gain_fits_delayed_phase_and_discards_old_samples_on_switch(self):
        _, targets, references = bank()
        deflection, stiffness, delay = targets - references, 0.6, 3
        controller = self.controller(learning_rate=0.5, action_delay=delay)
        pending = deque([(NEUTRAL, np.zeros(18))] * delay)
        angle = NEUTRAL
        for control in range(250):
            command = LEFT if control == 6 else FORWARD
            index = int(command == LEFT)
            phase = controller.phase
            sent = controller(command, angle)
            pending.append((sent, deflection[index, phase]))
            applied, offset = pending.popleft()
            angle = applied - offset / stiffness
            if 6 <= control < 12:
                self.assertEqual(controller.fit_samples, 0)
                np.testing.assert_array_equal(controller.gain, np.ones(18))
        np.testing.assert_allclose(controller.gain, 1 / stiffness, atol=1e-3)
        controller.reset()
        np.testing.assert_array_equal(controller.gain, np.ones(18))
        self.assertEqual(controller.fit_samples, 0)


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

    def test_rejects_a_nonrepeating_later_target(self):
        rng = np.random.default_rng(1)
        cycle = rng.normal(size=(60, 18))
        targets = np.tile(cycle, (15, 1))[:, None]
        targets[899, 0, 0] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                Path(directory) / "control_trace.npz",
                command=np.tile(np.float32(FORWARD), (900, 1, 1)),
                joint_target_rad=targets,
                joint_position_rad=targets,
            )
            path = Path(directory) / "control_trace.npz"
            with self.assertRaisesRegex(ValueError, "no complete repeated target cycle"):
                _bank_from_traces([(path, path.read_bytes())])

    def test_cli_accepts_only_admitted_phase_zero_and_preserves_existing_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            evaluation = folder / "replay_00_phase05/run/standing/evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "declaration.json").write_text(
                json.dumps({"schema": "hexapod_native_batch_v1", "cases": []})
            )
            path = evaluation / "control_trace.npz"
            wave = np.random.default_rng(4).normal(size=(60, 18))
            targets = np.tile(wave, (15, 1))[:, None]
            np.savez(
                path,
                command=np.tile(np.float32(FORWARD), (900, 1, 1)),
                joint_target_rad=targets,
                joint_position_rad=targets,
            )
            output = folder / "bank"
            manifest = folder / "admission.json"
            trace_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            clip = {"start_phase": 0.5, "trace_sha256": trace_hash}
            admission = {"admitted": True, "model_sha256": "model", "clips": [clip]}
            args = ["--replays", str(evaluation), "--output", str(output)]
            with patch.object(cycle_module, "ADMISSION", manifest):
                manifest.write_text(json.dumps(admission))
                with self.assertRaisesRegex(ValueError, "not an admitted phase-zero replay"):
                    bank_from_replays([evaluation])
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    cycle_module.main(args)
                self.assertIn("not an admitted phase-zero replay", stderr.getvalue())
                clip["start_phase"] = 0
                manifest.write_text(json.dumps(admission))
                _, targets, _ = bank_from_replays([evaluation])
                np.testing.assert_array_equal(targets[0], wave)
                raw = folder / "raw"
                raw.mkdir()
                (raw / "control_trace.npz").write_bytes(path.read_bytes() + b"other trace")
                with self.assertRaisesRegex(ValueError, "not an admitted phase-zero replay"):
                    bank_from_replays([evaluation, raw])
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    cycle_module.main(["--replays", str(raw), "--output", str(output)])
                cycle_module.main(args)
                saved = output.with_suffix(".npz")
                with np.load(saved, allow_pickle=False) as bank:
                    np.testing.assert_array_equal(bank["targets"][0], wave)
                    self.assertEqual(bank["trace_sha256"].tolist(), [trace_hash])
                original = saved.read_bytes()
                with self.assertRaises(FileExistsError):
                    cycle_module.main(args)
                self.assertEqual(saved.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
