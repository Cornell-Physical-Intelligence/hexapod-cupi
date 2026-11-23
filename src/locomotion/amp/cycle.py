"""Walk from stored optimized cycles: pick the cycle by command, ramp between sources, correct joint tracking.

A bank holds, for each command, one period of motor targets and the joint angles that the nominal
robot reached under them in native replay. The controller reads the command and the 18 measured
joint angles, the sensors a robot has. It solves nothing at run time.

    uv run python -m locomotion.amp.cycle --replays REPLAY_DIR... --output bank.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from locomotion.amp.replay_native import startup_target

# One period of settled native replay: the recorded controls 300 to 899 hold ten periods of 60.
SETTLED = slice(300, 900)


def command_key(command):
    return tuple(round(float(value), 4) + 0.0 for value in command)


def bank_from_replays(directories):
    """Commands [n, 3], targets [n, period, 18] and reference joint angles [n, period, 18] in rad.

    Each directory is one native replay of ``replay_native`` from start phase 0. Row ``r`` holds the
    target sent at a control with phase ``r`` and the joint angle measured after that control.
    """
    commands, targets, references = [], [], []
    for directory in directories:
        with np.load(Path(directory) / "control_trace.npz", allow_pickle=False) as trace:
            command, target = trace["command"][:, 0], trace["joint_target_rad"][:, 0]
            angle = trace["joint_position_rad"][:, 0]
        period = next(
            p
            for p in range(20, 151)
            if np.array_equal(target[300 : 300 + p], target[300 + p : 300 + 2 * p])
        )
        if (command != command[0]).any() or 300 % period or 600 % period:
            raise ValueError(f"{directory} is not one command at a period that divides the window")
        commands.append(command[0])
        targets.append(target[300 : 300 + period])
        references.append(angle[SETTLED].reshape(-1, period, 18).mean(0))
    if (
        len({command_key(c) for c in commands}) != len(commands)
        or len({len(t) for t in targets}) != 1
    ):
        raise ValueError("A bank needs one replay per command and one shared period")
    return np.array(commands), np.array(targets), np.array(references)


class CycleController:
    """Joint targets for one robot from the command and the measured joint angles.

    A changed command starts a cosine ramp from the last target sent to the new source: the
    command's cycle, or the neutral pose for the zero command. A walk that starts from the neutral
    pose starts at phase 0, as the native replays did.

    The servo holds a joint by the gap between its target and its angle, so a target leads its
    reference angle by a deflection. ``learning_rate`` above zero fits one gain per joint to that
    deflection: once per period the gain moves by the rate times the least-squares share of the
    tracking gap that the deflection explains. A motor with 0.6 of the nominal stiffness settles
    near a gain of 1/0.6 and follows the reference again. A gain of one sends the stored targets.
    """

    def __init__(
        self,
        commands,
        targets,
        references,
        neutral,
        *,
        learning_rate=0.0,
        gain_max=2.0,
        ramp_controls=25,
    ):
        self.index = {command_key(command): i for i, command in enumerate(commands)}
        self.targets = np.asarray(targets, float)
        self.deflections = self.targets - np.asarray(references, float)
        self.neutral = np.asarray(neutral, float)
        self.period = self.targets.shape[1]
        self.rate, self.limit, self.ramp = float(learning_rate), float(gain_max), ramp_controls
        self.reset()

    def reset(self):
        self.key, self.phase, self.since = None, 0, self.ramp
        self.start = self.sent = self.neutral.copy()
        self.gain, self.fit = np.ones(18), np.zeros((2, 18))

    def __call__(self, command, joint_angle):
        key = command_key(command)
        key = None if not any(key) else key
        if key is not None and key not in self.index:
            raise KeyError(f"The bank holds no cycle for command {key}")
        if key != self.key:
            if self.key is None:
                self.phase = 0
            self.key, self.start, self.since = key, self.sent, 0
        if key is None:
            source = self.neutral
        else:
            targets, deflections = self.targets[self.index[key]], self.deflections[self.index[key]]
            previous = (self.phase - 1) % self.period
            # The measured angle follows the previous control; a ramp is not the cycle, so it teaches nothing.
            if self.rate and self.since > self.ramp:
                gap = targets[previous] - deflections[previous] - np.asarray(joint_angle, float)
                self.fit += gap * deflections[previous], deflections[previous] ** 2
                if self.phase == 0:
                    step = self.rate * self.fit[0] / np.maximum(self.fit[1], 1e-9)
                    self.gain = np.clip(self.gain + step, 1 / self.limit, self.limit)
                    self.fit[:] = 0
            source = targets[self.phase] + (self.gain - 1) * deflections[self.phase]
            self.phase = (self.phase + 1) % self.period
        self.sent = startup_target(source, self.start, self.since, self.ramp)
        self.since += 1
        return self.sent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--replays",
        type=Path,
        nargs="+",
        required=True,
        help="Native replay evaluation directories.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New .npz file.")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new output file")
    commands, targets, references = bank_from_replays(args.replays)
    np.savez(args.output, commands=commands, targets=targets, references=references)
    print(f"{len(commands)} commands, period {targets.shape[1]} controls")


if __name__ == "__main__":
    main()
