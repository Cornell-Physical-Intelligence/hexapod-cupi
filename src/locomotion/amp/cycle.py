"""Walk from stored optimized cycles: pick the cycle by command, ramp between sources, correct joint tracking.

A bank holds, for each command, one period of motor targets and the joint angles that the nominal
robot reached under them in native replay. The controller reads the command and the 18 measured
joint angles, the sensors a robot has. It solves nothing at run time.

    uv run python -m locomotion.amp.cycle --replays REPLAY_DIR... --output bank.npz
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from locomotion.amp.replay_native import startup_target

# One period of settled native replay: the recorded controls 300 to 899 hold ten periods of 60.
SETTLED = slice(300, 900)
ADMISSION = (
    Path(__file__).resolve().parents[1] / "priors/datasets/amp_demonstrations_001/manifest.json"
)


def command_key(command):
    return tuple(round(float(value), 4) + 0.0 for value in command)


def bank_from_replays(directories):
    """Commands [n, 3], targets [n, period, 18] and reference joint angles [n, period, 18] in rad.

    If any directory has a native replay receipt, all inputs must be admitted phase-zero traces.
    Directories containing only numeric traces have unverifiable phase. Row ``r`` holds the target
    sent at phase ``r`` and the joint angle measured after that control.
    """
    directories = tuple(map(Path, directories))
    if any(
        (directory / "declaration.json").exists()
        or (directory / "report.json").exists()
        or (directory / "native400hz").exists()
        or (directory.parent / "state.json").exists()
        for directory in directories
    ):
        arrays, _ = _admitted_bank(directories)
        return arrays
    paths = (directory / "control_trace.npz" for directory in directories)
    return _bank_from_traces((path, path.read_bytes()) for path in paths)


def _admitted_bank(directories):
    admission_data = ADMISSION.read_bytes()
    admission = json.loads(admission_data)
    phase_zero = {clip["trace_sha256"] for clip in admission["clips"] if clip["start_phase"] == 0}
    trace_hashes = []

    def admitted_traces():
        for directory in directories:
            path = Path(directory) / "control_trace.npz"
            data = path.read_bytes()
            trace_hash = hashlib.sha256(data).hexdigest()
            if not admission["admitted"] or trace_hash not in phase_zero:
                raise ValueError(f"{directory} is not an admitted phase-zero replay")
            trace_hashes.append(trace_hash)
            yield path, data

    arrays = _bank_from_traces(admitted_traces())
    return arrays, {
        "trace_sha256": trace_hashes,
        "admission_sha256": hashlib.sha256(admission_data).hexdigest(),
        "model_sha256": admission["model_sha256"],
    }


def _bank_from_traces(traces):
    """Extract numeric cycles. Trace arrays alone cannot establish their start phase."""
    commands, targets, references = [], [], []
    length = SETTLED.stop - SETTLED.start
    for path, data in traces:
        with np.load(io.BytesIO(data), allow_pickle=False) as trace:
            command = trace["command"]
            target = trace["joint_target_rad"]
            angle = trace["joint_position_rad"]
        if (
            command.ndim != 3
            or command.shape[1:] != (1, 3)
            or len(command) < SETTLED.stop
            or target.shape != (len(command), 1, 18)
            or angle.shape != target.shape
            or not all(np.isfinite(value).all() for value in (command, target, angle))
            or (command != command[0]).any()
        ):
            raise ValueError(f"{path} needs a finite single-robot trace with one constant command")
        window = target[SETTLED, 0]
        period = next(
            (
                p
                for p in range(20, 151)
                if SETTLED.start % p == 0
                and length % p == 0
                and np.array_equal(window[p:], window[:-p])
            ),
            None,
        )
        if period is None:
            raise ValueError(f"{path} has no complete repeated target cycle in the settled window")
        commands.append(command[0, 0])
        targets.append(window[:period])
        references.append(angle[SETTLED, 0].reshape(-1, period, 18).mean(0))
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
        action_delay=0,
    ):
        commands = np.asarray(commands, float)
        self.targets = np.asarray(targets, float)
        self.references = np.asarray(references, float)
        self.neutral = np.asarray(neutral, float)
        if (
            commands.ndim != 2
            or commands.shape[1] != 3
            or len(commands) == 0
            or self.targets.ndim != 3
            or self.targets.shape[0] != len(commands)
            or self.targets.shape[1] == 0
            or self.targets.shape[2] != 18
            or self.references.shape != self.targets.shape
            or self.neutral.shape != (18,)
            or not all(
                np.isfinite(value).all()
                for value in (commands, self.targets, self.references, self.neutral)
            )
        ):
            raise ValueError(
                "A cycle bank needs finite commands [n, 3] and joint cycles [n, period, 18]"
            )
        self.index = {command_key(command): i for i, command in enumerate(commands)}
        if len(self.index) != len(commands):
            raise ValueError("A cycle bank needs distinct commands")
        self.deflections = self.targets - self.references
        self.period = self.targets.shape[1]
        self.rate, self.limit, self.ramp = float(learning_rate), float(gain_max), ramp_controls
        if not np.isfinite(self.rate) or self.rate < 0:
            raise ValueError("Learning rate must be finite and nonnegative")
        if not np.isfinite(self.limit) or self.limit < 1:
            raise ValueError("Gain limit must be finite and at least one")
        if type(action_delay) is not int or action_delay < 0:
            raise ValueError("Action delay counts whole controls from zero")
        self.delay = action_delay
        self.reset()

    def reset(self):
        self.key, self.phase, self.since = None, 0, self.ramp
        self.start = self.sent = self.neutral.copy()
        self.gain, self.fit = np.ones(18), np.zeros((2, 18))
        self.fit_samples = 0

    def __call__(self, command, joint_angle):
        key = command_key(command)
        key = None if not any(key) else key
        if key is not None and key not in self.index:
            raise KeyError(f"The bank holds no cycle for command {key}")
        if key != self.key:
            if self.key is None:
                self.phase = 0
            self.key, self.start, self.since = key, self.sent, 0
            self.fit[:] = 0
            self.fit_samples = 0
        if key is None:
            source = self.neutral
        else:
            index = self.index[key]
            targets, references, deflections = (
                self.targets[index],
                self.references[index],
                self.deflections[index],
            )
            # Skip prior commands and partial ramps. The sample follows the previous applied control.
            if self.rate and self.since >= self.delay + max(1, self.ramp):
                previous = (self.phase - 1 - self.delay) % self.period
                gap = references[previous] - np.asarray(joint_angle, float)
                self.fit += gap * deflections[previous], deflections[previous] ** 2
                self.fit_samples += 1
                if self.fit_samples == self.period:
                    step = self.rate * self.fit[0] / np.maximum(self.fit[1], 1e-9)
                    self.gain = np.clip(self.gain + step, 1 / self.limit, self.limit)
                    self.fit[:] = 0
                    self.fit_samples = 0
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
    output = args.output if args.output.suffix == ".npz" else Path(f"{args.output}.npz")
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new output file")
    try:
        arrays, provenance = _admitted_bank(args.replays)
    except ValueError as error:
        parser.error(str(error))
    commands, targets, references = arrays
    with output.open("xb") as stream:
        np.savez(stream, commands=commands, targets=targets, references=references, **provenance)
    print(f"{len(commands)} commands, period {targets.shape[1]} controls")


if __name__ == "__main__":
    main()
