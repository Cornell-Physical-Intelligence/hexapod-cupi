"""Replay the pre-change fixed-action protocol: ``--randomization none`` must reproduce the Phase 0 oracle bit for bit.

The oracles were captured at main 7478fca8 before any Step 7 physics edit; their full records stay outside the
checkout. ``fixtures/`` holds their state rows and protocol records. The test regenerates the actions from the
recorded seed and compares every state row, including the selected reset of replica 1 after control 40.

NumPy's bundled OpenBLAS picks its kernel by CPU, and the kernels round differently, so each oracle names the
kernel it was captured with and the replay runs in a subprocess pinned to it. ``step7_baseline_20261013_001``
used the AVX-512 SkylakeX kernel; ``step7_baseline_haswell_20261027_001`` used the AVX2 Haswell kernel, which
every x86-64 host in use can run.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).parent / "fixtures"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
ORIGINAL = "step7_baseline_20261013_001"
HASWELL = "step7_baseline_haswell_20261027_001"
# The extract names the packages before they moved under src/.
MOVED = {"locomotion/": "src/locomotion/"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def cpu_flags():
    """CPU feature flags from /proc/cpuinfo; empty where the file is missing."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    return next(
        (
            set(line.split(":", 1)[1].split())
            for line in text.splitlines()
            if line.startswith("flags")
        ),
        set(),
    )


def replay(protocol):
    """Run the capture protocol: one state row after each control and after each reset call."""
    import torch

    from locomotion.surrogate.env import make_env

    reset = protocol["selected_reset"]
    torch.set_num_threads(protocol["torch_threads"])
    env = make_env(protocol["num_envs"], threads=protocol["threads"], seed=protocol["env_seed"])
    generator = torch.Generator().manual_seed(protocol["action_seed"])
    actions, states, resets = [], [], []
    try:
        env.reset()
        states.append(env._state.copy())
        for control in range(protocol["controls"]):
            action = protocol["action_std"] * torch.randn(
                protocol["num_envs"], 18, generator=generator
            )
            output = env.step(action)
            actions.append(action.numpy().copy())
            states.append(env._state.copy())
            done = output["terminated"] | output["truncated"]
            rows = sorted(
                set(done.nonzero().flatten().tolist())
                | ({reset["row"]} if control == reset["after_control"] else set())
            )
            if rows:
                env.reset(rows)
                resets.append({"after_control": control, "rows": rows})
                states.append(env._state.copy())
        counters = {
            "physics_steps": env.physics_steps,
            "speed_clamp_events": env.speed_clamp_events,
            "touchdown_events": env.touchdown_events,
            "overshoot_events": env.overshoot_events,
            "peak_joint_speed": env.peak_joint_speed,
        }
    finally:
        env.close()
    return np.stack(actions), np.stack(states), resets, counters


def replay_pinned(record, coretype):
    """Replay in a fresh interpreter whose OpenBLAS uses ``coretype`` on one thread."""
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        env = {**os.environ, "OPENBLAS_CORETYPE": coretype, "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run(
            [sys.executable, __file__, "--replay", json.dumps(record["protocol"]), str(output)],
            env=env,
            check=True,
        )
        arrays = np.load(output / "replay.npz")
        meta = json.loads((output / "replay.json").read_text())
        return arrays["actions"], arrays["states"], meta["resets"], meta["counters"]


class OracleTests(unittest.TestCase):
    def check(self, name, coretype):
        record = extract(name)
        expected = np.load(FIXTURES / f"{name}.npz")["state_after_each_control_or_reset"]
        recorded = record["arrays"]["state_after_each_control_or_reset"]
        self.assertEqual(
            sha(np.ascontiguousarray(expected).tobytes()), recorded["sha256"], "fixture changed"
        )
        self.assertEqual(list(expected.shape), recorded["shape"])
        actions, states, resets, counters = replay_pinned(record, coretype)
        self.assertEqual(
            sha(np.ascontiguousarray(actions).tobytes()), record["arrays"]["actions"]["sha256"]
        )
        self.assertEqual(resets, record["resets"])
        self.assertEqual(states.shape, expected.shape)
        differing = np.argwhere(states != expected)
        self.assertEqual(
            len(differing),
            0,
            f"{len(differing)} state values differ; first at (row, replica, column) "
            f"{differing[0].tolist() if len(differing) else None}",
        )
        self.assertEqual(counters, record["counters"])

    @unittest.skipUnless(platform.machine() in ("x86_64", "AMD64"), "OpenBLAS kernels are x86-64")
    def test_none_reproduces_the_pre_change_trajectory_on_the_haswell_kernel(self):
        self.assertEqual(extract(HASWELL)["openblas"]["coretype"], "Haswell")
        self.check(HASWELL, "Haswell")

    @unittest.skipUnless("avx512f" in cpu_flags(), "the SkylakeX kernel needs AVX-512")
    def test_none_reproduces_the_original_oracle_on_the_skylakex_kernel(self):
        self.check(ORIGINAL, "SkylakeX")

    def test_both_oracles_share_the_capture_source_and_protocol(self):
        original, haswell = extract(ORIGINAL), extract(HASWELL)
        for key in (
            "source_commit",
            "source_files",
            "mjcf_sha256",
            "protocol",
            "resets",
            "counters",
        ):
            with self.subTest(key=key):
                self.assertEqual(haswell[key], original[key])


class PreservedBytesTests(unittest.TestCase):
    def test_surrogate_records_and_the_approved_model_keep_their_bytes(self):
        for name in (
            "locomotion/surrogate/records/self_check.json",
            "locomotion/surrogate/records/equivalence.json",
            "robot/active_model.json",
        ):
            with self.subTest(file=name):
                current = next(
                    (new + name[len(old) :] for old, new in MOVED.items() if name.startswith(old)),
                    name,
                )
                self.assertEqual(
                    sha((ROOT / current).read_bytes()), extract(ORIGINAL)["source_files"][name]
                )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--replay"]:
        protocol, output = json.loads(sys.argv[2]), Path(sys.argv[3])
        actions, states, resets, counters = replay(protocol)
        np.savez(output / "replay.npz", actions=actions, states=states)
        (output / "replay.json").write_text(json.dumps({"resets": resets, "counters": counters}))
    else:
        unittest.main()
