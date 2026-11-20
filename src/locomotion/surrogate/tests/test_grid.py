"""Check the fixed-condition grid on the surrogate: the cell environment, the cell evaluation and the sweep."""

import contextlib
import functools
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import locomotion.evaluation.requirements as requirements
from locomotion import randomization as rnd
from locomotion.surrogate import evaluate, sweep
from locomotion.surrogate.env import (
    DEFAULT_GEOMETRY,
    DEFAULT_MODEL,
    NEVER,
    SurrogateConfig,
    SurrogateEnv,
    make_env,
    sha,
)

GRID = "table2_v1_grid"
NOMINAL = "mu1_payload0.000kg_strength1"
CORNER = "mu0.1_payload1.464kg_strength0.8"


def cell_env(cell, num_envs=2):
    cfg = SurrogateConfig(num_envs=num_envs, seed=20260914)
    return SurrogateEnv(
        cfg,
        None,
        DEFAULT_MODEL,
        DEFAULT_GEOMETRY,
        None,
        threads=0,
        condition=rnd.cell_draw(GRID, cell),
    )


def trajectory(env):
    """States after 40 fixed-seed controls, with a selected reset of replica 1 after control 15."""
    generator = torch.Generator().manual_seed(7)
    states = []
    try:
        for control in range(40):
            env.step(0.3 * torch.randn(env.num_envs, 18, generator=generator))
            states.append(env._state.copy())
            if control == 15:
                env.reset([1])
                states.append(env._state.copy())
    finally:
        env.close()
    return np.stack(states)


class CellEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_the_nominal_cell_steps_as_the_nominal_robot_bit_for_bit(self):
        expected = trajectory(make_env(2, threads=0, seed=20260914))
        cell = trajectory(cell_env(NOMINAL))
        self.assertEqual(int((cell != expected).sum()), 0)

    def test_a_cell_holds_across_resets_and_never_pushes(self):
        env = cell_env(CORNER)
        draw = rnd.cell_draw(GRID, CORNER)
        try:
            for _ in range(3):
                env.step(0.2 * torch.ones(2, 18))
            env.reset([0])
            env.step(torch.zeros(2, 18))
            base = env._base_id
            for i in range(2):
                model = env._models[i]
                self.assertTrue(np.all(model.geom_friction[env._friction_geoms, 0] == 0.1))
                self.assertAlmostEqual(
                    float(model.body_mass[base]),
                    float(env._nominal["body_mass"][base]) + draw.payload_mass_kg,
                    places=12,
                )
                self.assertTrue(torch.all(env._strength[i] == 0.8))
                self.assertEqual(float(env._kp[i]), rnd.KP)
                self.assertEqual(env._friction[i], 0.1)
            self.assertTrue(np.all(env._push_start == NEVER) and np.all(env._push_end == NEVER))
            self.assertEqual(float(env.external_wrench_body.abs().sum()), 0.0)
            column = env.mj_model.nu + 6 * base
            self.assertEqual(float(np.abs(env._control[:, 0, column : column + 6]).sum()), 0.0)
            self.assertEqual(env.native_readback["condition"], draw.as_json())
        finally:
            env.close()

    def test_a_condition_replaces_the_draws_and_refuses_a_declaration(self):
        with self.assertRaisesRegex(ValueError, "fixed condition"):
            SurrogateEnv(
                SurrogateConfig(num_envs=1),
                None,
                DEFAULT_MODEL,
                DEFAULT_GEOMETRY,
                None,
                threads=0,
                randomization="table2_v1",
                condition=rnd.cell_draw(GRID, NOMINAL),
            )


class CellEvaluationTests(unittest.TestCase):
    def parse(self, directory, *extra):
        checkpoint = Path(directory) / "checkpoint.pt"
        checkpoint.with_suffix(".json").write_text(json.dumps({"identity": {"ppo_config": {}}}))
        return evaluate.build_parser().parse_args(
            ["--checkpoint", str(checkpoint), "--output", str(Path(directory) / "out"), *extra]
        )

    def test_grid_and_cell_are_checked_before_any_output(self):
        cases = [
            (("--grid", GRID), "go together"),
            (("--grid", GRID, "--cell", "mu9_payload0.000kg_strength1"), "Unknown cell"),
            (
                ("--grid", GRID, "--cell", NOMINAL, "--randomization", "table2_v1"),
                "fixes the conditions",
            ),
        ]
        for extra, message in cases:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    evaluate.evaluate_checkpoint(self.parse(directory, *extra))
                self.assertFalse((Path(directory) / "out").exists())

    def test_the_historical_consensus_keeps_its_variants_and_scope(self):
        self.assertEqual(
            evaluate.CONSENSUS_VARIANTS,
            {
                "pyramidal_cone": {"cone": "pyramidal"},
                "friction_0.9": {"friction": (0.9, 0.005, 0.0001)},
            },
        )
        result = {"pass": True, "checks": {}, "failed_checks": []}
        self.assertEqual(
            evaluate.consensus({"default": result, "pyramidal_cone": result})["scope"],
            evaluate.CONSENSUS_SCOPE,
        )
        self.assertEqual(
            evaluate.RELATIVE_FRICTION_VARIANTS,
            {"pyramidal_cone": ({"cone": "pyramidal"}, 1.0), "friction_0.9mu": ({}, 0.9)},
        )


@functools.cache
def source_identity():
    return evaluate.source_identity()


def fake_summary(cell, verdict=True, failure=None, command=None):
    """A cell summary as the evaluator writes it, bound to the request in ``command``."""
    case = {
        "case_id": "stand",
        "pass": verdict,
        "failed_checks": [] if verdict else ["tilt_deg"],
        "checks": {"tilt_deg": {"value": 0.0, "bound": 1.7, "pass": True}},
        "contact_consensus": {"verdict": "pass" if verdict else "unresolved", "failed_checks": {}},
    }
    summary = {
        "seed": 20260917,
        "condition": {"draw": rnd.cell_draw(GRID, cell).as_json()},
        "acquisition_failure": failure,
        "cases": [case],
    }
    if command is not None:

        def option(name, default=None):
            return command[command.index(name) + 1] if name in command else default

        probes = [
            case["case_id"] for case in requirements.cases(requirements.load_requirements()[0])
        ]
        summary["checkpoint_sha256"] = sha(Path(option("--checkpoint")))
        summary["seed"] = int(option("--seed", 20260917))
        summary["selected_case_ids"] = evaluate.selected_cases(option("--cases"), probes)
        summary["identity"] = source_identity()
        summary["condition"].update(
            grid=option("--grid"), grid_sha256=rnd.grid_declaration(GRID)["sha256"], cell=cell
        )
    return summary


class SweepTests(unittest.TestCase):
    def sweep(self, directory, *extra, fail=(), acquisition=(), weights=b"weights", calls=None):
        checkpoint = Path(directory) / "checkpoint.pt"
        checkpoint.write_bytes(weights)

        def run_process(command, log):
            cell = command[command.index("--cell") + 1]
            if calls is not None:
                calls.append(cell)
            output = Path(command[command.index("--output") + 1])
            output.mkdir(parents=True)
            Path(log).write_text("log\n")
            if cell in fail:
                return 1
            failure = "RuntimeError('recurrence')" if cell in acquisition else None
            summary = fake_summary(cell, cell != CORNER, failure, command)
            (output / "surrogate_summary.json").write_text(json.dumps(summary))
            return 0

        argv = ["--checkpoint", str(checkpoint), "--output", str(Path(directory) / "sweep"), *extra]
        with (
            mock.patch.object(sweep, "run_process", run_process),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = sweep.main(argv)
        return code, json.loads((Path(directory) / "sweep/grid_summary.json").read_text())

    def test_the_sweep_runs_every_cell_once_and_counts_verdicts(self):
        with tempfile.TemporaryDirectory() as directory:
            code, summary = self.sweep(directory, "--jobs", "3")
            self.assertEqual(code, 0)
            self.assertEqual(summary["cells_complete"], 27)
            self.assertEqual(summary["grid_sha256"], rnd.grid_declaration(GRID)["sha256"])
            self.assertEqual(
                summary["verdict_counts"]["stand"],
                {"pass": 26, "fail": 0, "unresolved": 1},
            )
            corner = next(row for row in summary["cells"] if row["cell"] == CORNER)
            self.assertEqual(corner["cases"]["stand"]["verdict"], "unresolved")
            self.assertEqual((corner["foot_friction"], corner["motor_strength"]), (0.1, 0.8))

    def test_continue_keeps_finished_cells_and_sets_failed_attempts_aside(self):
        with tempfile.TemporaryDirectory() as directory:
            code, summary = self.sweep(directory, "--cells", f"{NOMINAL},{CORNER}", fail=(CORNER,))
            self.assertEqual((code, summary["cells_complete"]), (1, 1))
            with self.assertRaisesRegex(ValueError, "--continue"):
                self.sweep(directory, "--cells", NOMINAL)
            code, summary = self.sweep(directory, "--cells", f"{NOMINAL},{CORNER}", "--continue")
            self.assertEqual((code, summary["cells_complete"]), (0, 2))
            cells = Path(directory) / "sweep/cells"
            self.assertTrue((cells / f"{CORNER}.attempt_001").is_dir())
            self.assertTrue((cells / f"{CORNER}.attempt_001.log").exists())
            row = next(row for row in summary["cells"] if row["cell"] == CORNER)
            self.assertEqual(row["set_aside_attempts"], [f"{CORNER}.attempt_001"])
            self.assertNotIn("exit_code", next(r for r in summary["cells"] if r["cell"] == NOMINAL))

    def test_continue_refuses_cells_of_another_request_and_changes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _ = self.sweep(directory, "--cells", NOMINAL, "--cases", "focus")
            self.assertEqual(code, 0)
            cells = Path(directory) / "sweep/cells"
            before = (cells / NOMINAL / "surrogate_summary.json").read_bytes()
            requests = {
                "checkpoint_sha256": ((), b"other weights"),
                "selected_case_ids": (("--cases", "all"), b"weights"),
                "seed": (("--seed", "42"), b"weights"),
            }
            for key, (option, weights) in requests.items():
                with self.subTest(key):
                    calls = []
                    extra = ("--cells", NOMINAL, "--cases", "focus", "--continue", *option)
                    with self.assertRaisesRegex(ValueError, key):
                        self.sweep(directory, *extra, weights=weights, calls=calls)
                    self.assertEqual(calls, [])
                    self.assertEqual(
                        (cells / NOMINAL / "surrogate_summary.json").read_bytes(), before
                    )
                    self.assertEqual(
                        sorted(p.name for p in cells.iterdir()), [NOMINAL, f"{NOMINAL}.log"]
                    )
            # The review's case: another checkpoint, every probe and another seed at once.
            calls = []
            extra = ("--cells", NOMINAL, "--continue", "--cases", "all", "--seed", "42")
            with self.assertRaisesRegex(ValueError, "Finished cells answer another request"):
                self.sweep(directory, *extra, weights=b"other weights", calls=calls)
            self.assertEqual(calls, [])
            # A source change also refuses the resume.
            identity = {**source_identity(), "model_sha256": "0" * 64}
            with mock.patch.object(evaluate, "source_identity", return_value=identity):
                with self.assertRaisesRegex(ValueError, "identity"):
                    self.sweep(directory, "--cells", NOMINAL, "--cases", "focus", "--continue")
            # So does a changed grid declaration.
            declaration = {**rnd.grid_declaration(GRID), "sha256": "0" * 64}
            with mock.patch.object(rnd, "grid_declaration", return_value=declaration):
                with self.assertRaisesRegex(ValueError, "grid_sha256"):
                    self.sweep(directory, "--cells", NOMINAL, "--cases", "focus", "--continue")
            self.assertEqual((cells / NOMINAL / "surrogate_summary.json").read_bytes(), before)

    def test_continue_sets_an_acquisition_failure_aside_and_runs_the_cell_again(self):
        with tempfile.TemporaryDirectory() as directory:
            code, summary = self.sweep(
                directory, "--cells", f"{NOMINAL},{CORNER}", acquisition=(CORNER,)
            )
            self.assertEqual((code, summary["cells_with_acquisition_failure"]), (1, [CORNER]))
            calls = []
            code, summary = self.sweep(
                directory, "--cells", f"{NOMINAL},{CORNER}", "--continue", calls=calls
            )
            self.assertEqual(calls, [CORNER])
            self.assertEqual((code, summary["cells_complete"]), (0, 2))
            self.assertEqual(summary["cells_with_acquisition_failure"], [])
            cells = Path(directory) / "sweep/cells"
            kept = json.loads((cells / f"{CORNER}.attempt_001/surrogate_summary.json").read_text())
            self.assertEqual(kept["acquisition_failure"], "RuntimeError('recurrence')")
            self.assertTrue((cells / f"{CORNER}.attempt_001.log").exists())
            row = next(row for row in summary["cells"] if row["cell"] == CORNER)
            self.assertEqual(row["set_aside_attempts"], [f"{CORNER}.attempt_001"])

    def test_an_acquisition_failure_is_no_complete_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.write_bytes(b"weights")

            def run_process(command, log):
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True)
                cell = command[command.index("--cell") + 1]
                summary = fake_summary(cell, failure="RuntimeError('recurrence')")
                (output / "surrogate_summary.json").write_text(json.dumps(summary))
                return 0

            argv = ["--checkpoint", str(checkpoint), "--output", str(Path(directory) / "s")]
            argv += ["--cells", CORNER]
            with mock.patch.object(sweep, "run_process", run_process):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = sweep.main(argv)
            summary = json.loads((Path(directory) / "s/grid_summary.json").read_text())
            self.assertEqual((code, summary["cells_complete"]), (1, 0))
            self.assertEqual(summary["cells_with_acquisition_failure"], [CORNER])
            self.assertEqual(summary["cells"][0]["status"], "acquisition_failure")

    def test_list_prints_the_cells_in_order(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sweep.main(["--checkpoint", "unused", "--output", "unused", "--list"])
        self.assertEqual(out.getvalue().split(), [cell for cell, _ in rnd.grid_cells(GRID)])


if __name__ == "__main__":
    unittest.main()
