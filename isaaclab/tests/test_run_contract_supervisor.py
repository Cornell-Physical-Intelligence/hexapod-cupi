"""Contract tests for the run contract and the attempt-aware startup supervisor.

The supervisor exists because four attempts were lost to the pre-AppReady stall
class recorded in ``docs/incidents/2026-08-26-preappready-stall.md``. Its rules
come from ``docs/OPERATIONS.md`` §6, and every one of them is a pure decision
over injected inputs, so the whole state machine and the whole retry truth table
are exercised here against a fake clock, a fake diagnostics capture, and a fake
artifact writer. No process is started and no file is written.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_train"
if str(PACKAGE_DIR) not in sys.path:  # mirrors the isaaclab/hexapod_rl bootstrap
    sys.path.insert(0, str(PACKAGE_DIR))

from hexapod_train import supervisor  # noqa: E402
from hexapod_train.contract import (  # noqa: E402
    AttemptOutcome,
    Milestone,
    ParentPin,
    Termination,
    verify_parent,
)


CONFIG_LINE = "[INFO] Loading user config from /workspace/hexapod/isaaclab\n"
APP_READY_LINE = "[ISAACLAB] AppLauncher initialization complete\n"
PPO_LINE = "Learning iteration 0/12\n"

RETRY_MOMENT = datetime(2026, 8, 27, 9, 15, 0, tzinfo=timezone.utc)


class RecordingCapture:
    """Stands in for the atomic diagnostics bundle."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def __call__(self, reason: str, elapsed: float) -> str:
        self.calls.append((reason, elapsed))
        return f"diagnostics/{len(self.calls) - 1:02d}"


class FakeWriter:
    """In-memory ``ArtifactWriter``: records paths and bodies, touches nothing."""

    def __init__(self) -> None:
        self.directories: list[str] = []
        self.files: dict[str, str] = {}

    def makedirs(self, path: Path) -> None:
        self.directories.append(str(path))

    def write_text(self, path: Path, text: str) -> None:
        self.files[str(path)] = text


def machine(capture: RecordingCapture | None = None) -> supervisor.MilestoneMachine:
    return supervisor.MilestoneMachine(
        capture_diagnostics=capture, started_at=0.0
    )


def stalled_attempt(**overrides) -> AttemptOutcome:
    """A first attempt that stalled before AppReady and cleaned up cleanly."""
    defaults = dict(
        label="accel_probe21_bilateral_seed99_20260827T090000Z",
        attempt_index=0,
        classification=Termination.PRE_APPREADY_STALL,
        reached=Milestone.LAUNCHED,
        exit_status=124,
        user_signal=None,
        docker_status="124",
        cleanup_required=True,
        cleanup_status="stopped",
        run_directory_exists=False,
        checkpoint_exists=False,
        parent_unchanged=True,
        traceback_seen=False,
        oom_seen=False,
    )
    defaults.update(overrides)
    return AttemptOutcome(**defaults)


class MilestoneMachineTests(unittest.TestCase):
    def test_healthy_startup_reaches_app_ready_and_training(self) -> None:
        subject = machine()
        subject.observe(CONFIG_LINE, 2.0)
        self.assertEqual(subject.state, Milestone.CONFIG_LOADED)
        subject.observe(APP_READY_LINE, 12.0)
        self.assertEqual(subject.state, Milestone.APP_READY)
        subject.observe(PPO_LINE, 20.0)
        self.assertEqual(subject.state, Milestone.TRAINING)
        self.assertIsNone(subject.tick(30.0))
        self.assertIsNone(subject.stall)

    def test_config_marker_after_the_45_second_budget_is_a_pre_appready_stall(self) -> None:
        subject = machine()
        subject.observe(CONFIG_LINE, 46.0)
        self.assertEqual(subject.state, Milestone.CONFIG_LOADED)
        self.assertIsNotNone(subject.stall)
        self.assertEqual(subject.stall.termination, Termination.PRE_APPREADY_STALL)
        self.assertIn("45", subject.stall.reason)

    def test_absent_config_marker_stalls_at_its_budget(self) -> None:
        subject = machine()
        self.assertIsNone(subject.tick(44.0))
        stall = subject.tick(45.5)
        self.assertIsNotNone(stall)
        self.assertEqual(stall.termination, Termination.PRE_APPREADY_STALL)
        self.assertIn("Loading user config", stall.reason)

    def test_app_ready_at_89_seconds_proceeds(self) -> None:
        subject = machine()
        subject.observe(CONFIG_LINE, 2.0)
        subject.observe(APP_READY_LINE, 89.0)
        self.assertIsNone(subject.stall)
        self.assertIsNone(subject.tick(89.0))
        self.assertEqual(subject.state, Milestone.APP_READY)

    def test_app_ready_after_90_seconds_is_a_pre_appready_stall(self) -> None:
        subject = machine()
        subject.observe(CONFIG_LINE, 2.0)
        subject.observe(APP_READY_LINE, 91.0)
        self.assertIsNotNone(subject.stall)
        self.assertEqual(subject.stall.termination, Termination.PRE_APPREADY_STALL)

    def test_absent_app_ready_marker_stalls_at_its_budget(self) -> None:
        subject = machine()
        subject.observe(CONFIG_LINE, 2.0)
        self.assertIsNone(subject.tick(89.0))
        stall = subject.tick(90.5)
        self.assertEqual(stall.termination, Termination.PRE_APPREADY_STALL)
        self.assertIn("AppLauncher initialization complete", stall.reason)

    def test_overall_cap_after_app_ready_is_a_post_appready_failure(self) -> None:
        subject = machine()
        subject.observe(CONFIG_LINE, 2.0)
        subject.observe(APP_READY_LINE, 12.0)
        subject.observe(PPO_LINE, 20.0)
        self.assertEqual(subject.tick(421.0).termination, Termination.POST_APPREADY_FAILURE)

    def test_overall_cap_before_app_ready_is_a_pre_appready_stall(self) -> None:
        # The 45 s and 90 s budgets normally fire long before the 420 s cap, so
        # this branch is exercised with the earlier budgets relaxed.
        subject = supervisor.MilestoneMachine(
            deadlines=supervisor.MilestoneDeadlines(
                config_loaded=500.0, app_ready=600.0, overall=420.0, no_progress=30.0
            ),
            started_at=0.0,
        )
        stall = subject.tick(421.0)
        self.assertEqual(stall.termination, Termination.PRE_APPREADY_STALL)
        self.assertIn("420s cap", stall.reason)

    def test_thirty_seconds_without_progress_captures_diagnostics_once_per_stall(self) -> None:
        capture = RecordingCapture()
        subject = machine(capture)
        subject.observe(CONFIG_LINE, 2.0)
        self.assertIsNone(subject.tick(20.0))
        self.assertEqual(len(capture.calls), 0)
        subject.tick(32.0)
        subject.tick(35.0)
        subject.tick(40.0)
        self.assertEqual(len(capture.calls), 1, "one capture per contiguous stall")
        # Progress re-arms the trigger; the next silent window captures again.
        subject.observe(APP_READY_LINE, 41.0)
        subject.tick(60.0)
        self.assertEqual(len(capture.calls), 1)
        subject.tick(72.0)
        self.assertEqual(len(capture.calls), 2)
        self.assertEqual(subject.captures, ("diagnostics/00", "diagnostics/01"))

    def test_captures_stop_once_training_is_underway(self) -> None:
        capture = RecordingCapture()
        subject = machine(capture)
        subject.observe(CONFIG_LINE, 2.0)
        subject.observe(APP_READY_LINE, 12.0)
        subject.observe(PPO_LINE, 20.0)
        subject.tick(300.0)
        self.assertEqual(capture.calls, [])

    def test_traceback_and_oom_are_observed_not_inferred(self) -> None:
        subject = machine()
        subject.observe("Traceback (most recent call last):\n", 5.0)
        self.assertTrue(subject.traceback_seen)
        self.assertFalse(subject.oom_seen)
        other = machine()
        other.observe("CUDA error: out of memory\n", 5.0)
        self.assertTrue(other.oom_seen)


class ClassificationTests(unittest.TestCase):
    def test_terminal_classes_follow_the_spec_precedence(self) -> None:
        cases = (
            (
                dict(reached=Milestone.LAUNCHED, user_signal="INT"),
                Termination.USER_SIGNAL,
            ),
            (
                dict(reached=Milestone.DONE, exit_status=0, completed=True),
                Termination.SUCCESS,
            ),
            (
                dict(
                    reached=Milestone.LAUNCHED,
                    cleanup_required=True,
                    cleanup_status="identity-mismatch",
                ),
                Termination.CLEANUP_UNVERIFIED,
            ),
            (
                dict(reached=Milestone.LAUNCHED, docker_status=125),
                Termination.DOCKER_ERROR,
            ),
            (
                dict(reached=Milestone.TRAINING, exit_status=1),
                Termination.POST_APPREADY_FAILURE,
            ),
            (
                dict(reached=Milestone.LAUNCHED, exit_status=124),
                Termination.PRE_APPREADY_STALL,
            ),
        )
        for keywords, expected in cases:
            with self.subTest(keywords=keywords):
                self.assertEqual(supervisor.classify_outcome(**keywords), expected)

    def test_proven_cleanup_states_match_the_launcher_vocabulary(self) -> None:
        for status in ("stopped", "container-absent", "already-stopped"):
            with self.subTest(status=status):
                self.assertEqual(
                    supervisor.classify_outcome(
                        reached=Milestone.LAUNCHED,
                        cleanup_required=True,
                        cleanup_status=status,
                        exit_status=124,
                    ),
                    Termination.PRE_APPREADY_STALL,
                )


class RetryTruthTableTests(unittest.TestCase):
    """Spec item 7, row by row. The default row is the only one that retries."""

    def test_the_one_permitted_retry(self) -> None:
        decision = supervisor.retry_decision(stalled_attempt(), moment=RETRY_MOMENT)
        self.assertTrue(decision.retry)
        self.assertEqual(decision.refusals, ())
        self.assertEqual(
            decision.next_label,
            "accel_probe21_bilateral_seed99_retry1_20260827T091500Z",
        )

    def test_a_retry_is_never_itself_retried(self) -> None:
        decision = supervisor.retry_decision(
            stalled_attempt(
                label="accel_probe21_bilateral_seed99_retry1_20260827T091500Z",
                attempt_index=1,
            ),
            moment=RETRY_MOMENT,
        )
        self.assertFalse(decision.retry)
        self.assertIn("last permitted attempt", decision.reason)

    def test_every_never_retry_rule(self) -> None:
        cases = (
            ("user signal", dict(classification=Termination.USER_SIGNAL, user_signal="INT")),
            ("docker error", dict(classification=Termination.DOCKER_ERROR)),
            ("post-AppReady", dict(classification=Termination.POST_APPREADY_FAILURE)),
            ("cleanup class", dict(classification=Termination.CLEANUP_UNVERIFIED)),
            ("success", dict(classification=Termination.SUCCESS)),
            ("unverified cleanup", dict(cleanup_status="identity-mismatch")),
            ("inspect failure", dict(cleanup_status="inspect-failed")),
            ("run directory exists", dict(run_directory_exists=True)),
            ("checkpoint exists", dict(checkpoint_exists=True)),
            ("parent not proven", dict(parent_unchanged=False)),
            ("traceback", dict(traceback_seen=True)),
            ("oom", dict(oom_seen=True)),
        )
        for name, overrides in cases:
            with self.subTest(case=name):
                decision = supervisor.retry_decision(
                    stalled_attempt(**overrides), moment=RETRY_MOMENT
                )
                self.assertFalse(decision.retry, f"{name} must never retry")
                self.assertIsNone(decision.next_label)
                self.assertTrue(decision.refusals)

    def test_refusal_reasons_are_recorded_not_summarised_away(self) -> None:
        decision = supervisor.retry_decision(
            stalled_attempt(checkpoint_exists=True, oom_seen=True), moment=RETRY_MOMENT
        )
        self.assertEqual(len(decision.refusals), 2)
        self.assertIn("checkpoint exists", decision.reason)
        self.assertIn("OOM", decision.reason)

    def test_a_decision_without_a_moment_still_decides(self) -> None:
        decision = supervisor.retry_decision(stalled_attempt())
        self.assertTrue(decision.retry)
        self.assertIsNone(decision.next_label)


class AttemptArtifactTests(unittest.TestCase):
    def test_attempts_retry_decision_and_root_outcome_land_where_specified(self) -> None:
        writer = FakeWriter()
        recorder = supervisor.AttemptRecorder("/batches/label", writer)
        first = stalled_attempt(diagnostics=("diagnostics/00",))
        decision = supervisor.retry_decision(first, moment=RETRY_MOMENT)
        second = stalled_attempt(label=decision.next_label, attempt_index=1)

        recorder.record_attempt(first)
        recorder.record_attempt(second)
        recorder.record_retry_decision(decision)
        recorder.record_root_outcome([first, second], decision)

        self.assertIn("/batches/label/attempts/00/outcome", writer.files)
        self.assertIn("/batches/label/attempts/01/outcome", writer.files)
        self.assertIn("/batches/label/retry.decision", writer.files)
        self.assertIn("/batches/label/outcome", writer.files)
        self.assertIn(
            "/batches/label/attempts/00", writer.directories
        )
        self.assertIn(
            "/batches/label/attempts/01", writer.directories
        )

    def test_the_root_outcome_states_that_a_retry_occurred(self) -> None:
        writer = FakeWriter()
        recorder = supervisor.AttemptRecorder("/batches/label", writer)
        first = stalled_attempt()
        decision = supervisor.retry_decision(first, moment=RETRY_MOMENT)
        second = stalled_attempt(
            label=decision.next_label,
            attempt_index=1,
            classification=Termination.SUCCESS,
            reached=Milestone.DONE,
            exit_status=0,
            cleanup_required=False,
            cleanup_status="not-required",
            run_directory_exists=True,
            checkpoint_exists=True,
        )
        recorder.record_root_outcome([first, second], decision)
        body = writer.files["/batches/label/outcome"]
        self.assertIn("attempts=2\n", body)
        self.assertIn("retry_performed=true\n", body)
        self.assertIn("overall=succeeded\n", body)
        self.assertIn("final_classification=SUCCESS\n", body)
        self.assertIn(first.label, body)

    def test_a_single_attempt_root_outcome_records_no_retry(self) -> None:
        writer = FakeWriter()
        recorder = supervisor.AttemptRecorder("/batches/label", writer)
        outcome = stalled_attempt(cleanup_status="identity-mismatch")
        decision = supervisor.retry_decision(outcome, moment=RETRY_MOMENT)
        recorder.record_root_outcome([outcome], decision)
        body = writer.files["/batches/label/outcome"]
        self.assertIn("attempts=1\n", body)
        self.assertIn("retry_performed=false\n", body)
        self.assertIn("retry_allowed=false\n", body)
        self.assertIn("overall=failed\n", body)

    def test_the_attempt_artifact_carries_the_retry_evidence(self) -> None:
        writer = FakeWriter()
        recorder = supervisor.AttemptRecorder("/batches/label", writer)
        outcome = stalled_attempt(diagnostics=("diagnostics/00",))
        recorder.record_attempt(outcome)
        body = writer.files["/batches/label/attempts/00/outcome"]
        for expected in (
            "classification=PRE_APPREADY_STALL\n",
            "cleanup_proven=true\n",
            "run_directory_exists=false\n",
            "checkpoint_exists=false\n",
            "parent_unchanged=true\n",
            "diagnostics[0]=diagnostics/00\n",
        ):
            self.assertIn(expected, body)


class ParentPinTests(unittest.TestCase):
    def test_verify_parent_hashes_through_the_injected_reader(self) -> None:
        pin = ParentPin(
            run_dir="2026-08-25_23-44-14_accel_scale4",
            checkpoint="model_2.pt",
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        empty = verify_parent(pin.path, pin.sha256, lambda _path: [])
        self.assertTrue(empty.matches)
        changed = verify_parent(pin.path, pin.sha256, lambda _path: [b"changed"])
        self.assertFalse(changed.matches)
        self.assertNotEqual(changed.actual_sha256, changed.expected_sha256)
        self.assertTrue(pin.path.endswith("/model_2.pt"))


class ShardSequencingTests(unittest.TestCase):
    """Spec item 10: shard B is admissible only after shard A reaches AppReady."""

    def test_correct_ordering_is_admissible(self) -> None:
        sequencer = supervisor.ShardSequencer()
        sequencer.observe("shard_a", "starting\n", 1.0)
        sequencer.observe("shard_a", APP_READY_LINE, 12.0)
        sequencer.observe("shard_b", "starting\n", 13.0)
        self.assertTrue(sequencer.admissible)
        self.assertIsNone(sequencer.violation())

    def test_shard_b_before_shard_a_app_ready_is_a_violation(self) -> None:
        sequencer = supervisor.ShardSequencer()
        sequencer.observe("shard_b", "starting\n", 3.0)
        sequencer.observe("shard_a", APP_READY_LINE, 12.0)
        self.assertFalse(sequencer.admissible)
        self.assertIn("before shard A reached AppReady", sequencer.violation())

    def test_shard_a_never_reaching_app_ready_is_never_admissible(self) -> None:
        sequencer = supervisor.ShardSequencer()
        sequencer.observe("shard_a", "starting\n", 1.0)
        self.assertFalse(sequencer.admissible)
        self.assertIn("never reached AppReady", sequencer.violation())


if __name__ == "__main__":
    unittest.main()
