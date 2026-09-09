"""Contract tests for the label grammar and the resource gates.

Two invariants live here. Labels are immutable: a label is generated once,
never reused, and never rewritten, including a failed attempt's label. And the
resource gates fail closed: a producer script or any descendant of one keeps the
machine unclear even while the GPU reads idle, which is exactly the ordering the
NSVA race in ``docs/incidents/2026-08-26-preappready-stall.md`` §5 exploited.

Everything is pure. The existence check is injected and the gates run over fake
snapshots, so nothing is read from the filesystem and no process is inspected.
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

from hexapod_train import gates, labels  # noqa: E402


LOGS_ROOT = Path("/home/orionh/HEXAPOD/isaaclab/logs/probe_batches/stage2c")
MOMENT = datetime(2026, 8, 26, 17, 26, 45, tzinfo=timezone.utc)
LATER = datetime(2026, 8, 26, 17, 42, 0, tzinfo=timezone.utc)
SLUG = "accel_probe21_bilateral_long2_ref1800_seed99_slew040"

NOTHING_EXISTS = lambda _path: False  # noqa: E731 - an injected predicate


class LabelGrammarTests(unittest.TestCase):
    def test_generated_labels_round_trip(self) -> None:
        label = labels.build_label(SLUG, MOMENT)
        self.assertEqual(label, f"{SLUG}_20260826T172645Z")
        parsed = labels.parse_label(label)
        self.assertEqual(parsed.slug, SLUG)
        self.assertEqual(parsed.attempt, 0)
        self.assertFalse(parsed.is_retry)
        self.assertEqual(parsed.stamp, MOMENT)

    def test_retry_labels_round_trip_and_carry_their_attempt(self) -> None:
        first = labels.build_label(SLUG, MOMENT)
        second = labels.retry_label(first, LATER)
        self.assertEqual(second, f"{SLUG}_retry1_20260826T174200Z")
        parsed = labels.parse_label(second)
        self.assertEqual(parsed.slug, SLUG)
        self.assertEqual(parsed.attempt, 1)
        self.assertTrue(parsed.is_retry)
        # The grammar can express a second retry; policy is what forbids it.
        self.assertEqual(labels.parse_label(labels.retry_label(second, LATER)).attempt, 2)
        self.assertEqual(labels.MAX_LABEL_LENGTH, 96)

    def test_the_recorded_incident_labels_parse(self) -> None:
        recorded = {
            "accel_probe19_yaw_slew4_ref040_slew040_20260826T132100Z": 0,
            "accel_probe20_matched_zero_seed99_slew040_20260826T152600Z": 0,
            "accel_probe20_matched_zero_seed99_slew040_retry1_20260826T160100Z": 1,
            "accel_probe21_bilateral_long2_ref1800_seed99_slew040_20260826T172645Z": 0,
            "accel_probe21_bilateral_long2_ref1800_seed99_slew040_retry1_20260826T174200Z": 1,
        }
        for label, attempt in recorded.items():
            with self.subTest(label=label):
                self.assertEqual(labels.parse_label(label).attempt, attempt)

    def test_labels_outside_the_grammar_are_refused(self) -> None:
        cases = (
            "no_stamp_here",
            "accel_probe21_20260826T172645",  # missing the trailing Z
            "accel_probe21_20260231T172645Z",  # impossible date
            "-leading_dash_20260826T172645Z",
            "spaces are not allowed_20260826T172645Z",
            "..",
            "",
        )
        for label in cases:
            with self.subTest(label=label):
                with self.assertRaises(labels.LabelError):
                    labels.parse_label(label)

    def test_a_label_longer_than_the_launcher_allows_is_refused(self) -> None:
        with self.assertRaises(labels.LabelError) as raised:
            labels.build_label("x" * 90, MOMENT)
        self.assertIn("96", str(raised.exception))

    def test_generation_refuses_an_unsafe_slug(self) -> None:
        for slug in ("../escape", "has space", "", "trailing_"):
            with self.subTest(slug=slug):
                with self.assertRaises(labels.LabelError):
                    labels.build_label(slug, MOMENT)


class LabelImmutabilityTests(unittest.TestCase):
    def test_a_label_whose_batch_directory_exists_is_refused(self) -> None:
        taken = labels.build_label(SLUG, MOMENT)
        existing = labels.batch_directory(LOGS_ROOT, taken)
        with self.assertRaises(labels.LabelCollisionError) as raised:
            labels.assert_available(taken, LOGS_ROOT, lambda path: path == existing)
        self.assertIn("never reused", str(raised.exception))

    def test_a_label_already_used_in_this_run_is_refused(self) -> None:
        taken = labels.build_label(SLUG, MOMENT)
        with self.assertRaises(labels.LabelCollisionError) as raised:
            labels.assert_available(taken, LOGS_ROOT, NOTHING_EXISTS, known_labels=[taken])
        self.assertIn("immutable", str(raised.exception))

    def test_a_failed_attempt_label_is_still_taken(self) -> None:
        # Failed-attempt evidence is append-only, so a stalled attempt's label
        # is as permanently spent as a successful one's.
        failed = "accel_probe21_bilateral_long2_ref1800_seed99_slew040_20260826T172645Z"
        with self.assertRaises(labels.LabelCollisionError):
            labels.assert_available(
                failed,
                LOGS_ROOT,
                lambda path: path.name == failed,
            )
        # The retry under a fresh stamp is free.
        retry = labels.retry_label(failed, LATER)
        self.assertEqual(
            labels.assert_available(retry, LOGS_ROOT, lambda path: path.name == failed),
            retry,
        )

    def test_allocate_generates_and_proves_availability(self) -> None:
        label = labels.allocate(SLUG, MOMENT, LOGS_ROOT, NOTHING_EXISTS)
        self.assertEqual(label, f"{SLUG}_20260826T172645Z")
        self.assertEqual(
            labels.batch_directory(LOGS_ROOT, label), LOGS_ROOT / label
        )


def snapshot(**overrides) -> gates.ResourceSnapshot:
    defaults = dict(
        processes=(),
        gpu_processes=(),
        docker_containers=(),
        service_state="inactive",
        gpu_lock_free=True,
        observed_at_utc="2026-08-27T09:00:00Z",
    )
    defaults.update(overrides)
    return gates.ResourceSnapshot(**defaults)


# The exact ancestry recorded in the incident: the producer script existed
# before the preflight and spawned its GPU child afterwards.
NSVA_PRODUCER = gates.ProcessEntry(pid=215354, ppid=1, command="bash /root/nsva_dl.sh")
NSVA_GPU_CHILD = gates.ProcessEntry(
    pid=217667,
    ppid=215354,
    command=(
        "/root/cvnba/venv/bin/python sq/validate_nsva.py data/nsva "
        "--out results/nsva_validation.json"
    ),
)


class ResourceGateTests(unittest.TestCase):
    def test_a_quiet_machine_is_clear(self) -> None:
        report = gates.evaluate(snapshot())
        self.assertTrue(report.clear)
        self.assertEqual(report.failures, ())
        self.assertIn("clear=yes", report.render())

    def test_the_nsva_race_is_not_clear(self) -> None:
        report = gates.evaluate(
            snapshot(
                processes=(NSVA_PRODUCER, NSVA_GPU_CHILD),
                gpu_processes=(
                    gates.GpuProcess(pid=217667, process_name="python", used_memory_mib=1366),
                ),
            )
        )
        self.assertFalse(report.clear)
        self.assertFalse(report.check("gpu_idle").passed)
        self.assertFalse(report.check("producers_absent").passed)
        evidence = "\n".join(report.check("producers_absent").evidence)
        self.assertIn("pid=215354", evidence)
        self.assertIn("pid=217667", evidence)
        self.assertIn("nsva_dl", evidence)

    def test_a_free_gpu_does_not_clear_a_live_producer(self) -> None:
        # The rule the incident proves: the producer spawns GPU children after
        # its own start, so an idle GPU is not evidence that it has finished.
        report = gates.evaluate(snapshot(processes=(NSVA_PRODUCER,)))
        self.assertTrue(report.check("gpu_idle").passed)
        self.assertFalse(report.check("producers_absent").passed)
        self.assertFalse(report.clear)

    def test_a_grandchild_of_a_producer_is_a_descendant(self) -> None:
        shell = gates.ProcessEntry(pid=215999, ppid=215354, command="sh -c ./work")
        worker = gates.ProcessEntry(
            pid=216111, ppid=215999, command="python -m torch._inductor.compile_worker"
        )
        report = gates.evaluate(snapshot(processes=(NSVA_PRODUCER, shell, worker)))
        self.assertFalse(report.check("producers_absent").passed)
        hits = {hit.process.pid: hit for hit in gates.producer_hits(snapshot(
            processes=(NSVA_PRODUCER, shell, worker)
        ))}
        self.assertEqual(hits[215999].relation, "descendant")
        self.assertEqual(hits[215999].ancestor_pid, 215354)

    def test_every_named_producer_pattern_is_detected(self) -> None:
        commands = (
            "bash /root/nsva_dl.sh",
            "python sq/validate_nsva.py data/nsva",
            "python score_clip.py --shard 3",
            "python validate_b51.py",
            "python -m torch._inductor.compile_worker",
            "python clip_scoring_worker.py",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(gates.match_producer(command))

    def test_our_own_processes_are_not_producers(self) -> None:
        for command in (
            "/workspace/isaaclab/_isaac_sim/python.sh train_model_only_resume.py",
            "python3 evaluate_checkpoint.py --task Isaac-Velocity-Omni",
            "bash ./isaaclab/deploy/probe-stage2c-single-current-best label 99",
        ):
            with self.subTest(command=command):
                self.assertIsNone(gates.match_producer(command))

    def test_an_active_container_or_service_is_not_clear(self) -> None:
        containers = gates.evaluate(snapshot(docker_containers=("hexapod-s2c-shard_a-1",)))
        self.assertFalse(containers.check("containers_absent").passed)
        active = gates.evaluate(snapshot(service_state="active"))
        self.assertFalse(active.check("training_service_inactive").passed)

    def test_unprovable_conditions_fail_closed(self) -> None:
        unknown_service = gates.evaluate(snapshot(service_state="unknown"))
        self.assertFalse(unknown_service.check("training_service_inactive").passed)
        unprobed_lock = gates.evaluate(snapshot(gpu_lock_free=None))
        self.assertFalse(unprobed_lock.check("gpu_lock_available").passed)
        self.assertIn("could not be probed", "\n".join(
            unprobed_lock.check("gpu_lock_available").evidence
        ))
        held_lock = gates.evaluate(snapshot(gpu_lock_free=False))
        self.assertFalse(held_lock.check("gpu_lock_available").passed)


class SnapshotParsingTests(unittest.TestCase):
    def test_process_table_rows_parse_and_junk_is_dropped(self) -> None:
        parsed = gates.parse_process_table(
            "  215354     1 bash /root/nsva_dl.sh\n"
            "  217667 215354 python sq/validate_nsva.py\n"
            "header row that is not a process\n"
            "\n"
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[1].ppid, 215354)

    def test_gpu_compute_apps_parse(self) -> None:
        parsed = gates.parse_gpu_compute_apps(
            "217667, /root/cvnba/venv/bin/python, 1366\n"
        )
        self.assertEqual(parsed[0].pid, 217667)
        self.assertEqual(parsed[0].used_memory_mib, 1366)
        self.assertEqual(gates.parse_gpu_compute_apps(""), ())

    def test_docker_names_and_service_status_map_as_the_launchers_do(self) -> None:
        self.assertEqual(gates.parse_docker_names("a\n\nb\n"), ("a", "b"))
        self.assertEqual(gates.service_state_from_status(0), "active")
        self.assertEqual(gates.service_state_from_status(3), "inactive")
        for status in (1, 4, None):
            with self.subTest(status=status):
                self.assertEqual(gates.service_state_from_status(status), "unknown")

    def test_snapshot_from_text_builds_a_gateable_snapshot(self) -> None:
        built = gates.snapshot_from_text(
            process_table="  215354     1 bash /root/nsva_dl.sh\n",
            gpu_compute_apps="",
            docker_names="",
            service_status=3,
            gpu_lock_free=True,
        )
        self.assertFalse(gates.evaluate(built).clear)


if __name__ == "__main__":
    unittest.main()
