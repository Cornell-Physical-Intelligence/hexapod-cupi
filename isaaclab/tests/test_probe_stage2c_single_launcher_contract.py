from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ISAACLAB_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = ISAACLAB_DIR / "deploy" / "probe-stage2c-single-current-best"


class ProbeStage2CSingleLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def _array_entries(self, name: str) -> list[str]:
        match = re.search(
            rf"(?m)^{re.escape(name)}=\(\n(?P<body>.*?)^\)$",
            self.source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, f"missing shell array {name}")
        return [
            line.strip()
            for line in match.group("body").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def _invalid_invocation(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_intervention_allowlist_is_closed_and_pinned_once(self) -> None:
        expected = {
            "env.inactive_yaw_rate_slew_reward_scale",
            "env.inactive_yaw_rate_slew_reference_rad_s_per_step",
            "env.inactive_ground_contact_yaw_moment_reward_scale",
            "env.inactive_ground_contact_yaw_moment_reference_nm",
            "env.inactive_bilateral_longitudinal_contact_moment_reward_scale",
            "env.inactive_bilateral_longitudinal_contact_moment_reference_nm",
            "env.joint_torque_slew_reward_scale",
            "env.foot_slip_reward_scale",
        }
        allowed = self._array_entries("allowed_intervention_keys")
        baseline_tokens = self._array_entries("baseline_env_overrides")
        baseline_keys = [token.split("=", 1)[0] for token in baseline_tokens]
        self.assertEqual(set(allowed), expected)
        self.assertEqual(len(allowed), len(set(allowed)))
        self.assertEqual(len(baseline_keys), len(set(baseline_keys)))
        self.assertTrue(expected.issubset(baseline_keys))

    def test_invalid_or_abusive_tokens_fail_before_machine_preflight(self) -> None:
        cases = (
            (("batch", "98", "agent.save_interval=9"), "not a declared"),
            (("batch", "98", "hydra.run.dir=/tmp/x"), "not a declared"),
            (("batch", "98", "env.inactive_yaw_rate_reward_scale=-1"), "not a declared"),
            (
                (
                    "batch",
                    "98",
                    "env.foot_slip_reward_scale=-3",
                    "--run_name=attacker-controlled",
                ),
                "not a declared",
            ),
            (("batch", "98", "env.foot_slip_reward_scale=nan"), "must be finite"),
            (("batch", "98", "env.foot_slip_reward_scale=1e999"), "must be finite"),
            (
                (
                    "batch",
                    "98",
                    "env.foot_slip_reward_scale=-3",
                    "env.foot_slip_reward_scale=-2",
                ),
                "duplicate experimental override key",
            ),
            (
                (
                    "batch",
                    "98",
                    "env.inactive_ground_contact_yaw_moment_reference_nm=0",
                ),
                "reference must be strictly positive",
            ),
        )
        for arguments, expected_error in cases:
            with self.subTest(arguments=arguments):
                result = self._invalid_invocation(*arguments)
                self.assertEqual(result.returncode, 64, result.stderr)
                self.assertIn(expected_error, result.stderr)
                self.assertNotIn("unbound variable", result.stderr)

    def test_raw_arguments_never_reach_hydra(self) -> None:
        docker_section = self.source.split('cd "${isaac_lab_dir}"', 1)[1]
        self.assertIn('"${effective_hydra_overrides[@]}"', docker_section)
        self.assertNotIn('\n  "$@"', docker_section)
        self.assertIn("duplicate effective Hydra key", self.source)

    def test_shared_gpu_lock_precedes_contention_and_launch(self) -> None:
        self.assertIn("gpu_lock=/tmp/hexapod-isaac-gpu.lock", self.source)
        lock = self.source.index('exec 9>>"${gpu_lock}"')
        acquired = self.source.index("flock -n 9", lock)
        contention = self.source.index("docker ps --format", acquired)
        launch = self.source.index("timeout --signal=INT", contention)
        self.assertLess(lock, acquired)
        self.assertLess(acquired, contention)
        self.assertLess(contention, launch)
        self.assertIn("service_status != 3", self.source)

    def test_artifact_leaf_and_run_discovery_are_fail_closed(self) -> None:
        stale_check = self.source.index("refusing stale run(s)")
        atomic_mkdir = self.source.index('mkdir -- "${artifact_dir}"')
        launch = self.source.index("timeout --signal=INT", atomic_mkdir)
        self.assertLess(stale_check, atomic_mkdir)
        self.assertLess(atomic_mkdir, launch)
        self.assertNotIn('mkdir -p "${artifact_dir}"', self.source)
        self.assertIn("has a symlink/non-canonical ancestor", self.source)
        self.assertIn('[[ -e "${artifact_dir}" || -L "${artifact_dir}" ]]', self.source)

    def test_status_and_provenance_artifacts_are_distinct(self) -> None:
        for artifact in (
            "invocation.argv",
            "effective.params",
            "parent.checkpoint.sha256",
            "launcher.sha256",
            "docker.status",
            "train.status",
            "overall.status",
            "cleanup.status",
            "cleanup.provenance",
            "status.schema",
            "started_at_utc",
            "finished_at_utc",
        ):
            self.assertIn(artifact, self.source)
        self.assertIn("docker_status=not-started", self.source)
        self.assertIn("write_lines_atomic overall.status running", self.source)

    def test_signal_handling_targets_only_owned_child(self) -> None:
        for signal in ("HUP", "INT", "QUIT", "TERM"):
            self.assertRegex(self.source, rf"trap 'handle_signal {signal} \d+' {signal}")
        self.assertIn('kill -s "${signal_name}" -- "${child_pid}"', self.source)
        self.assertIn('kill -s TERM -- "${child_pid}"', self.source)
        self.assertNotRegex(self.source, r"\b(?:pkill|killall)\b")
        self.assertNotRegex(self.source, r"docker(?:\s+container)?\s+(?:kill|rm)\b")
        self.assertNotRegex(self.source, r"kill\s+(?:-[A-Z]+\s+)?(?:0|--\s+-?1)\b")

    def test_nonzero_docker_exit_cleans_only_exact_owned_container(self) -> None:
        cleanup = self.source.split("cleanup_exact_container() {", 1)[1].split(
            "handle_signal() {", 1
        )[0]
        self.assertIn('"${container_name}" 2>&1', cleanup)
        self.assertIn('"${observed_name:-}" != "/${container_name}"', cleanup)
        self.assertIn('"${observed_id:-}" =~ ^[0-9a-f]{64}$', cleanup)
        self.assertIn("--format '{{.Id}}|{{.Name}}|{{.State.Running}}|{{.State.Status}}'", cleanup)
        self.assertIn("IFS='|' read -r observed_id observed_name observed_running observed_state extra", cleanup)
        self.assertNotIn("--format '{{.Id}}\\t{{.Name}}", cleanup)
        self.assertIn('docker container stop --timeout 30 "${observed_id}"', cleanup)
        self.assertIn('"${observed_id}" 2>&1', cleanup)
        self.assertNotRegex(cleanup, r"docker(?:\s+container)?\s+(?:kill|rm)\b")
        self.assertNotRegex(cleanup, r"docker\s+(?:ps|container\s+ls)\b")
        self.assertNotRegex(cleanup, r"\brm\s+-r")

        wait_end = self.source.index("child_pid=\nwrite_lines_atomic docker.status")
        failure_gate = self.source.index("if (( docker_status != 0 )); then", wait_end)
        cleanup_call = self.source.index("cleanup_exact_container || true", failure_gate)
        docker_failure_exit = self.source.index(
            'echo "probe Docker command failed with status ${docker_status}"', cleanup_call
        )
        self.assertLess(failure_gate, cleanup_call)
        self.assertLess(cleanup_call, docker_failure_exit)

    def test_checkpoint_inventory_is_exact(self) -> None:
        self.assertIn('checkpoint_inventory=("${run_path}"/model_*.pt)', self.source)
        self.assertIn('(( ${#checkpoint_inventory[@]} != 12 ))', self.source)
        self.assertIn("for checkpoint_number in {0..11}; do", self.source)
        self.assertIn("unexpected checkpoint inventory member", self.source)
        self.assertIn("child.checkpoints.sha256", self.source)


if __name__ == "__main__":
    unittest.main()
