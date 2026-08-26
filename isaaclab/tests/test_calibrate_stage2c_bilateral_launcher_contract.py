from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ISAACLAB_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = ISAACLAB_DIR / "deploy" / "calibrate-stage2c-bilateral-current-best"


class CalibrateStage2CBilateralLauncherContractTests(unittest.TestCase):
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

    def test_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_collection_contract_is_fixed_and_no_learning(self) -> None:
        self.assertRegex(self.source, r"(?m)^training_seed=99$")
        self.assertRegex(self.source, r"(?m)^num_envs=12288$")
        self.assertRegex(self.source, r"(?m)^rollout_steps_per_env=24$")
        self.assertRegex(self.source, r"(?m)^max_iterations=1$")
        overrides = self._array_entries("fixed_hydra_overrides")
        expected = {
            "agent.save_interval=1",
            "agent.algorithm.learning_rate=0.0",
            "agent.algorithm.clip_param=0.06",
            "env.gate_longitudinal_reward_by_command=false",
            "env.longitudinal_signed_progress_reward_scale=2.0",
            "env.longitudinal_normalized_error_penalty_scale=1.0",
            "env.inactive_yaw_rate_reward_scale=-160.0",
            "env.inactive_yaw_rate_slew_reward_scale=0.0",
            "env.inactive_yaw_rate_slew_reference_rad_s_per_step=0.04",
            "env.inactive_ground_contact_yaw_moment_reward_scale=0.0",
            "env.inactive_ground_contact_yaw_moment_reference_nm=0.5",
            "env.inactive_bilateral_longitudinal_contact_moment_reward_scale=-1.0",
            "env.inactive_bilateral_longitudinal_contact_moment_reference_nm=1.8",
            "env.support_contact_low_speed_threshold_mps=0.18",
            "env.support_contact_target_high_speed=4.0",
            "env.support_shortfall_reward_scale=-0.90",
            "env.rated_torque_excess_reward_scale=-0.65",
            "env.max_joint_rated_torque_excess_l1_reward_scale=-0.50",
            "env.torque_saturation_reward_scale=-1.50",
            "env.joint_torque_slew_reward_scale=-0.01",
            "env.processed_joint_target_slew_limit_rad_per_20ms=0.040",
            "env.foot_slip_reward_scale=-0.50",
        }
        self.assertEqual(set(overrides), expected)
        self.assertEqual(len(overrides), len(expected))
        self.assertNotIn("agent.num_steps_per_env", self.source)
        self.assertIn("rollout_steps_source=inherited_registered_ppo_config", self.source)

    def test_only_output_label_is_caller_controlled(self) -> None:
        for arguments in ((), ("one", "two"), ("../escape",), ("bad/value",)):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 64, result.stderr)
        docker_section = self.source.split('cd "${isaac_lab_dir}"', 1)[1]
        self.assertIn('"${fixed_hydra_overrides[@]}"', docker_section)
        self.assertNotIn('\n+  "$@"', docker_section)

    def test_parent_identity_and_model_zero_inventory_are_exact(self) -> None:
        self.assertIn(
            "parent_sha256=a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a",
            self.source,
        )
        self.assertIn('checkpoint_inventory=("${run_path}"/model_*.pt)', self.source)
        self.assertIn('(( ${#checkpoint_inventory[@]} != 1 ))', self.source)
        self.assertIn('checkpoint_path=${run_path}/model_0.pt', self.source)
        self.assertIn("child.checkpoint.sha256", self.source)
        self.assertGreaterEqual(self.source.count("verify_parent || exit 65"), 3)

    def test_shared_lock_and_contention_checks_precede_launch(self) -> None:
        self.assertIn("gpu_lock=/tmp/hexapod-isaac-gpu.lock", self.source)
        lock = self.source.index('exec 9>>"${gpu_lock}"')
        acquired = self.source.index("flock -n 9", lock)
        service = self.source.index("systemctl is-active --quiet", acquired)
        containers = self.source.index("docker ps --format", service)
        launch = self.source.index("timeout --signal=INT", containers)
        self.assertLess(lock, acquired)
        self.assertLess(acquired, service)
        self.assertLess(service, containers)
        self.assertLess(containers, launch)
        self.assertIn("service_status != 3", self.source)
        self.assertIn("refusing to contend with active Docker containers", self.source)

    def test_failure_cleanup_targets_only_exact_owned_container(self) -> None:
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
        self.assertNotRegex(cleanup, r"docker(?:\s+container)?\s+(?:kill|rm)\b")
        self.assertNotRegex(cleanup, r"\b(?:pkill|killall)\b")
        failure_gate = self.source.index("if (( docker_status != 0 )); then")
        cleanup_call = self.source.index("cleanup_exact_container || true", failure_gate)
        failure_exit = self.source.index(
            'echo "calibration Docker command failed with status ${docker_status}"',
            cleanup_call,
        )
        self.assertLess(failure_gate, cleanup_call)
        self.assertLess(cleanup_call, failure_exit)

    def test_logs_and_provenance_are_preserved(self) -> None:
        for artifact in (
            "train.log",
            "invocation.argv",
            "effective.params",
            "parent.checkpoint.sha256",
            "child.checkpoint.sha256",
            "launcher.sha256",
            "workspace.source.sha256",
            "child.run_path",
            "docker.status",
            "overall.status",
            "cleanup.status",
            "cleanup.provenance",
            "started_at_utc",
            "finished_at_utc",
        ):
            self.assertIn(artifact, self.source)


if __name__ == "__main__":
    unittest.main()
