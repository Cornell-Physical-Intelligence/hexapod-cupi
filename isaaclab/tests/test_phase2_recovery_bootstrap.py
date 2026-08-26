"""CPU tests for the auditable Phase 2 recovery checkpoint transform."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "bootstrap_phase2_recovery.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_phase2_recovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def make_model_state(*, actor: bool, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    observation_dim = 66
    state: dict[str, torch.Tensor] = {
        "obs_normalizer._mean": torch.randn(
            1, observation_dim, generator=generator
        )
        * 0.05,
        "obs_normalizer._std": torch.rand(
            1, observation_dim, generator=generator
        )
        * 0.2
        + 0.02,
        "obs_normalizer.count": torch.tensor(19_000_000, dtype=torch.long),
        "mlp.0.weight": torch.randn(32, observation_dim, generator=generator) * 0.1,
        "mlp.0.bias": torch.randn(32, generator=generator) * 0.1,
        "mlp.2.weight": torch.randn(16, 32, generator=generator) * 0.1,
        "mlp.2.bias": torch.randn(16, generator=generator) * 0.1,
        "mlp.4.weight": torch.randn(18 if actor else 1, 16, generator=generator)
        * 0.1,
        "mlp.4.bias": torch.randn(18 if actor else 1, generator=generator) * 0.1,
    }
    state["obs_normalizer._mean"][0, 9:12] = torch.tensor([0.2487, -2e-6, -2e-5])
    state["obs_normalizer._std"][0, 9:12] = torch.tensor(
        [0.0599, 0.00582, 0.01455]
    )
    state["obs_normalizer._var"] = torch.square(state["obs_normalizer._std"])
    if actor:
        state["distribution.std_param"] = torch.linspace(0.26, 0.35, 18)
    return state


class Phase2RecoveryBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = {
            "actor_state_dict": make_model_state(actor=True, seed=10),
            "critic_state_dict": make_model_state(actor=False, seed=20),
            "optimizer_state_dict": {"state": {1: {"step": torch.tensor(8)}}},
            "iter": 200,
            "infos": None,
        }

    def test_transform_preserves_raw_forward_actor_and_critic(self):
        raw_observations = torch.randn(1024, 66)
        raw_observations[:, 9] = torch.linspace(-0.4, 0.6, 1024)
        raw_observations[:, 10:12] = 0.0
        before = {
            key: bootstrap.deterministic_mlp_forward(
                self.checkpoint[key], raw_observations
            )
            for key in ("actor_state_dict", "critic_state_dict")
        }

        transformed, report = bootstrap.transform_checkpoint(self.checkpoint)
        for key in ("actor_state_dict", "critic_state_dict"):
            after = bootstrap.deterministic_mlp_forward(
                transformed[key], raw_observations
            )
            torch.testing.assert_close(after, before[key], atol=2e-5, rtol=1e-5)
            self.assertLessEqual(
                report["models"][key]["forward_equivalence_max_abs_error"],
                2e-5,
            )

    def test_transform_sets_fixed_stats_zero_columns_and_action_std(self):
        transformed, report = bootstrap.transform_checkpoint(self.checkpoint)
        for key in ("actor_state_dict", "critic_state_dict"):
            state = transformed[key]
            torch.testing.assert_close(
                state["obs_normalizer._mean"][0, 9:12],
                torch.tensor(bootstrap.FIXED_COMMAND_MEAN),
            )
            torch.testing.assert_close(
                state["obs_normalizer._std"][0, 9:12],
                torch.tensor(bootstrap.FIXED_COMMAND_STD),
            )
            torch.testing.assert_close(
                state["obs_normalizer._var"][0, 9:12],
                torch.square(torch.tensor(bootstrap.FIXED_COMMAND_STD)),
            )
            self.assertEqual(
                int(state["obs_normalizer.count"]), bootstrap.FIXED_NORMALIZER_COUNT
            )
            self.assertEqual(
                report["models"][key]["new_normalizer_count"],
                bootstrap.FIXED_NORMALIZER_COUNT,
            )
            self.assertEqual(int(torch.count_nonzero(state["mlp.0.weight"][:, 10:12])), 0)
        torch.testing.assert_close(
            transformed["actor_state_dict"]["distribution.std_param"],
            torch.full((18,), 0.10),
        )
        self.assertEqual(report["source_iteration_preserved_but_model_only_loader_ignores_it"], 200)
        self.assertEqual(transformed["iter"], 200)
        self.assertIn("optimizer_state_dict", transformed)

    def test_transform_does_not_mutate_source(self):
        actor_weight = self.checkpoint["actor_state_dict"]["mlp.0.weight"].clone()
        actor_std = self.checkpoint["actor_state_dict"]["distribution.std_param"].clone()
        bootstrap.transform_checkpoint(self.checkpoint)
        torch.testing.assert_close(
            self.checkpoint["actor_state_dict"]["mlp.0.weight"], actor_weight
        )
        torch.testing.assert_close(
            self.checkpoint["actor_state_dict"]["distribution.std_param"], actor_std
        )

    def test_zeroed_new_axes_start_with_no_actor_effect(self):
        transformed, _ = bootstrap.transform_checkpoint(self.checkpoint)
        observations = torch.randn(64, 66)
        observations[:, 9] = 0.25
        observations[:, 10:12] = 0.0
        baseline = bootstrap.deterministic_mlp_forward(
            transformed["actor_state_dict"], observations
        )
        observations[:, 10] = torch.linspace(-0.15, 0.15, 64)
        observations[:, 11] = torch.linspace(0.30, -0.30, 64)
        changed = bootstrap.deterministic_mlp_forward(
            transformed["actor_state_dict"], observations
        )
        torch.testing.assert_close(changed, baseline)

    def test_source_provenance_gate_rejects_corrupted_resume_checkpoint(self):
        bootstrap.validate_source_sha256(
            bootstrap.EXPECTED_V5_MODEL200_SHA256,
            bootstrap.EXPECTED_V5_MODEL200_SHA256,
        )
        with self.assertRaisesRegex(ValueError, "Source checkpoint SHA-256 mismatch"):
            bootstrap.validate_source_sha256(
                "4f4f2572e778f64523c55069d5723605e5d06c655e4c8fddf8baf7ccfd237f42",
                bootstrap.EXPECTED_V5_MODEL200_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
