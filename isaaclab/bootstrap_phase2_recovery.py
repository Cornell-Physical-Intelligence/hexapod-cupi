#!/usr/bin/env python3
"""Create an auditable, forward-equivalent Phase 2 recovery bootstrap.

This utility operates only on RSL-RL checkpoint tensors. It does not import
Isaac Lab, start Kit, or mutate the source checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional


TRANSFORM_VERSION = 1
COMMAND_INDICES = (9, 10, 11)
FIXED_COMMAND_MEAN = (0.10, 0.0, 0.0)
FIXED_COMMAND_STD = (0.20, 0.15, 0.30)
DEFAULT_NORMALIZER_EPS = 1.0e-2
DEFAULT_ACTION_STD = 0.10
FIXED_NORMALIZER_COUNT = 1_000_000_000_000
EXPECTED_V5_MODEL200_SHA256 = (
    "43129a598f348d433bc724fb9fcb51b6620e8732e623cace0dc6fb1789f2d0e5"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_sha256(actual: str, expected: str) -> None:
    """Reject a wrong or already-resumed model_200 before transformation."""

    normalized_actual = actual.strip().lower()
    normalized_expected = expected.strip().lower()
    if len(normalized_expected) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_expected
    ):
        raise ValueError(f"Invalid expected SHA-256 value: {expected!r}")
    if normalized_actual != normalized_expected:
        raise ValueError(
            "Source checkpoint SHA-256 mismatch. Recovery must start from the "
            f"accepted pristine v5 model_200: expected {normalized_expected}, "
            f"got {normalized_actual}"
        )


def _state_tensor(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Checkpoint state is missing tensor {key!r}")
    return value


def deterministic_mlp_forward(
    state: dict[str, torch.Tensor],
    raw_observations: torch.Tensor,
    *,
    normalizer_eps: float = DEFAULT_NORMALIZER_EPS,
) -> torch.Tensor:
    """Evaluate the ELU MLP directly from an RSL-RL state dictionary."""

    mean = _state_tensor(state, "obs_normalizer._mean")
    std = _state_tensor(state, "obs_normalizer._std")
    if raw_observations.ndim != 2 or raw_observations.shape[1] != mean.shape[1]:
        raise ValueError(
            f"Expected raw observations shaped (N, {mean.shape[1]}), "
            f"got {tuple(raw_observations.shape)}"
        )
    value = (raw_observations - mean) / (std + normalizer_eps)
    layer_indices = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith("mlp.") and key.endswith(".weight")
    )
    if not layer_indices:
        raise ValueError("Checkpoint contains no MLP linear layers")
    for position, layer_index in enumerate(layer_indices):
        weight = _state_tensor(state, f"mlp.{layer_index}.weight")
        bias = _state_tensor(state, f"mlp.{layer_index}.bias")
        value = functional.linear(value, weight, bias)
        if position + 1 != len(layer_indices):
            value = functional.elu(value)
    return value


def rebase_model_command_normalization(
    state: dict[str, torch.Tensor],
    *,
    command_mean: tuple[float, float, float] = FIXED_COMMAND_MEAN,
    command_std: tuple[float, float, float] = FIXED_COMMAND_STD,
    normalizer_eps: float = DEFAULT_NORMALIZER_EPS,
) -> dict[str, Any]:
    """Rebase command RMS and preserve the raw forward policy exactly.

    The forward-command column is affine-compensated in the first layer.
    Lateral/yaw columns are zeroed, while their old contribution at raw zero is
    folded into the bias. Thus every raw observation with ``vy=wz=0`` retains
    the original first-layer preactivation for any raw ``vx``.
    """

    if normalizer_eps <= 0.0:
        raise ValueError("normalizer_eps must be positive")
    if len(command_mean) != 3 or len(command_std) != 3:
        raise ValueError("command_mean and command_std must each contain three values")
    if any(value <= 0.0 for value in command_std):
        raise ValueError(f"command_std must be positive, got {command_std!r}")

    mean = _state_tensor(state, "obs_normalizer._mean")
    var = _state_tensor(state, "obs_normalizer._var")
    std = _state_tensor(state, "obs_normalizer._std")
    count = _state_tensor(state, "obs_normalizer.count")
    first_weight = _state_tensor(state, "mlp.0.weight")
    first_bias = _state_tensor(state, "mlp.0.bias")
    if mean.ndim != 2 or mean.shape[0] != 1 or mean.shape[1] <= COMMAND_INDICES[-1]:
        raise ValueError(f"Unexpected observation-normalizer shape: {tuple(mean.shape)}")
    if var.shape != mean.shape or std.shape != mean.shape:
        raise ValueError("Observation-normalizer mean/variance/std shapes differ")
    if first_weight.ndim != 2 or first_weight.shape[1] != mean.shape[1]:
        raise ValueError(
            "First MLP layer input width does not match observation normalizer: "
            f"{tuple(first_weight.shape)} versus {tuple(mean.shape)}"
        )
    if first_bias.shape != (first_weight.shape[0],):
        raise ValueError(f"Unexpected first-layer bias shape: {tuple(first_bias.shape)}")
    if count.numel() != 1 or count.dtype != torch.long:
        raise ValueError(
            "Observation normalizer count must be one torch.long value, "
            f"got shape={tuple(count.shape)} dtype={count.dtype}"
        )

    old_mean = mean[0, list(COMMAND_INDICES)].clone()
    old_std = std[0, list(COMMAND_INDICES)].clone()
    old_count = int(count.item())
    old_effective_std = old_std + normalizer_eps
    new_mean = torch.as_tensor(command_mean, dtype=mean.dtype, device=mean.device)
    new_std = torch.as_tensor(command_std, dtype=std.dtype, device=std.device)
    new_effective_std = new_std + normalizer_eps
    if torch.any(old_effective_std <= 0.0):
        raise ValueError(f"Source checkpoint has invalid command std: {old_std.tolist()}")

    with torch.no_grad():
        original_columns = first_weight[:, list(COMMAND_INDICES)].clone()

        # Preserve the affine map for raw vx under the new normalization.
        first_weight[:, COMMAND_INDICES[0]] = (
            original_columns[:, 0]
            * new_effective_std[0]
            / old_effective_std[0]
        )
        first_bias.add_(
            original_columns[:, 0]
            * (new_mean[0] - old_mean[0])
            / old_effective_std[0]
        )

        # The accepted v5 task never supplied lateral/yaw commands. Preserve
        # their contribution at raw zero, then start Stage 1 with zero command
        # sensitivity instead of arbitrary 10-25 sigma extrapolation.
        for local_index, observation_index in enumerate(COMMAND_INDICES[1:], start=1):
            first_bias.add_(
                original_columns[:, local_index]
                * (0.0 - old_mean[local_index])
                / old_effective_std[local_index]
            )
            first_weight[:, observation_index].zero_()

        mean[0, list(COMMAND_INDICES)] = new_mean
        std[0, list(COMMAND_INDICES)] = new_std
        var[0, list(COMMAND_INDICES)] = torch.square(new_std)
        count.fill_(FIXED_NORMALIZER_COUNT)

    return {
        "old_command_mean": [float(value) for value in old_mean],
        "old_command_std": [float(value) for value in old_std],
        "new_command_mean": [float(value) for value in new_mean],
        "new_command_std": [float(value) for value in new_std],
        "old_normalizer_count": old_count,
        "new_normalizer_count": int(count.item()),
        "zeroed_first_layer_columns": list(COMMAND_INDICES[1:]),
    }


def _equivalence_observations(
    state: dict[str, torch.Tensor], sample_count: int
) -> torch.Tensor:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    mean = _state_tensor(state, "obs_normalizer._mean")
    std = _state_tensor(state, "obs_normalizer._std")
    generator = torch.Generator(device="cpu").manual_seed(20260825)
    observations = mean.repeat(sample_count, 1).clone()
    observations += torch.randn(
        observations.shape,
        generator=generator,
        dtype=observations.dtype,
        device=observations.device,
    ) * (std + DEFAULT_NORMALIZER_EPS)
    observations[:, COMMAND_INDICES[0]] = torch.linspace(
        -0.40, 0.60, sample_count, dtype=observations.dtype
    )
    observations[:, COMMAND_INDICES[1]] = 0.0
    observations[:, COMMAND_INDICES[2]] = 0.0
    return observations


def transform_checkpoint(
    checkpoint: dict[str, Any],
    *,
    action_std: float = DEFAULT_ACTION_STD,
    normalizer_eps: float = DEFAULT_NORMALIZER_EPS,
    equivalence_samples: int = 2048,
    equivalence_tolerance: float = 2.0e-5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a transformed checkpoint and an audit report."""

    if action_std <= 0.0:
        raise ValueError("action_std must be positive")
    required = ("actor_state_dict", "critic_state_dict")
    if any(not isinstance(checkpoint.get(key), dict) for key in required):
        raise ValueError(f"Checkpoint must contain dictionaries {required!r}")

    transformed = copy.deepcopy(checkpoint)
    report: dict[str, Any] = {
        "transform_version": TRANSFORM_VERSION,
        "command_indices": list(COMMAND_INDICES),
        "normalizer_eps": normalizer_eps,
        "action_std": action_std,
        "optimizer_state_preserved_but_model_only_loader_ignores_it": (
            "optimizer_state_dict" in checkpoint
        ),
        "source_iteration_preserved_but_model_only_loader_ignores_it": checkpoint.get(
            "iter"
        ),
        "models": {},
    }

    for state_key in required:
        source_state = checkpoint[state_key]
        target_state = transformed[state_key]
        raw_observations = _equivalence_observations(source_state, equivalence_samples)
        before = deterministic_mlp_forward(
            source_state, raw_observations, normalizer_eps=normalizer_eps
        )
        model_report = rebase_model_command_normalization(
            target_state, normalizer_eps=normalizer_eps
        )
        after = deterministic_mlp_forward(
            target_state, raw_observations, normalizer_eps=normalizer_eps
        )
        max_abs_error = float(torch.max(torch.abs(before - after)))
        model_report["forward_equivalence_max_abs_error"] = max_abs_error
        model_report["forward_equivalence_samples"] = equivalence_samples
        if max_abs_error > equivalence_tolerance:
            raise RuntimeError(
                f"{state_key} forward equivalence error {max_abs_error:.8g} "
                f"exceeds tolerance {equivalence_tolerance:.8g}"
            )
        report["models"][state_key] = model_report

    actor_state = transformed["actor_state_dict"]
    std_param = _state_tensor(actor_state, "distribution.std_param")
    if std_param.ndim != 1:
        raise ValueError(f"Expected scalar action std vector, got {tuple(std_param.shape)}")
    source_std_param = _state_tensor(
        checkpoint["actor_state_dict"], "distribution.std_param"
    )
    report["source_action_std"] = {
        "minimum": float(torch.min(source_std_param)),
        "mean": float(torch.mean(source_std_param)),
        "maximum": float(torch.max(source_std_param)),
    }
    with torch.no_grad():
        std_param.fill_(action_std)
    report["new_action_std"] = {
        "minimum": float(torch.min(std_param)),
        "mean": float(torch.mean(std_param)),
        "maximum": float(torch.max(std_param)),
    }
    transformed["phase2_recovery_bootstrap"] = copy.deepcopy(report)
    return transformed, report


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Accepted v5 model_200.pt checkpoint")
    parser.add_argument("destination", type=Path, help="Transformed bootstrap checkpoint")
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--action-std", type=float, default=DEFAULT_ACTION_STD)
    parser.add_argument("--normalizer-eps", type=float, default=DEFAULT_NORMALIZER_EPS)
    parser.add_argument("--equivalence-samples", type=int, default=2048)
    parser.add_argument("--equivalence-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--expected-source-sha256",
        default=EXPECTED_V5_MODEL200_SHA256,
        help="Hard provenance gate for the accepted pristine v5 model_200",
    )
    parser.add_argument("--force", action="store_true", help="Replace explicit output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    audit_path = (
        args.audit_json.expanduser().resolve()
        if args.audit_json is not None
        else Path(f"{destination}.audit.json")
    )
    if source == destination:
        raise ValueError("Source and destination checkpoint paths must differ")
    if not source.is_file():
        raise FileNotFoundError(source)
    for output in (destination, audit_path):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to replace {output}; pass --force explicitly")

    source_sha256 = sha256_file(source)
    validate_source_sha256(source_sha256, args.expected_source_sha256)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    transformed, report = transform_checkpoint(
        checkpoint,
        action_std=args.action_std,
        normalizer_eps=args.normalizer_eps,
        equivalence_samples=args.equivalence_samples,
        equivalence_tolerance=args.equivalence_tolerance,
    )
    report.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_sha256": source_sha256,
            "expected_source_sha256": args.expected_source_sha256.lower(),
            "destination": str(destination),
        }
    )
    transformed["phase2_recovery_bootstrap"] = copy.deepcopy(report)
    _atomic_torch_save(transformed, destination)
    report["destination_sha256"] = sha256_file(destination)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"checkpoint={destination}")
    print(f"audit={audit_path}")


if __name__ == "__main__":
    main()
