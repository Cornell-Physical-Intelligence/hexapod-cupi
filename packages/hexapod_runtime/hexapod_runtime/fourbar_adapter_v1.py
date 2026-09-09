"""Explicit physical-tree/motor adapter and deployable observation/action seam."""
from __future__ import annotations

from hexapod_core import fourbar_v1 as contract
from .action_pipeline import ActionPipeline


class FourbarJointAdapter:
    def __init__(self, tree_joint_names, kinematics):
        self.kinematics = contract.validate_kinematics(kinematics)
        self.tree_joint_names = contract.validate_tree_names(tree_joint_names)
        self.active_indices = contract.active_indices(tree_joint_names)

    def gather_active(self, values):
        values = contract.finite_vector(values, 30, "physical tree state")
        return [values[index] for index in self.active_indices]

    def closed_state(self, motor_positions, motor_velocities):
        positions = dict(zip(contract.ACTIVE_JOINT_NAMES, contract.finite_vector(motor_positions, 18, "motor positions")))
        velocities = dict(zip(contract.ACTIVE_JOINT_NAMES, contract.finite_vector(motor_velocities, 18, "motor velocities")))
        for name, relation in self.kinematics["passive_relations"].items():
            source, factor = relation["source_joint"], relation["multiplier"]
            positions[name] = factor * positions[source] + relation["offset_rad"]
            velocities[name] = factor * velocities[source]
        for name, position in positions.items():
            lower, upper = self.kinematics["joint_limits_rad"][name]
            if not lower <= position <= upper:
                raise ValueError(f"Closed reset lies outside {name} limits")
        return ([positions[name] for name in self.tree_joint_names], [velocities[name] for name in self.tree_joint_names])

    def active_targets(self, values):
        """Return exactly 18 (tree index, target) pairs; never command passives."""
        return list(zip(self.active_indices, contract.finite_vector(values, 18, "motor targets")))


def build_fourbar_observation(**fields):
    if set(fields) != {name for name, _ in contract.OBSERVATION_FIELDS}:
        raise ValueError("Observation fields do not match the physical motor-coordinate contract")
    result = []
    for name, width in contract.OBSERVATION_FIELDS:
        values = contract.finite_vector(fields[name], width, name)
        if name == "estimated_motor_burst_headroom" and any(not 0 <= value <= 1 for value in values):
            raise ValueError("Motor burst headroom must lie in [0,1]")
        result.extend(values)
    return result


class FourbarActionPipeline:
    def __init__(self, manifest, *, expected_manifest):
        if manifest != expected_manifest or manifest.get("schema") != "hexapod.physical_fourbar_runtime.v1":
            raise ValueError("Physical asset/motor/runtime manifest mismatch")
        self._pipeline = ActionPipeline(default_joint_positions=manifest["default_motor_positions_rad"],
            action_scale=contract.ACTION_SCALE_RAD, stand_action_scale=1., command_active_threshold=.01,
            slew_limit_rad_per_20ms=contract.SLEW_RAD_PER_20MS, step_dt=contract.POLICY_DT_S,
            soft_limits=manifest["soft_motor_limits_rad"])
        self._soft_limits = manifest["soft_motor_limits_rad"]
        self._ready = False

    def reset(self, measured_motor_positions, *, motor_names):
        self._ready = False
        if tuple(motor_names) != contract.ACTIVE_JOINT_NAMES:
            raise ValueError("Wrong physical motor feedback order")
        measured = contract.finite_vector(measured_motor_positions, 18, "motor feedback")
        if any(not low <= position <= high for position, (low, high) in zip(measured, self._soft_limits)):
            raise ValueError("Motor feedback is outside soft limits")
        self._pipeline.reset(measured)
        self._ready = True

    def step(self, action, command, *, motor_names, command_frame):
        if not self._ready:
            raise RuntimeError("Measured motor reset is required")
        if tuple(motor_names) != contract.ACTIVE_JOINT_NAMES or command_frame != contract.COMMAND_FRAME:
            raise ValueError("Physical motor/frame contract mismatch")
        return self._pipeline.step(contract.finite_vector(action, 18, "action"),
                                   contract.finite_vector(command, 3, "command"))
