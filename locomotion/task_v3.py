"""Experimental immediate tracking feedback with the version 2 penalty coefficients."""
from dataclasses import replace
import hashlib
from pathlib import Path

from . import task_v2


REWARD_VERSION = "paper_table1_immediate_v3"
REWARD_V3_CONFIG = replace(task_v2.REWARD_V2_CONFIG, tracking_kernel="scaled")


class TrainingTaskV3(task_v2.TrainingTaskV2):
    """Use each completed control's velocity for the translation and yaw rewards."""

    def __init__(self, env, config=None, output_dir=None):
        super().__init__(env, config, output_dir, reward_config=REWARD_V3_CONFIG)

    def reward_telemetry(self, command):
        return {**self.env.telemetry, "previous_action": self.previous_action,
                "previous_joint_velocity_rad_s": self.previous_joint_velocity}

    def declaration(self):
        result = super().declaration()
        result["reward_version"] = REWARD_VERSION
        result["reward_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["reward_base_source_sha256"] = hashlib.sha256(Path(task_v2.__file__).read_bytes()).hexdigest()
        reward = result["reward"]
        reward["version"] = REWARD_VERSION
        reward["base_reward_review"] = reward.pop("review")
        reward["review"] = {"status": "experimental", "basis": "docs/REWARD_V2_TABLE1_AUDIT.md section 12",
                            "native_outcome": "pending"}
        reward["adaptations"] = [item for item in reward["adaptations"]
                                 if not item.startswith("The stride kernel")]
        reward["adaptations"].append("Translation and yaw tracking use this control's endpoint velocity, with no stride average.")
        reward["unused_config_fields"] = ["stride_controls"]
        reward["termination_reference_scope"] = (
            "Version 2 reference retained for provenance. The timing comparison keeps its coefficient; "
            "the worst-return bound has not been established for version 3.")
        return result

    def status(self, *, reset_interval=False):
        return {**super().status(reset_interval=reset_interval), "reward_version": REWARD_VERSION}


scorer_default = task_v2.scorer_reward("tracking_kernel=scaled")
