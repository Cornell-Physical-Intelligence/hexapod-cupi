"""Opt-in registration for the physical motor-coordinate lineage."""
from hexapod_core.fourbar_v1 import TASK_ID

MKII_FOURBAR_V1_TASK_ID = TASK_ID


def register_mkii_fourbar_v1():
    import gymnasium as gym
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point="hexapod_rl.tasks.mkii_fourbar_v1:HexapodMkiiFourbarEnv",
            disable_env_checker=True, kwargs={
                "env_cfg_entry_point": "hexapod_rl.tasks.mkii_fourbar_v1:HexapodMkiiFourbarV1EnvCfg",
                "rsl_rl_cfg_entry_point": "hexapod_rl.tasks.mkii_fourbar_v1:HexapodMkiiFourbarV1PPORunnerCfg"})
    return [TASK_ID]

