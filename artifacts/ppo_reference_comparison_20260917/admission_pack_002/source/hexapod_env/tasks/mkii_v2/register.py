"""Explicit opt-in registration for the corrected CAD simulation lineage.

Future launch/validation entry points call ``register_mkii_v2()`` before task
resolution. Importing this module does not register tasks or import Isaac Sim.
The frozen register.py and every existing task ID remain unchanged.
"""
MKII_V2_FLAT_TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0"


def register_mkii_v2() -> list[str]:
    import gymnasium as gym

    if MKII_V2_FLAT_TASK_ID not in gym.registry:
        gym.register(
            id=MKII_V2_FLAT_TASK_ID,
            entry_point="hexapod_rl.env:HexapodEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.tasks.mkii_v2:HexapodMkiiV2FlatEnvCfg",
                "rsl_rl_cfg_entry_point": "hexapod_rl.tasks.mkii_v2:HexapodMkiiV2PPORunnerCfg",
            },
        )
    return [MKII_V2_FLAT_TASK_ID]


__all__ = ["MKII_V2_FLAT_TASK_ID", "register_mkii_v2"]
