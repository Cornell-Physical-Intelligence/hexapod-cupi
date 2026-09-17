"""Opt-in registration for the approved mass-corrected direct-drive robot.

Importing this module registers nothing and imports no simulator. The entry
point strings keep the historical ``hexapod_rl`` names, so the deployment shim
resolves them inside the container.
"""
MKII_RS05_FLAT_TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-RS05-Direct-v0"


def register_mkii_rs05() -> list[str]:
    import gymnasium as gym

    if MKII_RS05_FLAT_TASK_ID not in gym.registry:
        gym.register(
            id=MKII_RS05_FLAT_TASK_ID,
            entry_point="hexapod_rl.tasks.mkii_rs05:HexapodMkiiRs05Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "hexapod_rl.tasks.mkii_rs05:HexapodMkiiRs05FlatEnvCfg",
                "rsl_rl_cfg_entry_point": "hexapod_rl.tasks.mkii_rs05:HexapodMkiiRs05PPORunnerCfg",
            },
        )
    return [MKII_RS05_FLAT_TASK_ID]


__all__ = ["MKII_RS05_FLAT_TASK_ID", "register_mkii_rs05"]
