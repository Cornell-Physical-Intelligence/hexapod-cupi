"""Evaluate native locomotion and bind its evidence.

The package holds the motion acceptance scorer, the native policy evaluation runner,
standing capture and its scorer, the Kit RGB camera, training load accumulation,
contact-force and torque summaries, the fixed-mean noise probe, checkpoint identity
checks and standing admission. Callers import the scorer names from the package.
"""
from locomotion.evaluation.scoring import MOVING_RMS, QUIET_GATES, STAGE2C_TORQUE, STAND_RMS, TAIL, evaluation_scenarios, score_recording, score_transitions, summarize_suite, trajectory_command, transition_sequence
