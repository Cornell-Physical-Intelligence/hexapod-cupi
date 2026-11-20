"""The navigation bank equals the stage 2 walking requirements.

``configs/walking_requirements.json`` owns the commands the survey may send.
The follower and training bank read them through ``contracts.command``.
"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from navigation.waypoint import (
    BANK_ARC_SPEED_MPS,
    BANK_ARC_YAW_RATE_RAD_S,
    BANK_BEARINGS,
    BANK_SPEED_LEVELS_MPS,
    BANK_YAW_RATE_RAD_S,
    TRAINING_HOLD_MIN_S,
    WaypointFollower,
)

ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS = ROOT / "configs/walking_requirements.json"


class RequirementsBindingTests(unittest.TestCase):
    def setUp(self):
        self.stage2 = json.loads(REQUIREMENTS.read_text())["sets"]["stage2_flat"]

    def test_bank_commands_equal_the_requirements(self):
        commands = self.stage2["commands"]
        self.assertEqual(list(BANK_SPEED_LEVELS_MPS), commands["translate_speeds_mps"])
        self.assertEqual(BANK_BEARINGS, commands["translate_bearings"])
        self.assertEqual(BANK_YAW_RATE_RAD_S, commands["turn_rate_rad_s"])
        self.assertEqual(BANK_ARC_SPEED_MPS, commands["arc"]["speed_mps"])
        self.assertEqual(BANK_ARC_YAW_RATE_RAD_S, commands["arc"]["turn_rate_rad_s"])
        self.assertTrue(commands["stand"])

    def test_follower_defaults_equal_the_requirements(self):
        self.assertEqual(TRAINING_HOLD_MIN_S, self.stage2["hold_min_s"])
        defaults = inspect.signature(WaypointFollower).parameters
        tolerance = self.stage2["stop"]["position_tolerance_m"]
        stopping = self.stage2["controller"]["stopping_distance_max_m"]
        self.assertGreater(defaults["stop_radius_m"].default, 0.0)
        self.assertLessEqual(defaults["stop_radius_m"].default, tolerance - stopping)
        self.assertEqual(defaults["hold_min_s"].default, self.stage2["hold_min_s"])

    def test_requirements_name_the_payload(self):
        payload = self.stage2["payload"]
        self.assertEqual(payload["device"], "Livox Mid-360 lidar")
        self.assertEqual(payload["mass_kg"], 0.265)
