"""Check caller-owned limits and the approved anatomical axes."""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from contracts.command import CommandEnvelope, CommandSet, VelocityCommand, load_requirement_set
from contracts.frames import body_to_navigation, navigation_to_body
from contracts.pose import Pose2D


class ContractTests(unittest.TestCase):
    def test_consumer_must_supply_limits(self):
        command = VelocityCommand(0.05, 0, 0)
        with self.assertRaises(TypeError):
            command.validate()
        command.validate(CommandEnvelope((0, 0.1), (-0.1, 0.1), (-1, 1)))
        with self.assertRaises(ValueError):
            command.validate(CommandEnvelope((0, 0.01), (0, 0), (0, 0)))

    def test_stand_exception_does_not_admit_nonfinite_commands(self):
        envelope = CommandEnvelope((0.02, 0.05), (0, 0), (0, 0))
        VelocityCommand(0, 0, 0).validate(envelope)
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                VelocityCommand(value, 0, 0).validate(envelope)
            with self.assertRaises(ValueError):
                Pose2D(0, 0, value)

    def test_anatomical_forward_and_left_axes(self):
        self.assertEqual(body_to_navigation((0, -1, 0)), (1, 0, 0))
        self.assertEqual(body_to_navigation((1, 0, 0)), (0, 1, 0))
        self.assertEqual(navigation_to_body((0.2, -0.4, 0.7)), (-0.4, -0.2, 0.7))

    def test_a_command_set_builds_only_the_declared_speeds_and_bearings(self):
        commands = CommandSet.from_mapping(
            {
                "translate_speeds_mps": [0.01, 0.02],
                "translate_bearings": 4,
                "turn_rate_rad_s": 0.1,
                "arc": {"speed_mps": 0.015, "turn_rate_rad_s": 0.05},
                "stand": True,
            }
        )
        bank = commands.nonzero()
        self.assertEqual(len(bank), 12)
        self.assertEqual(bank[0].as_tuple(), (0.01, 0.0, 0.0))
        self.assertEqual(bank[4].as_tuple(), (0.02, 0.0, 0.0))
        self.assertEqual(bank[-4].as_tuple(), (0.0, 0.0, -0.1))
        self.assertEqual(bank[-1].as_tuple(), (0.015, 0.0, 0.05))
        self.assertEqual(commands.nonzero(0.01), bank[:4] + bank[8:10])
        for command in bank:
            command.validate(commands.envelope())

    def test_named_owner_loading_hashes_the_bytes_it_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.json"
            raw = b'{"sets": {"custom": {"commands": {"stand": true}}}}\n'
            path.write_bytes(raw)
            rules, digest = load_requirement_set(path, "custom")
            self.assertEqual(rules, {"commands": {"stand": True}})
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())

    def test_frozen_source_imports_use_the_packed_owner_without_a_checkout(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            for package, names in (
                ("contracts", ("__init__.py", "command.py", "pose.py")),
                ("navigation", ("__init__.py", "producer.py", "waypoint.py")),
                (
                    "locomotion",
                    (
                        "__init__.py",
                        "omniwalk/__init__.py",
                        "omniwalk/mdp/__init__.py",
                        "omniwalk/mdp/commands.py",
                    ),
                ),
            ):
                for name in names:
                    target = source / package / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root / "src" / package / name, target)
            (source / "configs").mkdir()
            rules, _ = load_requirement_set()
            rules["commands"]["translate_speeds_mps"] = [0.01, 0.02]
            rules["commands"]["arc"]["speed_mps"] = 0.015
            (source / "configs/walking_requirements.json").write_text(
                json.dumps({"sets": {"stage2_flat": rules}})
            )
            program = """
import sys
sys.path.insert(0, sys.argv[1])
from contracts.command import default_requirements_path
from navigation.waypoint import command_bank as navigation_bank
from locomotion.omniwalk.mdp.commands import TaskConfig, command_bank
assert default_requirements_path().parent.parent == __import__('pathlib').Path(sys.argv[1]).resolve()
assert TaskConfig().speed_levels_mps == (0.01, 0.02)
assert command_bank(TaskConfig()) == navigation_bank()
assert navigation_bank()[0] == [0.01, 0.0, 0.0]
"""
            subprocess.run([sys.executable, "-I", "-c", program, str(source)], check=True)
