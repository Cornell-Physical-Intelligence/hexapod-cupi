"""Check caller-owned limits and the approved anatomical axes."""
import math
import unittest

from contracts.command import CommandEnvelope, VelocityCommand
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
