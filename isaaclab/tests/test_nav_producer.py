"""Pure-math tests for the navigation command seam and its example follower.

``WaypointFollower`` is an example, not a tuned planner, so what is tested here
is not path quality. It is the three properties any producer has to have before
it may drive a real robot: it converges rather than oscillating, it never
exceeds its own clamps, and every command it emits is admissible for the policy
it is driving. That last one is the reason the seam exists -- a command outside
the trained envelope asks a network for behavior it has never seen.

Everything is closed-form or a fixed-step kinematic rollout. No torch, no
simulator, no randomness.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_core"
NAV_PACKAGE_DIR = REPO_ROOT / "packages" / "hexapod_nav"

for _package_dir in (CORE_PACKAGE_DIR, NAV_PACKAGE_DIR):
    if str(_package_dir) not in sys.path:  # mirrors the isaaclab/hexapod_rl bootstrap
        sys.path.insert(0, str(_package_dir))

from hexapod_core.action import POLICY_STEP_DT_S  # noqa: E402
from hexapod_core.command import (  # noqa: E402
    PHASE2_TARGET_ENVELOPE,
    CommandEnvelope,
    Frame,
    VelocityCommand,
)
from hexapod_nav.producer import CommandProducer, Pose2D, wrap_to_pi  # noqa: E402
from hexapod_nav.waypoint import WaypointFollower  # noqa: E402


def _rollout(
    follower: WaypointFollower,
    pose: Pose2D,
    *,
    steps: int = 3000,
    dt: float = POLICY_STEP_DT_S,
) -> tuple[Pose2D, list[VelocityCommand]]:
    """Integrate a unicycle at the policy rate and collect every command emitted.

    The robot is assumed to track its command exactly, which is generous but is
    the right abstraction here: this exercises the producer, not the gait.
    """

    emitted: list[VelocityCommand] = []
    for step in range(steps):
        command = follower.update(pose, step * dt)
        emitted.append(command)
        if follower.is_finished:
            break
        heading = wrap_to_pi(pose.heading_rad + command.wz_rad_s * dt)
        pose = Pose2D(
            x_m=pose.x_m + command.vx_mps * math.cos(heading) * dt,
            y_m=pose.y_m + command.vy_mps * 0.0 + command.vx_mps * math.sin(heading) * dt,
            heading_rad=heading,
        )
    return pose, emitted


class WrapToPiTests(unittest.TestCase):
    def test_wrapping(self) -> None:
        self.assertAlmostEqual(wrap_to_pi(0.0), 0.0, places=12)
        self.assertAlmostEqual(wrap_to_pi(math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(-math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(3.0 * math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(1.5 * math.pi), -0.5 * math.pi, places=12)
        self.assertAlmostEqual(wrap_to_pi(-1.5 * math.pi), 0.5 * math.pi, places=12)
        for angle in (0.3, -0.3, 2.0, -2.0, 12.0, -12.0):
            self.assertLessEqual(abs(wrap_to_pi(angle)), math.pi + 1.0e-12)
        with self.assertRaises(ValueError):
            wrap_to_pi(float("nan"))


class PoseTests(unittest.TestCase):
    def test_rejects_non_finite_components(self) -> None:
        Pose2D(0.0, 0.0, 0.0)
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Pose2D(bad, 0.0, 0.0)
            with self.assertRaises(ValueError):
                Pose2D(0.0, 0.0, bad)

    def test_follower_satisfies_the_producer_protocol(self) -> None:
        follower = WaypointFollower([(1.0, 0.0)])
        self.assertIsInstance(follower, CommandProducer)


class WaypointFollowerTests(unittest.TestCase):
    def test_stops_inside_the_stop_radius(self) -> None:
        follower = WaypointFollower([(1.0, 0.0)], stop_radius_m=0.12)
        command = follower.update(Pose2D(0.95, 0.02, 0.0), 0.0)
        self.assertTrue(command.is_stand())
        self.assertTrue(follower.is_finished)
        self.assertIsNone(follower.current_waypoint)
        # A finished follower keeps standing rather than drifting.
        self.assertTrue(follower.update(Pose2D(0.95, 0.02, 0.0), 1.0).is_stand())

    def test_no_waypoints_is_an_immediate_stand(self) -> None:
        follower = WaypointFollower([])
        self.assertTrue(follower.is_finished)
        self.assertTrue(follower.update(Pose2D(0.0, 0.0, 0.0), 0.0).is_stand())

    def test_turns_toward_the_goal_with_the_correct_sign(self) -> None:
        follower = WaypointFollower([(0.0, 2.0)])
        # Goal is 90 degrees to the left, which is past the heading gate, so the
        # follower should turn in place: positive yaw, zero forward.
        command = follower.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertGreater(command.wz_rad_s, 0.0)
        self.assertEqual(command.vx_mps, 0.0)
        # Mirror image, mirror sign.
        mirrored = WaypointFollower([(0.0, -2.0)]).update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertLess(mirrored.wz_rad_s, 0.0)
        self.assertEqual(mirrored.vx_mps, 0.0)

    def test_drives_forward_when_already_aligned(self) -> None:
        follower = WaypointFollower([(3.0, 0.0)], cruise_speed_mps=0.24)
        command = follower.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(command.wz_rad_s, 0.0, places=12)
        self.assertAlmostEqual(command.vx_mps, 0.24, places=12)
        self.assertEqual(command.vy_mps, 0.0)
        self.assertIs(command.frame, Frame.NAVIGATION)

    def test_forward_speed_is_distance_gated_then_clamped(self) -> None:
        follower = WaypointFollower(
            [(3.0, 0.0)], cruise_speed_mps=0.24, approach_gain_per_s=0.60
        )
        far = follower.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(far.vx_mps, 0.24, places=12, msg="clamped to cruise")
        near = follower.update(Pose2D(2.70, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(near.vx_mps, 0.60 * 0.30, places=12)
        self.assertLess(near.vx_mps, 0.24)

    def test_respects_its_clamps_everywhere(self) -> None:
        follower = WaypointFollower(
            [(2.0, 2.0), (-2.0, 1.0), (0.0, -3.0)],
            cruise_speed_mps=0.24,
            max_yaw_rate_rad_s=0.40,
        )
        _, emitted = _rollout(follower, Pose2D(0.0, 0.0, math.pi))
        self.assertGreater(len(emitted), 10)
        for command in emitted:
            self.assertLessEqual(abs(command.wz_rad_s), 0.40 + 1.0e-12)
            self.assertGreaterEqual(command.vx_mps, 0.0)
            self.assertLessEqual(command.vx_mps, 0.24 + 1.0e-12)
            self.assertEqual(command.vy_mps, 0.0, "the deployed policy has no lateral axis")

    def test_heading_error_converges_without_overshoot(self) -> None:
        # Start facing 90 degrees away from the goal, which is past the heading
        # gate: the follower turns in place, then drives as the error shrinks.
        # A proportional law with a clamped output can still ring if the sign
        # handling is wrong, so the interesting assertion is monotonicity, not
        # just the final value.
        follower = WaypointFollower([(0.0, 2.0)])
        pose = Pose2D(0.0, 0.0, 0.0)
        errors: list[float] = []
        turned_in_place = False
        for step in range(3000):
            command = follower.update(pose, step * POLICY_STEP_DT_S)
            if follower.is_finished:
                break
            bearing = math.atan2(2.0 - pose.y_m, 0.0 - pose.x_m)
            errors.append(abs(wrap_to_pi(bearing - pose.heading_rad)))
            turned_in_place = turned_in_place or (
                command.vx_mps == 0.0 and abs(command.wz_rad_s) > 0.0
            )
            heading = wrap_to_pi(pose.heading_rad + command.wz_rad_s * POLICY_STEP_DT_S)
            pose = Pose2D(
                x_m=pose.x_m + command.vx_mps * math.cos(heading) * POLICY_STEP_DT_S,
                y_m=pose.y_m + command.vx_mps * math.sin(heading) * POLICY_STEP_DT_S,
                heading_rad=heading,
            )
        self.assertTrue(turned_in_place, "expected an in-place turn past the gate")
        self.assertTrue(follower.is_finished, "follower never reached the waypoint")
        for earlier, later in zip(errors, errors[1:]):
            self.assertLessEqual(later, earlier + 1.0e-9, "heading error must not grow")
        self.assertAlmostEqual(errors[0], math.pi / 2.0, places=12)
        self.assertLess(errors[-1], 1.0e-2)

    def test_reaches_a_sequence_of_waypoints(self) -> None:
        waypoints = [(1.5, 0.0), (1.5, 1.5), (0.0, 1.5)]
        follower = WaypointFollower(waypoints, stop_radius_m=0.12)
        final_pose, emitted = _rollout(follower, Pose2D(0.0, 0.0, 0.0))
        self.assertTrue(follower.is_finished, "follower did not retire every waypoint")
        self.assertTrue(emitted[-1].is_stand())
        self.assertLessEqual(
            math.hypot(final_pose.x_m - waypoints[-1][0], final_pose.y_m - waypoints[-1][1]),
            0.12 + 1.0e-9,
        )

    def test_retires_several_satisfied_waypoints_in_one_tick(self) -> None:
        follower = WaypointFollower(
            [(0.02, 0.0), (0.0, 0.03), (1.0, 0.0)], stop_radius_m=0.12
        )
        command = follower.update(Pose2D(0.0, 0.0, 0.0), 0.0)
        self.assertEqual(follower.current_waypoint, (1.0, 0.0))
        self.assertGreater(command.vx_mps, 0.0)

    def test_every_emitted_command_passes_validation(self) -> None:
        follower = WaypointFollower([(2.0, -1.5), (-1.0, -2.0)])
        _, emitted = _rollout(follower, Pose2D(0.0, 0.0, 0.0))
        self.assertGreater(len(emitted), 10)
        for command in emitted:
            command.validate(PHASE2_TARGET_ENVELOPE)
            self.assertTrue(command.is_within(PHASE2_TARGET_ENVELOPE))

    def test_output_is_clamped_into_a_narrower_envelope(self) -> None:
        narrow = CommandEnvelope(
            forward_mps=(0.0, 0.10),
            lateral_mps=(0.0, 0.0),
            yaw_rate_rad_s=(-0.05, 0.05),
            name="narrow",
        )
        follower = WaypointFollower(
            [(2.0, 2.0)], cruise_speed_mps=0.24, max_yaw_rate_rad_s=0.40, envelope=narrow
        )
        _, emitted = _rollout(follower, Pose2D(0.0, 0.0, 0.0), steps=400)
        self.assertTrue(any(abs(command.wz_rad_s) > 0.0 for command in emitted))
        for command in emitted:
            command.validate(narrow)
            self.assertLessEqual(command.vx_mps, 0.10 + 1.0e-12)
            self.assertLessEqual(abs(command.wz_rad_s), 0.05 + 1.0e-12)

    def test_rejects_bad_construction(self) -> None:
        with self.assertRaises(ValueError):
            WaypointFollower([(0.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            WaypointFollower([(float("nan"), 0.0)])
        with self.assertRaises(ValueError):
            WaypointFollower([(1.0, 0.0)], cruise_speed_mps=0.0)
        with self.assertRaises(ValueError):
            WaypointFollower([(1.0, 0.0)], stop_radius_m=-0.1)
        with self.assertRaises(TypeError):
            WaypointFollower([(1.0, 0.0)]).update((0.0, 0.0, 0.0), 0.0)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
