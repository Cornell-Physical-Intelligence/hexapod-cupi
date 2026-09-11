"""Bind the Mid-360 placement study to the data it claims to come from.

`robot/sensors/livox_mid360/config/mid360_sensor.json` is a transcription of
Livox's manual and STEP models, and `experiments/terrain/tools/lidar_placement_study.py` is the
geometry that consumes it. These tests re-derive the transcribed numbers from
the STEP file itself, bind the field of view to the copy already held in
`isaaclab/hexapod_phase3/sensor_cfg.py`, and check the rasterizer against two
cases whose answer is known in closed form.

numpy is required; Isaac Sim, Isaac Lab, torch and a GPU are not.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "experiments/terrain/tools" / "lidar_placement_study.py"
SENSOR_JSON = REPO_ROOT / "robot" / "sensors" / "livox_mid360" / "config" / "mid360_sensor.json"
MOUNTS_JSON = REPO_ROOT / "robot" / "sensors" / "livox_mid360" / "config" / "mounts.json"
DEVICE_STEP = REPO_ROOT / "robot" / "sensors" / "livox_mid360" / "source" / "mid-360-asm.stp"
PHASE3_SENSOR_CFG = REPO_ROOT / "isaaclab" / "hexapod_phase3" / "sensor_cfg.py"

SENSOR = json.loads(SENSOR_JSON.read_text())
MOUNTS = json.loads(MOUNTS_JSON.read_text())

_CARTESIAN_POINT = re.compile(
    rb"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*"
    rb"\(\s*([-0-9.E+]+)\s*,\s*([-0-9.E+]+)\s*,\s*([-0-9.E+]+)\s*\)"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("lidar_placement_study", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _ground_plane(half_extent: float = 50.0) -> np.ndarray:
    """Two triangles covering a large square of the z = 0 plane."""
    e = half_extent
    return np.array(
        [
            [[-e, -e, 0.0], [e, -e, 0.0], [e, e, 0.0]],
            [[-e, -e, 0.0], [e, e, 0.0], [-e, e, 0.0]],
        ]
    )


def _finite_plane_share(
    height: float,
    half_extent: float,
    fov_deg: tuple[float, float],
    band_deg: tuple[float, float],
    samples: int = 20000,
) -> float:
    """Solid-angle share of ``band_deg`` that a finite square plane occludes.

    Seen from ``height`` above the centre of a square of side ``2*half_extent``
    lying on z = 0, the plane fills every elevation from the field-of-view floor
    up to the elevation of its own edge, which varies with azimuth.
    """
    psi = np.linspace(-np.pi, np.pi, samples, endpoint=False)
    edge = half_extent / np.maximum(np.abs(np.cos(psi)), np.abs(np.sin(psi)))
    edge_elevation = -np.degrees(np.arctan2(height, edge))
    low = max(fov_deg[0], band_deg[0])
    high = min(fov_deg[1], band_deg[1])
    top = np.clip(edge_elevation, low, high)
    blocked = np.sin(np.radians(top)) - math.sin(math.radians(low))
    total = math.sin(math.radians(high)) - math.sin(math.radians(low))
    return float(blocked.mean() / total)


class SensorTranscriptionTest(unittest.TestCase):
    def test_optical_centre_offset_is_the_step_models_own_origin(self):
        """The STEP origin is the FOV centre; the device hangs below it by this much."""
        points = _CARTESIAN_POINT.findall(DEVICE_STEP.read_bytes())
        self.assertGreater(len(points), 10000)
        heights = np.array([float(p[1]) for p in points])
        bottom_mm = float(heights.min())
        top_mm = float(heights.max())
        self.assertAlmostEqual(top_mm - bottom_mm, 60.0, delta=0.2)  # datasheet height
        offset_m = SENSOR["geometry"]["optical_centre_above_mount_plate_m"]
        self.assertAlmostEqual(offset_m * 1000.0, -bottom_mm, delta=0.2)

    def test_body_size_and_mass_match_the_specifications_table(self):
        self.assertEqual(SENSOR["geometry"]["body_size_m"], [0.065, 0.065, 0.060])
        self.assertEqual(SENSOR["geometry"]["mass_kg"], 0.265)
        self.assertEqual(SENSOR["ranging"]["close_proximity_blind_zone_m"], 0.1)
        self.assertEqual(SENSOR["ranging"]["range_10pct_reflectivity_m"], 40.0)
        self.assertEqual(SENSOR["ranging"]["range_80pct_reflectivity_m"], 70.0)

    def test_field_of_view_agrees_with_the_phase3_hardware_accounting(self):
        source = PHASE3_SENSOR_CFG.read_text()
        tree = ast.parse(source)
        found = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "mid360_vertical_fov_range_deg"
                and node.value is not None
            ):
                found = ast.literal_eval(node.value)
        self.assertIsNotNone(found, "Phase3 accounting no longer declares the Mid-360 FoV")
        self.assertEqual(tuple(SENSOR["fov"]["elevation_deg"]), found)
        self.assertEqual(SENSOR["fov"]["azimuth_deg"], [-180.0, 180.0])

    def test_inverted_mounting_clearance_is_recorded_and_unmet_on_this_robot(self):
        constraint = next(
            c for c in SENSOR["installation_constraints"] if c["id"] == "inverted_ground_clearance"
        )
        self.assertIn("0.5 m", constraint["quote"])
        deck_heights = [s["root_height"] for s in TOOL.STANCES.values()]
        self.assertTrue(all(h < 0.5 for h in deck_heights))


class MountCandidateTest(unittest.TestCase):
    def test_every_candidate_is_well_formed(self):
        names = [m["name"] for m in MOUNTS["mounts"]]
        self.assertEqual(len(names), len(set(names)), "duplicate mount name")
        for mount in MOUNTS["mounts"]:
            with self.subTest(mount=mount["name"]):
                self.assertEqual(len(mount["xyz"]), 3)
                self.assertEqual(len(mount["rpy_deg"]), 3)

    def test_phase3_candidate_reproduces_the_configured_lidar_origin(self):
        source = PHASE3_SENSOR_CFG.read_text()
        tree = ast.parse(source)
        configured = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "lidar_pos_b_m"
                and node.value is not None
            ):
                configured = ast.literal_eval(node.value)
        self.assertIsNotNone(configured, "Phase3 mounts no longer declare lidar_pos_b_m")
        candidate = next(
            m for m in MOUNTS["mounts"] if m["name"] == "phase3_current_assumption"
        )
        offset = SENSOR["geometry"]["optical_centre_above_mount_plate_m"]
        self.assertAlmostEqual(candidate["xyz"][2] + offset, configured[2], places=3)


class GeometryTest(unittest.TestCase):
    def test_rotation_helpers_agree_on_a_quarter_turn(self):
        rot = TOOL.rotation_from_rpy((0.0, 0.0, math.pi / 2))
        np.testing.assert_allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
        axis = TOOL.rotation_about_axis(np.array([0.0, 0.0, 1.0]), math.pi / 2)
        np.testing.assert_allclose(axis, rot, atol=1e-12)

    def test_empty_scene_reproduces_the_closed_form_blind_radius(self):
        height = 1.0
        result = TOOL.evaluate_mount(
            triangles=np.zeros((0, 3, 3)),
            plate_xyz=(0.0, 0.0, height),
            plate_rpy_deg=(0.0, 0.0, 0.0),
            optical_offset=0.0,
            root_height=0.0,
            el_range=(-7.0, 52.0),
            az_res_deg=1.0,
            el_res_deg=0.25,
            mast_radius=0.0,
        )
        expected = height / math.tan(math.radians(7.0))
        # The grid reports cell centres, so the steepest ray it can offer sits
        # half an elevation cell above the -7 degree edge.
        expected_cell_centre = height / math.tan(math.radians(7.0 - 0.25 / 2))
        self.assertEqual(result["blocked_fraction_fov"], 0.0)
        self.assertEqual(result["ground_visible_azimuth_fraction"], 1.0)
        self.assertAlmostEqual(result["unobstructed_blind_radius_m"], expected, places=3)
        self.assertAlmostEqual(
            result["forward_ground_blind_radius_m"], expected_cell_centre, delta=0.01
        )

    def test_a_finite_ground_plane_blocks_the_share_quadrature_predicts(self):
        height, half_extent = 1.0, 50.0
        result = TOOL.evaluate_mount(
            triangles=_ground_plane(half_extent),
            plate_xyz=(0.0, 0.0, height),
            plate_rpy_deg=(0.0, 0.0, 0.0),
            optical_offset=0.0,
            root_height=0.0,
            el_range=(-7.0, 52.0),
            az_res_deg=1.0,
            el_res_deg=0.25,
            mast_radius=0.0,
        )
        # An occupancy grid blocks a whole cell as soon as any occluder falls in
        # it, so the answer is bracketed below by the exact share and above by
        # that share plus one elevation cell along the grazing edge.
        for key, band in (
            ("blocked_fraction_fov", (-7.0, 52.0)),
            ("blocked_fraction_below_horizon", (-7.0, 0.0)),
        ):
            with self.subTest(metric=key):
                exact = _finite_plane_share(height, half_extent, (-7.0, 52.0), band)
                cell_bias = (
                    math.sin(math.radians(band[0] + 0.25)) - math.sin(math.radians(band[0]))
                ) / (math.sin(math.radians(band[1])) - math.sin(math.radians(band[0])))
                self.assertGreaterEqual(result[key], exact - 1e-3)
                self.assertLessEqual(result[key], exact + cell_bias + 1e-3)
        # The plane hides every ground point inside its own footprint, so the
        # nearest ground the sensor can still reach lies just past its edge.
        self.assertEqual(result["ground_visible_azimuth_fraction"], 1.0)
        self.assertGreaterEqual(result["ground_blind_radius_m"]["min"], half_extent - 1.0)

    def test_forward_kinematics_places_the_six_coxa_axes_on_the_deck_perimeter(self):
        links, joints = TOOL.parse_urdf(TOOL.DEFAULT_URDF)
        poses = TOOL.forward_kinematics(links, joints, TOOL.STANCES["stage2c"])
        coxa = sorted(
            tuple(round(float(v), 4) for v in poses[name][1])
            for name in links
            if name.startswith("coxa")
        )
        self.assertEqual(
            coxa,
            sorted(
                [
                    (-0.12, 0.0, -0.023),
                    (-0.08, -0.2, -0.023),
                    (-0.08, 0.2, -0.023),
                    (0.08, -0.2, -0.023),
                    (0.08, 0.2, -0.023),
                    (0.12, 0.0, -0.023),
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
