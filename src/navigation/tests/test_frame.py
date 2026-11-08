"""Checks for the draft fixed-scale WGS84 forward conversion."""

from __future__ import annotations

import unittest

from navigation.frame import LocalFrame


class LocalFrameTests(unittest.TestCase):
    def test_origin_is_read_only_and_maps_to_zero(self) -> None:
        frame = LocalFrame(origin_lat_deg=42.4447, origin_lon_deg=-76.4825)
        self.assertEqual(frame.origin_lat_deg, 42.4447)
        self.assertEqual(frame.origin_lon_deg, -76.4825)
        self.assertEqual(frame.to_local(lat_deg=42.4447, lon_deg=-76.4825), (0.0, 0.0))
        for name in ("origin_lat_deg", "origin_lon_deg"):
            with self.subTest(property=name), self.assertRaises(AttributeError):
                setattr(frame, name, 0.0)

    def test_east_and_north_axes_have_the_documented_order(self) -> None:
        frame = LocalFrame(origin_lat_deg=42.4447, origin_lon_deg=-76.4825)
        east, north = frame.to_local(lat_deg=42.4447, lon_deg=-76.4824)
        self.assertGreater(east, 0.0)
        self.assertEqual(north, 0.0)
        east, north = frame.to_local(lat_deg=42.4448, lon_deg=-76.4825)
        self.assertEqual(east, 0.0)
        self.assertGreater(north, 0.0)

    def test_small_offsets_against_independent_tangent_plane_fixtures(self) -> None:
        # Reference: zero-height WGS84 geodetic -> ECEF -> ENU, not fixed scales.
        # https://proj.org/en/stable/operations/conversions/topocentric.html
        # Origin lat/lon, target lat/lon, reference east/north (metres).
        fixtures = (
            (42.4447, -76.4825, 42.4448, -76.4824, 8.227127495, 11.108197551),
            (42.4447, -76.4825, 42.4446, -76.4826, -8.227153664, -11.108187664),
            (-42.4447, 76.4825, -42.4446, 76.4826, 8.227153664, 11.108187664),
            (0.0, 0.0, 0.0001, 0.0001, 11.131949079, 11.057427582),
        )
        for origin_lat, origin_lon, lat, lon, expected_east, expected_north in fixtures:
            with self.subTest(origin=(origin_lat, origin_lon), point=(lat, lon)):
                frame = LocalFrame(origin_lat_deg=origin_lat, origin_lon_deg=origin_lon)
                east, north = frame.to_local(lat_deg=lat, lon_deg=lon)
                # A fixture-level check at ~10 m offsets, not a survey-wide bound.
                self.assertAlmostEqual(east, expected_east, delta=0.0001)
                self.assertAlmostEqual(north, expected_north, delta=0.0001)

    def test_longitude_wraps_across_the_antimeridian(self) -> None:
        for origin_lon, target_lon, expected_east in (
            (179.99995, -179.99995, 11.131949079),
            (-179.99995, 179.99995, -11.131949079),
            (180.0, -180.0, 0.0),
        ):
            with self.subTest(origin_lon=origin_lon):
                frame = LocalFrame(origin_lat_deg=0.0, origin_lon_deg=origin_lon)
                east, north = frame.to_local(lat_deg=0.0, lon_deg=target_lon)
                self.assertAlmostEqual(east, expected_east, delta=0.000001)
                self.assertEqual(north, 0.0)

    def test_invalid_origin_coordinates_are_rejected(self) -> None:
        for name, bad_values in (
            ("origin_lat_deg", (float("nan"), float("inf"), -float("inf"), -90, 90, -91, 91)),
            ("origin_lon_deg", (float("nan"), float("inf"), -float("inf"), -181, 181)),
        ):
            for value in bad_values:
                with self.subTest(coordinate=name, value=value), self.assertRaises(ValueError):
                    origin = {"origin_lat_deg": 42.4447, "origin_lon_deg": -76.4825}
                    origin[name] = value
                    LocalFrame(**origin)

    def test_invalid_target_coordinates_are_rejected(self) -> None:
        frame = LocalFrame(origin_lat_deg=42.4447, origin_lon_deg=-76.4825)
        for name, bad_values in (
            ("lat_deg", (float("nan"), float("inf"), -float("inf"), -91, 91)),
            ("lon_deg", (float("nan"), float("inf"), -float("inf"), -181, 181)),
        ):
            for value in bad_values:
                with self.subTest(coordinate=name, value=value), self.assertRaises(ValueError):
                    point = {"lat_deg": 42.4448, "lon_deg": -76.4824}
                    point[name] = value
                    frame.to_local(**point)


if __name__ == "__main__":
    unittest.main()
