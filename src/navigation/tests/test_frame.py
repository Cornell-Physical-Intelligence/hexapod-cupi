"""Checks for the draft fixed-scale WGS84 converter and polygon helpers."""

from __future__ import annotations

import unittest

from shapely import MultiPolygon, Point, Polygon

from navigation.frame import LocalFrame


class LocalFrameTests(unittest.TestCase):
    def test_origin_is_read_only_and_maps_to_zero(self) -> None:
        frame = LocalFrame(origin_lon_deg=-76.4825, origin_lat_deg=42.4447)
        self.assertEqual(frame.origin_lat_deg, 42.4447)
        self.assertEqual(frame.origin_lon_deg, -76.4825)
        self.assertEqual(frame.to_local(-76.4825, 42.4447), (0.0, 0.0))
        for name in ("origin_lat_deg", "origin_lon_deg"):
            with self.subTest(property=name), self.assertRaises(AttributeError):
                setattr(frame, name, 0.0)

    def test_east_and_north_axes_have_the_documented_order(self) -> None:
        frame = LocalFrame(origin_lon_deg=-76.4825, origin_lat_deg=42.4447)
        east, north = frame.to_local(-76.4824, 42.4447)
        self.assertGreater(east, 0.0)
        self.assertEqual(north, 0.0)
        east, north = frame.to_local(-76.4825, 42.4448)
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
                frame = LocalFrame(origin_lon_deg=origin_lon, origin_lat_deg=origin_lat)
                east, north = frame.to_local(lon, lat)
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
                frame = LocalFrame(origin_lon_deg=origin_lon, origin_lat_deg=0.0)
                east, north = frame.to_local(target_lon, 0.0)
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
        frame = LocalFrame(origin_lon_deg=-76.4825, origin_lat_deg=42.4447)
        for name, bad_values in (
            ("lat_deg", (float("nan"), float("inf"), -float("inf"), -91, 91)),
            ("lon_deg", (float("nan"), float("inf"), -float("inf"), -181, 181)),
        ):
            for value in bad_values:
                with self.subTest(coordinate=name, value=value), self.assertRaises(ValueError):
                    point = {"lat_deg": 42.4448, "lon_deg": -76.4824}
                    point[name] = value
                    frame.to_local(**point)

    def test_inverse_origin_and_independent_reference_offsets(self) -> None:
        frame = LocalFrame(origin_lon_deg=-76.4825, origin_lat_deg=42.4447)
        self.assertEqual(frame.to_lon_lat(0.0, 0.0), (-76.4825, 42.4447))
        # Same independent ECEF/ENU fixtures used by the forward checks.
        for origin_lat, origin_lon, east, north, expected_lat, expected_lon in (
            (42.4447, -76.4825, 8.227127495, 11.108197551, 42.4448, -76.4824),
            (0.0, 0.0, 11.131949079, 11.057427582, 0.0001, 0.0001),
        ):
            with self.subTest(origin=(origin_lat, origin_lon)):
                frame = LocalFrame(origin_lon_deg=origin_lon, origin_lat_deg=origin_lat)
                lon, lat = frame.to_lon_lat(east, north)
                self.assertAlmostEqual(lat, expected_lat, delta=0.000000001)
                self.assertAlmostEqual(lon, expected_lon, delta=0.000000001)

    def test_local_round_trips_including_longitude_wrap(self) -> None:
        for origin_lat, origin_lon in ((42.4447, -76.4825), (-42.0, 76.0), (0.0, 179.99995)):
            frame = LocalFrame(origin_lon_deg=origin_lon, origin_lat_deg=origin_lat)
            for east, north in ((0.0, 0.0), (100.0, 100.0), (-50.0, -100.0)):
                with self.subTest(origin=(origin_lat, origin_lon), point=(east, north)):
                    lon, lat = frame.to_lon_lat(east, north)
                    self.assertGreaterEqual(lon, -180.0)
                    self.assertLessEqual(lon, 180.0)
                    restored = frame.to_local(lon, lat)
                    self.assertAlmostEqual(restored[0], east, delta=0.00000001)
                    self.assertAlmostEqual(restored[1], north, delta=0.00000001)

    def test_invalid_inverse_inputs_are_rejected(self) -> None:
        frame = LocalFrame(origin_lon_deg=-76.4825, origin_lat_deg=42.4447)
        for bad in (float("nan"), float("inf"), -float("inf")):
            for axis in ("east_m", "north_m"):
                with self.subTest(axis=axis, value=bad), self.assertRaises(ValueError):
                    point = {"east_m": 0.0, "north_m": 0.0}
                    point[axis] = bad
                    frame.to_lon_lat(**point)
        for north in (-1e9, 1e9):
            with self.subTest(north=north), self.assertRaises(ValueError):
                frame.to_lon_lat(east_m=0.0, north_m=north)

    def test_inverse_overflow_is_rejected(self) -> None:
        frame = LocalFrame(origin_lon_deg=0.0, origin_lat_deg=89.9999)
        with self.assertRaises(ValueError):
            frame.to_lon_lat(east_m=1e308, north_m=0.0)


class LocalFramePolygonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = LocalFrame(origin_lon_deg=-76.4825, origin_lat_deg=42.4447)
        self.geographic = Polygon(
            [
                (-76.4825, 42.4447),
                (-76.4821354, 42.4447),
                (-76.4821354, 42.4448800),
                (-76.4825, 42.4448800),
            ],
            holes=[
                [
                    (-76.4823420, 42.4447720),
                    (-76.4822934, 42.4447720),
                    (-76.4822934, 42.4448080),
                    (-76.4823420, 42.4448080),
                ]
            ],
        )

    def test_geographic_polygon_order_and_dimensions(self) -> None:
        original = self.geographic.wkb
        local = self.frame.polygon_to_local(self.geographic)
        # V1's rounded geographic fixture represents a ~30 x 20 m boundary.
        for actual, expected in zip(local.bounds, (0.0, 0.0, 29.996, 19.995), strict=True):
            self.assertAlmostEqual(actual, expected, delta=0.001)
        self.assertEqual(len(local.interiors), 1)
        for actual, expected in zip(
            Polygon(local.interiors[0]).bounds, (12.999, 7.998, 16.997, 11.997), strict=True
        ):
            self.assertAlmostEqual(actual, expected, delta=0.001)
        self.assertEqual(self.geographic.wkb, original)

    def test_polygon_round_trip_preserves_vertex_order_and_holes(self) -> None:
        local = self.frame.polygon_to_local(self.geographic)
        restored = self.frame.polygon_to_lon_lat(local)
        self.assertTrue(restored.equals_exact(self.geographic, tolerance=1e-12))

    def test_empty_polygons_stay_empty(self) -> None:
        for convert in (self.frame.polygon_to_local, self.frame.polygon_to_lon_lat):
            with self.subTest(method=convert.__name__):
                result = convert(Polygon())
                self.assertIsInstance(result, Polygon)
                self.assertTrue(result.is_empty)

    def test_non_polygon_inputs_are_rejected(self) -> None:
        for convert in (self.frame.polygon_to_local, self.frame.polygon_to_lon_lat):
            for value in (None, Point(0, 0), MultiPolygon([self.geographic])):
                with self.subTest(method=convert.__name__, value=value):
                    with self.assertRaises(TypeError):
                        convert(value)

    def test_non_2d_coordinates_are_rejected(self) -> None:
        polygon = Polygon([(0, 0, 1), (1, 0, 1), (1, 1, 1)])
        for convert in (self.frame.polygon_to_local, self.frame.polygon_to_lon_lat):
            with self.subTest(method=convert.__name__), self.assertRaises(ValueError):
                convert(polygon)

    def test_invalid_vertex_coordinates_are_rejected(self) -> None:
        for polygon in (
            Polygon([(0, 0), (181, 0), (1, 1)]),
            Polygon([(0, 0), (1, 91), (1, 1)]),
            Polygon([(0, 0), (float("inf"), 0), (1, 1)]),
        ):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                self.frame.polygon_to_local(polygon)
        with self.assertRaises(ValueError):
            self.frame.polygon_to_lon_lat(Polygon([(0, 0), (1, 1e9), (2, 0)]))
        with self.assertRaises(ValueError):
            self.frame.polygon_to_lon_lat(Polygon([(0, 0), (float("inf"), 0), (1, 1)]))

    def test_self_crossing_polygon_is_not_repaired(self) -> None:
        crossing = Polygon([(0, 0), (0.0001, 0.0001), (0.0001, 0), (0, 0.0001)])
        frame = LocalFrame(origin_lon_deg=0.0, origin_lat_deg=0.0)
        local = frame.polygon_to_local(crossing)
        self.assertFalse(local.is_valid)
        self.assertTrue(frame.polygon_to_lon_lat(local).equals_exact(crossing, tolerance=1e-12))


if __name__ == "__main__":
    unittest.main()
