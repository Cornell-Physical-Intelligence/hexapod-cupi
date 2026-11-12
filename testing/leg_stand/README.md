# Leg stand: camera height tracker

A camera at the top of the [leg stand](../../docs/LEG_STAND_HARDWARE.md) rail looks straight down
at two AprilTags on the carriage and reports the carriage height at camera frame rate, in a
browser dashboard. It is an independent position reference for the ENC4110 encoder.

## Hardware

- **Camera:** ELP USBGS720P02 global shutter (onsemi AR0144), USB `32e4:0144`, 1280 × 720 MJPG,
  up to 60 fps. Intrinsics (`cv2.fisheye` model) are in `tag_height.py`. They come from another
  unit of the same model, calibrated on 2026-11-01. The focal length agrees within 0.3 % across
  units; the principal point does not, so set the zero and add reference points.
- **Tag plate** on the orange HexMount: two tag36h11 tags, ids 0 (leg side) and 1 (rail side).
  The black squares are 20.8 mm (8 bits × 2.6 mm), centres 35.55 mm apart, and raised 0.4 mm on a
  62.7 × 34 mm white base. The two HexMount M4s nearest the rail are M4 × 12.
- **Plate frame:** origin midway between the tag centres on the top face; +x toward the rail, +y
  toward the top of the printed tags, +z toward the camera.
- **Camera position:** over the plate centre, at stand CAD (x, y) = (−53.5, 0) mm. The CAD origin
  is the 2020 axis, with z = 0 at the 2020 top (`HEXAPOD_MKII_STAND_ASSY.step`).

## Run

```sh
uv run python testing/leg_stand/tag_height.py selftest            # simulated frames, prints PASS
uv run python testing/leg_stand/tag_height.py serve --camera 0    # http://localhost:8765
```

On macOS, allow Camera access for the terminal app. The MacBook camera can take index 0; pick
the ELP in the dashboard's Camera menu. `serve --sim` shows a simulated plate.

1. With both tags in view, the first 60 two-tag frames lock the camera orientation (`lock.json`).
   Press **Re-lock orientation** after the camera mount is bumped, then set the zero again.
2. Put the carriage at the reference position and press **Set zero here** (1 s average).
3. Move to known offsets (gauge blocks or the encoder), type each offset in mm and press **Add
   point**. The scale is the least-squares fit through the zero; the table shows each error.
   With no points, a move toward the camera reads positive (**Flip direction** reverses it).

The zero and each point need the lock and 1 s of two-tag frames that hold still: the page refuses
an average that moves more than 0.5 mm from its first third to its last (`--steady`). A point
needs a finite offset and at least 5 mm of travel from the zero.

The zero and the points are kept in `calibration.json`. Both JSON files stay out of Git.

## Method

1. **Detect.** OpenCV ArUco `DICT_APRILTAG_36h11` with AprilTag corner refinement, searched in a
   crop around the tags' last position. Each crop is tried as a max-of-channels image (coloured
   filament turns light, black ink stays dark), luma and CLAHE, with loosened thresholds. If the
   tags are lost, a full-frame search runs with one variant per frame.
2. **Undistort** the eight corners with the fisheye model.
3. **Solve.** The first 60 frames that show both tags solve the full pose, and their mean
   orientation is locked. After that only the translation is solved, by linear least squares,
   because the rail only translates the plate.
4. **Fill dropouts.** After the lock, a frame with one tag uses that tag alone. It is corrected by
   that tag's offset from the two-tag height, learned on the two-tag frames.
5. **Report** the lens centre's height above the plate top face, z = a (h − h0) once zeroed. The
   page receives every frame over server-sent events; the big number defaults to a 0.1 s average.

## Measured (9 November 2026, one session, carriage at about 255 mm)

| Quantity | Value |
| --- | --- |
| Camera / processed frame rate | 61 / 58–61 fps (60 fps requested; 15 fps from a single loop before) |
| Frame to number | 10–16 ms |
| Noise | 0.2 mm sd |
| Both tags found | 100 % of frames with the contrast variants |
| Reprojection error | 1.3–1.6 px |
| Camera tilt from the plate normal | 7.35° (the self-test covers 0–1.5°) |
| Tag 0 alone vs both tags | +1.7 mm (tag 1: +0.2 mm), corrected as in step 4 |
| One reference point | 90.00 mm reference read 87.44 mm uncorrected: scale 1.029 |

The self-test renders the plate at 60–470 mm with the fisheye distortion, blur, noise, JPEG, tilt
up to 1.5°, ±5 mm offset and ±5° yaw. With the locked orientation the worst error is about
0.6–1.6 mm at 470 mm, depending on the random poses, and below 0.4 mm up to 300 mm. A free 6-DOF
pose drifts by several millimetres at 470 mm. The simulation does not model lens focus.

## Limits and next steps

- The cause of the scale correction (2.9 %) is open. The focal length agrees within 0.3 % across
  units, so the borrowed intrinsics do not explain it. One candidate: with exact simulated
  corners, squares 0.5 mm wider per edge than 20.8 mm on unchanged centres read a 90 mm move as
  87.4 mm, give a 1.6 px reprojection error at 255 mm and raise the tilt shown by 2.5°. Measure
  the inked squares with calipers. Calibrate this camera with the
  [Camera-Calibration](https://github.com/Cornell-Physical-Intelligence/Camera-Calibration)
  tool, and add at least five reference points plus two held-out points over the full stroke.
- The tags are inked by marker on light-blue filament. Black-on-white tags (printed matte paper on
  the raised squares, or black filament on white) give more contrast; avoid glare.
- Check the 7.35° tilt against the camera mount design.
- Log the height next to the ENC4110 counts once the Pico logger exists.
