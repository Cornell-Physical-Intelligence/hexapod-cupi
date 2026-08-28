# Physical sensor references

This directory stores manufacturer-originated geometry and documentation used
to build the simulated perception rig.  Binary source assets are downloaded
from the manufacturers and retained without modification; derived USD assets
and simulator configuration belong in sibling `usd/` and `config/` folders.

## Selected Phase 3 rig

- Intel RealSense D455 stereo depth camera. Isaac Sim 6.0.1 provides the
  RealSense-certified digital twin at
  `/Isaac/Sensors/RealSense/D455/rsd455.usd`.
- Livox Mid-360 lidar. The `livox_mid360/source/` folder contains Livox's
  official device and field-of-view STEP models plus the user manual.

The selected Mid-360 is a **near-hemispherical** 360 x 59-degree sensor
(360-degree azimuth, -7 to +52-degree elevation), not a full hemisphere. The
phrase "hemispherical lidar" could instead mean the Unitree 4D LiDAR L1, whose
published FoV is 360 x 90 degrees. Confirm the physical label or provide a clear
photo before treating either identity as final; a Unitree label requires a
different simulation profile.

## Published reference envelope

| Device | Dimensions | Mass | Power | FoV | Range |
| --- | --- | ---: | ---: | --- | --- |
| Livox Mid-360 | 65 x 65 x 60 mm | 0.265 kg | 6.5 W average; 14 W cold peak | 360 x 59 degrees | 0.10 m minimum; 40 m at 10% and 70 m at 80% reflectivity |
| RealSense D455 | 124 x 26 x 29 mm | 0.116 kg | 3.46147 W maximum operating mode | 87 x 58 degrees depth | 0.52 m minimum at maximum resolution; 0.6--6 m ideal |

The two sensors total 0.381 kg. Added only as accounting to the existing 1.500
kg sensor-free body and 6.300 kg robot, they project to 1.881 kg body payload
and 6.681 kg complete robot before brackets, cables, and compute. These values
are not approximate inertia and are not applied to the Phase1/2 or Phase3
articulation.

The difficult-terrain sensor-fusion training job is approval-gated. Sensor
files, mount configuration, accounting, and short smoke tests do not authorize
a long or overnight run.

## Mount placement study

`tools/lidar_placement_study.py` scores candidate Mid-360 mounts by
self-occlusion. It runs forward kinematics on
`robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf` at a fixed
stance, transforms the visual meshes into world space, and rasterizes them into
a spherical occupancy grid seen from each candidate optical centre. No
simulator, GPU, or Isaac Sim install is involved; it needs only numpy and runs
in about a second.

Sensor geometry comes from `livox_mid360/config/mid360_sensor.json`, which is a
transcription of the manual plus two numbers measured from Livox's own STEP
models: the FOV shell in `mid-360-fov-asm.stp` is centred on the STEP origin,
and `mid-360-asm.stp` places the housing from -25.92 mm to +34.18 mm about that
origin (60.10 mm, matching the 60.0 mm datasheet height). The optical centre
therefore sits **25.92 mm above the bottom mounting face**.
`isaaclab/tests/test_lidar_placement_study.py` re-derives that offset from the
STEP file and binds the field of view to the copy in
`isaaclab/hexapod_phase3/sensor_cfg.py`.

Candidate mounts live in `livox_mid360/config/mounts.json` as poses of the
mounting plate in the imported body frame, where the deck top face is z = 0.
Results are recorded under `artifacts/sensor_placement/livox_mid360/`.

### What the geometry forces

The Mid-360's vertical FOV runs from -7 to +52 degrees, so all but seven
degrees of it points above the horizon. On a robot whose deck sits at 0.185 m
the -7 degree lower edge alone puts the nearest possible ground return at
`h / tan(7 deg)`, which is 1.7 m for a flush mount and grows with any mast.
Raising the sensor improves nothing about terrain and makes it worse. Three
consequences follow, all measured at the Stage2C stance:

1. **Upright, the Mid-360 cannot see the robot's own footing.** Nearest ground
   is 1.75 m flush, 2.37 m on a 75 mm mast, 3.41 m on a 200 mm mast. Local
   terrain is the D455's job; the Mid-360's job is mid-field mapping,
   obstacle detection, and lidar-inertial odometry.
2. **A 75 mm mast is the knee for leg occlusion.** Flush on the deck the legs
   and frame block 18.8 percent of the below-horizon band and close 11 percent
   of azimuths entirely. That falls to 5.9 percent at the mount Phase 3
   currently assumes, 2.5 percent at 60 mm, and 0.0 percent at 75 mm, where
   the plate stands 75 mm above the deck top and the optical centre 0.286 m
   above the ground. Past 75 mm nothing further is gained.
3. **Pitching the sensor down hits a floor at about 0.58 m.** Forward ground
   improves to 0.89 m at 10 degrees and 0.59 m at 20 degrees, then stops: the
   deck edge, not the FOV, becomes the limit. Beyond 20 degrees the extra tilt
   only destroys the 360-degree property, dropping usable azimuths from 100 to
   about 55 percent and raising total FOV occlusion from 0.3 to 27 percent.

Inverted mounts reach the ground sooner (0.15 to 0.44 m) but Livox's manual
states, on p. 9, "If mounting the Mid-360 upside down, allow a space of no less
than 0.5 m between the mounting surface and the ground." No deck-mounted
inverted pose on a 0.185 m robot meets that, so they are recorded for reference
and not recommended.

Two further manual constraints bear on the bracket rather than the pose: the
Mid-360 "cannot bear any extra payload", and Livox recommends a metal base
plate at least 3 mm thick with at least 10000 mm2 exposed to air, which is a
100 x 100 mm plate, larger than the 65 x 65 mm sensor footprint.

### Limits of the study

One static stance, one flat ground plane, and the robot's own visual meshes.
The mast is modelled as a plain cylinder; brackets, cabling, the D455, the
sensor housing itself, and the swept leg envelope over a gait cycle are not.
The occupancy grid is conservative by up to half a cell at grazing edges.
Treat every number as the geometric ceiling for a mount, not as a predicted
point cloud, and confirm a shortlisted pose in Isaac Sim before committing to
a bracket.

### Confirmed in Isaac Sim

`isaaclab/hexapod_phase3/sensor_cfg.py` originally ray-cast the Mid-360 against
only the ground plane and the calibration wall, so the simulated sensor
reported a clean 360 degrees by construction and could not test any mount.
`SELF_OCCLUSION_MESH_EXPRESSIONS` now adds the robot's own nineteen leaf
meshes as tracked targets, and `isaaclab/phase3_lidar_placement_probe.py`
measures the same quantities the offline study does. Results at the default
stance, which settles with the deck at 0.201 m, are in
`artifacts/sensor_placement/livox_mid360/sim/`:

| Optical centre above deck | Sim blocked below horizon | Offline | Sim ground forward | Offline |
| --- | ---: | ---: | ---: | ---: |
| 26 mm, flush | 15.5% | 16.2% | 1.86 m | 1.88 m |
| 75 mm, the mount Phase 3 assumed | 0.2% | 0.1% | 2.26 m | 2.29 m |
| 101 mm, a 75 mm plate mast | 0.0% | 0.0% | 2.47 m | 2.50 m |

At 101 mm, zero of 20000 rays returned off the robot, on two independent runs.
Two implementations that share no code agree across the range, so the offline
study can be trusted for the rest of the candidate list.

Three notes for whoever runs this next. Ray-cast target expressions are matched
by a helper that rewrites a bare `*` and scopes `.*` to one path segment, and
the tibia link is an instanceable prim with no mesh child while coxa and femur
are not, so the four expressions are not interchangeable. The robot prim also
carries the D455 visual asset, which has no `RigidBodyAPI`; any target broad
enough to reach it aborts the run inside the PhysX tensor view. And livestream
must be requested through the `LIVESTREAM=1` environment variable: the
`--livestream` CLI flag parsed without error and silently produced no stream.
