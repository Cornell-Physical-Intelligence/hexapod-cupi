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
