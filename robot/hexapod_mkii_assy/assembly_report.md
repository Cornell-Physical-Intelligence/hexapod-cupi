# Hexapod MKII assembly import report

Source: `/Users/andreboufama/Downloads/export (1)` (`hexapod-mkii-assy`), 1927 part instances, Onshape fused mass 5.94233 kg.

Conventions: body frame = export frame (z up), forward = `-y`, leg names l/r = left/right, f/m/r = front/middle/rear; positive coxa_yaw counter-clockwise about body +z; zero pose = femur/tibia: leg reference CAD zero (leg export pose); coxa_yaw zero: leg points radially outward from the body centre.

## Leg mounts (yaw axes in the body frame)

| leg | x (m) | y (m) | z (m) | azimuth | radius (m) | tilt from z | heading at URDF zero yaw | CAD mount heading vs radial | chassis fasteners within 60 mm |
|---|---|---|---|---|---|---|---|---|---|
| lf | +0.0800 | -0.1999 | +0.0528 | -68.2 deg | 0.2153 | 0.004 deg | -68.2 deg | +2.3 deg | 21 |
| lm | +0.1200 | +0.0001 | +0.0528 | +0.0 deg | 0.1200 | 0.004 deg | +0.0 deg | -27.3 deg | 20 |
| lr | +0.0800 | +0.2001 | +0.0528 | +68.2 deg | 0.2155 | 0.004 deg | +68.2 deg | -50.6 deg | 21 |
| rf | -0.0800 | -0.1999 | +0.0528 | -111.8 deg | 0.2153 | 0.004 deg | -111.8 deg | -53.1 deg | 21 |
| rm | -0.1200 | +0.0001 | +0.0528 | +180.0 deg | 0.1200 | 0.004 deg | +180.0 deg | -26.7 deg | 20 |
| rr | -0.0800 | +0.2001 | +0.0528 | +111.8 deg | 0.2155 | 0.004 deg | +111.8 deg | +5.2 deg | 21 |

Yaw zero convention: `radial`. The yaw mates are locked in the CAD, so the CAD mount headings (last column but one) are where the CAD happens to hold each leg, not a designed stance; with `radial` every leg points straight out from the body centre at coxa_yaw = 0 and the CAD placement appears as the coxa_yaw CAD-pose angle below. Heading = direction from the yaw axis to the hip axis, projected on the body xy plane.


## Registration of the leg record onto the export

| leg | body | method | fit | parts |
|---|---|---|---|---|
| lf | leg_base | hinge:coxa_yaw | seed fit 0.733 mm (+0.01 deg); consensus rms 0.0003 mm on 32 parts | 35 (0 moved) |
| lf | coxa | kabsch | seed rms 0.000 mm; consensus rms 0.0902 mm on 50 parts | 63 (13 moved) |
| lf | femur | kabsch | seed rms 0.200 mm; consensus rms 0.0768 mm on 72 parts | 81 (6 moved) |
| lf | tibia | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 46 parts | 46 (0 moved) |
| lf | tibia_push_lever | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 4 parts | 4 (0 moved) |
| lf | tibia_pushrod | kabsch | seed rms 0.000 mm; consensus rms 0.0004 mm on 5 parts | 6 (1 moved) |
| lm | leg_base | hinge:coxa_yaw | seed fit 0.425 mm (+0.00 deg); consensus rms 0.0003 mm on 32 parts | 35 (0 moved) |
| lm | coxa | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 45 parts | 63 (18 moved) |
| lm | femur | kabsch | seed rms 0.200 mm; consensus rms 0.0003 mm on 66 parts | 81 (12 moved) |
| lm | tibia | kabsch | seed rms 0.000 mm; consensus rms 0.0002 mm on 46 parts | 46 (0 moved) |
| lm | tibia_push_lever | kabsch | seed rms 0.000 mm; consensus rms 0.0001 mm on 4 parts | 4 (0 moved) |
| lm | tibia_pushrod | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 5 parts | 6 (1 moved) |
| lr | leg_base | hinge:coxa_yaw | seed fit 0.224 mm (+0.00 deg); consensus rms 0.0003 mm on 32 parts | 35 (0 moved) |
| lr | coxa | kabsch | seed rms 0.000 mm; consensus rms 0.0908 mm on 54 parts | 63 (9 moved) |
| lr | femur | kabsch | seed rms 0.200 mm; consensus rms 0.0572 mm on 72 parts | 81 (6 moved) |
| lr | tibia | kabsch | seed rms 0.000 mm; consensus rms 0.0005 mm on 46 parts | 46 (0 moved) |
| lr | tibia_push_lever | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 4 parts | 4 (0 moved) |
| lr | tibia_pushrod | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 5 parts | 6 (1 moved) |
| rf | leg_base | hinge:coxa_yaw | seed fit 0.661 mm (+0.00 deg); consensus rms 0.0003 mm on 32 parts | 35 (0 moved) |
| rf | coxa | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 63 parts | 63 (0 moved) |
| rf | femur | kabsch | seed rms 0.200 mm; consensus rms 0.0004 mm on 72 parts | 81 (6 moved) |
| rf | tibia | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 46 parts | 46 (0 moved) |
| rf | tibia_push_lever | kabsch | seed rms 0.000 mm; consensus rms 0.0002 mm on 4 parts | 4 (0 moved) |
| rf | tibia_pushrod | kabsch | seed rms 0.000 mm; consensus rms 0.0005 mm on 5 parts | 6 (1 moved) |
| rm | leg_base | hinge:coxa_yaw | seed fit 0.426 mm (+0.00 deg); consensus rms 0.0003 mm on 32 parts | 35 (0 moved) |
| rm | coxa | kabsch | seed rms 0.000 mm; consensus rms 0.0004 mm on 45 parts | 63 (18 moved) |
| rm | femur | kabsch | seed rms 0.200 mm; consensus rms 0.0003 mm on 66 parts | 81 (12 moved) |
| rm | tibia | kabsch | seed rms 0.000 mm; consensus rms 0.0002 mm on 46 parts | 46 (0 moved) |
| rm | tibia_push_lever | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 4 parts | 4 (0 moved) |
| rm | tibia_pushrod | kabsch | seed rms 0.000 mm; consensus rms 0.0002 mm on 5 parts | 6 (1 moved) |
| rr | leg_base | hinge:coxa_yaw | seed fit 0.084 mm (+0.00 deg); consensus rms 0.0002 mm on 32 parts | 35 (3 moved) |
| rr | coxa | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 63 parts | 63 (0 moved) |
| rr | femur | kabsch | seed rms 0.200 mm; consensus rms 0.0004 mm on 72 parts | 81 (6 moved) |
| rr | tibia | kabsch | seed rms 0.000 mm; consensus rms 0.0004 mm on 46 parts | 46 (0 moved) |
| rr | tibia_push_lever | kabsch | seed rms 0.000 mm; consensus rms 0.0004 mm on 4 parts | 4 (0 moved) |
| rr | tibia_pushrod | kabsch | seed rms 0.000 mm; consensus rms 0.0003 mm on 5 parts | 6 (1 moved) |

Parts whose CAD position differs from the leg record by more than 0.5 mm (assigned to the nearest expected position; these are free-spinning motor internals and re-solved fasteners, not structure):

- coxa: `hex_socket_head_countersunk_screw_grade_c_m3x0_5_x_10.stl` x12 (up to 0.77 mm)
- coxa: `motor_0001755652_00mini_000.stl` x2 (up to 4.96 mm)
- coxa: `motor_0001755652_1_1_30_gr_2202.stl` x6 (up to 4.96 mm)
- coxa: `motor_1_2_05_020051.stl` x8 (up to 0.74 mm)
- coxa: `motor_1_2_05_020051_2.stl` x6 (up to 0.74 mm)
- femur: `cirpattern2.stl` x9 (up to 0.50 mm)
- femur: `femur_plate_motor_spacer.stl` x2 (up to 0.50 mm)
- femur: `femur_plate_non_motor.stl` x2 (up to 0.50 mm)
- femur: `mirror1.stl` x9 (up to 0.50 mm)
- tibia_pushrod: `m5_nylon_lubricant_filled_nylon_washer__5_5mm_id__10mm_od__1` x5 (up to 0.50 mm)

## Hinge axes carried through parent and child registrations

| leg | joint | dir mismatch | line gap | CAD-pose angle |
|---|---|---|---|---|
| lf | coxa_yaw | 0.055 deg | 0.041 mm | +0.0407 rad (+2.33 deg) |
| lf | femur_pitch | 0.058 deg | 0.039 mm | -0.0280 rad (-1.61 deg) |
| lf | tibia_pitch | 0.000 deg | 0.009 mm | +0.0280 rad (+1.60 deg) |
| lf | tibia_lever_pivot | 0.000 deg | 0.003 mm | +0.0280 rad (+1.60 deg) |
| lf | tibia_rod_pivot | 0.009 deg | 0.000 mm | -0.0281 rad (-1.61 deg) |
| lm | coxa_yaw | 0.000 deg | 0.000 mm | -0.4771 rad (-27.34 deg) |
| lm | femur_pitch | 0.000 deg | 0.000 mm | -0.0402 rad (-2.30 deg) |
| lm | tibia_pitch | 0.000 deg | 0.000 mm | +0.0402 rad (+2.30 deg) |
| lm | tibia_lever_pivot | 0.000 deg | 0.000 mm | +0.0402 rad (+2.30 deg) |
| lm | tibia_rod_pivot | 0.009 deg | 0.000 mm | -0.0402 rad (-2.30 deg) |
| lr | coxa_yaw | 0.030 deg | 0.022 mm | -0.8832 rad (-50.60 deg) |
| lr | femur_pitch | 0.027 deg | 0.006 mm | -0.0206 rad (-1.18 deg) |
| lr | tibia_pitch | 0.000 deg | 0.007 mm | +0.0208 rad (+1.19 deg) |
| lr | tibia_lever_pivot | 0.000 deg | 0.002 mm | +0.0209 rad (+1.19 deg) |
| lr | tibia_rod_pivot | 0.009 deg | 0.000 mm | -0.0209 rad (-1.20 deg) |
| rf | coxa_yaw | 0.000 deg | 0.000 mm | -0.9262 rad (-53.07 deg) |
| rf | femur_pitch | 0.000 deg | 0.000 mm | -0.0000 rad (-0.00 deg) |
| rf | tibia_pitch | 0.000 deg | 0.000 mm | +0.0000 rad (+0.00 deg) |
| rf | tibia_lever_pivot | 0.000 deg | 0.000 mm | +0.0000 rad (+0.00 deg) |
| rf | tibia_rod_pivot | 0.009 deg | 0.000 mm | -0.0000 rad (-0.00 deg) |
| rm | coxa_yaw | 0.000 deg | 0.000 mm | -0.4660 rad (-26.70 deg) |
| rm | femur_pitch | 0.000 deg | 0.000 mm | +0.0372 rad (+2.13 deg) |
| rm | tibia_pitch | 0.000 deg | 0.001 mm | -0.0372 rad (-2.13 deg) |
| rm | tibia_lever_pivot | 0.000 deg | 0.000 mm | -0.0372 rad (-2.13 deg) |
| rm | tibia_rod_pivot | 0.010 deg | 0.000 mm | +0.0372 rad (+2.13 deg) |
| rr | coxa_yaw | 0.000 deg | 0.000 mm | +0.0913 rad (+5.23 deg) |
| rr | femur_pitch | 0.000 deg | 0.000 mm | +0.0000 rad (+0.00 deg) |
| rr | tibia_pitch | 0.000 deg | 0.000 mm | -0.0000 rad (-0.00 deg) |
| rr | tibia_lever_pivot | 0.000 deg | 0.000 mm | +0.0000 rad (+0.00 deg) |
| rr | tibia_rod_pivot | 0.009 deg | 0.000 mm | -0.0000 rad (-0.00 deg) |

## Part assignment

| leg | yaw output side (body) | coxa | femur | tibia | push lever | pushrod | worst residual (excl. free motor internals) | free motor internals |
|---|---|---|---|---|---|---|---|---|
| lf | 2 | 122 | 109 | 46 | 11 | 6 | 1.624 mm | 3 |
| lm | 2 | 122 | 109 | 46 | 11 | 6 | 4.959 mm | 2 |
| lr | 2 | 122 | 109 | 46 | 11 | 6 | 2.426 mm | 2 |
| rf | 2 | 122 | 109 | 46 | 11 | 6 | 1.623 mm | 3 |
| rm | 2 | 122 | 109 | 46 | 11 | 6 | 4.964 mm | 0 |
| rr | 2 | 122 | 109 | 46 | 11 | 6 | 1.624 mm | 0 |

Matched 1746 instances (354 fasteners matched by size class because the catalogue part changed), 151 chassis parts, 18 new leg parts adopted by keyword, 0 leg-record parts missing from the assembly, 0 chassis parts within 5 mm of a leg part (review).

New leg parts (not in the leg record): femur_plate_cover_large.stl -> femur x6, femur_plate_cover_small.stl -> femur x6, tibia_pushrod_cover.stl -> tibia_pushrod x6

## Mass model

Source: Onshape per-part mass properties (robot.pkl), parallel-axis sums per link; each of the 18 RS05 actuators overridden from its CAD shell mass to 191 g on its housing link. Chassis structure alone 1.7566 kg, plus the six yaw-motor output sides (flange + hub, bolted through the frame) 0.0863 kg in the body link; the motor bodies, their top plates and the screws on them rotate with the coxae. Legs lf 1.0840, lm 1.0840, lr 1.0840, rf 1.0840, rm 1.0840, rr 1.0840 kg (each including its yaw output side).

| structural part | count | mass each (g) | density (kg/m3) |
|---|---|---|---|
| machined_tibia.stl | 6 | 99.34 | 1180 |
| bottom_enclosure.stl | 1 | 409.98 | 2810 |
| top_enclosure.stl | 1 | 404.27 | 2810 |
| motor_1_1_06_eb463_507.stl | 18 | 21.83 | 2684 |
| standoff_plate.stl | 4 | 95.70 | 2681 |
| bottom_plate.stl | 1 | 250.76 | 2680 |
| tibia_push_lever.stl | 6 | 40.35 | 7856 |
| first_joint_bottom_plate.stl | 6 | 33.00 | 2681 |
| motor_1_1_02_eb461_502.stl | 18 | 10.04 | 2683 |
| first_joint_spacer.stl | 6 | 29.60 | 1300 |
| motor_bearing_holder.stl | 12 | 14.52 | 2703 |
| tibia_pushrod.stl | 6 | 28.16 | 7851 |
| femur_plate.stl | 6 | 26.22 | 2680 |
| standoff_plate__2.stl | 2 | 77.02 | 2681 |
| motor_0001755650_00mini_000.stl | 18 | 8.10 | 2683 |
| femur_plate_non_motor.stl | 6 | 20.55 | 2680 |
| silicone_foot.stl | 6 | 20.22 | 1100 |
| motor_1_1_06_eb463_509.stl | 18 | 6.37 | 2682 |
| motor_1_2_16_001425.stl | 18 | 5.65 | 2682 |
| bearing_insert.stl | 18 | 5.56 | 2703 |
| first_joint_top.stl | 6 | 13.63 | 2681 |
| motor_flange.stl | 12 | 6.28 | 2683 |
| femur_plate_cover_large.stl | 6 | 12.33 | 1052 |
| femur_plate_cover_small.stl | 6 | 8.74 | 1052 |
| femur_plate_motor_spacer.stl | 6 | 4.29 | 1301 |
| tibia_pushrod_cover.stl | 6 | 3.48 | 1052 |

Leg record (v3, derived from the older v2 per-part export) versus this assembly's per-part properties, per body:

| body | record mass (kg) | assembly mass (kg) | delta (g) | COM shift (mm) | inertia trace ratio |
|---|---|---|---|---|---|
| leg_base | 0.0967 | 0.0144 | -82.3 | 18.56 | 0.02 |
| coxa | 0.1199 | 0.3709 | +251.0 | 21.63 | 2.41 |
| femur | 0.2110 | 0.4592 | +248.2 | 2.24 | 2.15 |
| tibia | 0.2849 | 0.1520 | -132.9 | 3.57 | 0.62 |
| tibia_push_lever | 0.0431 | 0.0547 | +11.6 | 0.81 | 1.26 |
| tibia_pushrod | 0.0332 | 0.0329 | -0.4 | 0.35 | 0.99 |

Actuators: the RS05 vendor CAD is a hollow shell weighing 62.2 g (min 62.2, max 62.2); each of the 18 motors is overridden to 191 g, the difference placed on the link that carries its housing as a solid cylinder the size of the housing (+2.318 kg in total). Everything else keeps its CAD mass, so the assembled mass exceeds Onshape's fused value by exactly that amount; `--motor-mass 0` reproduces the CAD masses.

Assembled: 8.26081 kg vs Onshape fused 5.94233 kg; COM error 0.430 mm; worst inertia term error 34.100 % of the largest term.

| link | mass (kg) |
|---|---|
| lf yaw output side (in body) | 0.01438 |
| lf_coxa | 0.37089 |
| lf_femur | 0.45918 |
| lf_tibia | 0.15196 |
| lf_tibia_push_lever | 0.05474 |
| lf_tibia_pushrod | 0.03288 |
| lm yaw output side (in body) | 0.01438 |
| lm_coxa | 0.37089 |
| lm_femur | 0.45918 |
| lm_tibia | 0.15196 |
| lm_tibia_push_lever | 0.05474 |
| lm_tibia_pushrod | 0.03288 |
| lr yaw output side (in body) | 0.01438 |
| lr_coxa | 0.37089 |
| lr_femur | 0.45918 |
| lr_tibia | 0.15196 |
| lr_tibia_push_lever | 0.05474 |
| lr_tibia_pushrod | 0.03288 |
| rf yaw output side (in body) | 0.01438 |
| rf_coxa | 0.37089 |
| rf_femur | 0.45918 |
| rf_tibia | 0.15196 |
| rf_tibia_push_lever | 0.05474 |
| rf_tibia_pushrod | 0.03288 |
| rm yaw output side (in body) | 0.01438 |
| rm_coxa | 0.37089 |
| rm_femur | 0.45918 |
| rm_tibia | 0.15196 |
| rm_tibia_push_lever | 0.05474 |
| rm_tibia_pushrod | 0.03288 |
| rr yaw output side (in body) | 0.01438 |
| rr_coxa | 0.37089 |
| rr_femur | 0.45918 |
| rr_tibia | 0.15196 |
| rr_tibia_push_lever | 0.05474 |
| rr_tibia_pushrod | 0.03288 |
| body | 1.84287 |

## Leg-internal joint transforms (measured from the assembly, mean of six legs)

| joint | origin in parent frame (m) | rpy | shift vs leg record (mm) | worst leg deviation from mean |
|---|---|---|---|---|
| femur_pitch | -0.04900 -0.02500 +0.02551 | +1.5705 -0.0001 -0.0000 | -0.002 +0.007 +0.001 (0.014 deg) | 0.0318 mm / 0.0437 deg |
| tibia_pitch | -0.10062 -0.10441 -0.00676 | +0.0000 +0.0000 +0.0000 | -0.002 +0.002 +0.500 (0.000 deg) | 0.0065 mm / 0.0012 deg |
| tibia_lever_pivot | -0.04683 -0.04861 -0.00550 | +0.0000 +0.0000 +0.0000 | -0.001 +0.000 -0.000 (0.001 deg) | 0.0019 mm / 0.0018 deg |
| tibia_rod_pivot | +0.01019 -0.02821 +0.00330 | -0.0000 -0.0001 +0.0000 | +0.000 -0.000 +0.000 (0.003 deg) | 0.0006 mm / 0.0020 deg |

Mount cross-check (coxa zero-yaw pose from the mount-plate registration vs from the coxa registration): worst 0.042 mm / 0.055 deg.

## Round trip

URDF forward kinematics at the measured CAD-pose joint angles reproduces every registered link pose to 0.0866 mm / 0.0436 deg.

Foot pad lowest point in the CAD pose (m, body frame): lf -0.0908, lm -0.0921, lr -0.0901, rf -0.0880, rm -0.0843, rr -0.0880
