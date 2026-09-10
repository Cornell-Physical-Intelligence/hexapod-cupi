# Stance audit: why tibia length and body height changed the shortlist

The original Stage 1 shortlist did not establish the best body posture for steady walking. C remains a provisional geometry candidate; the earlier tall poses are examples, not required geometry or validated walking stances. A remains excluded by the user for motor packaging. Stage 2 is still held.

## Torque explanation

C and D used the same nominal femur/knee angles (10/110 degrees), but extending a tibia that points inward brought the foot closer to the hip. At the nominal left-front pose, the radial foot-to-hip offset fell from 61.45 mm for C to 53.66 mm for D; the knee-to-foot offset magnitude increased from 9.95 to 17.74 mm. Longer links do not increase every joint's perpendicular force lever arm together.

The full load calculation balances link gravity, motion, motor terms and 3D contact forces. The contact solver minimizes the largest motor torque at every phase and can change horizontal force sharing. It does not prescribe a realizable feedback controller or smooth force transitions. The 1.381 versus 1.349 N·m difference is about 2.3%, not decisive evidence for the longer tibia. Force-to-joint torque follows the Jacobian-transpose relationship explained in [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-2-statics-of-open-chains/).

## Selection gap and additional checks

The original pose selection retained minimum-static-torque seeds per knee angle and minimum-clearance threshold. A minimum-height threshold can repeatedly select a taller posture. Lower poses with slightly higher static load but better motion headroom could be discarded before the path screen. The screen also prescribed a level, constant-velocity body and therefore did not measure roll, pitch, heave or disturbance recovery. The previous recommendation overstated how much those results supported overall steady walking.

The additional CPU audit evaluated all retained original static-grid poses between 80 and 140 mm belly clearance for C and D: 459 trajectory configurations, with tripod/ripple/wave schedules, 40/60/100 mm strides, 20 mm lift and 0.05/0.10/0.20 m/s targets. This is an expanded finite search, not an exhaustive workspace or gait proof. The 40 mm stride extends the original trajectory range.

For each geometry, the lowest estimated peak at 0.20 m/s within 95–120 mm belly clearance was rechecked at 256 phase samples. The heights differ, so this is a low-posture comparison rather than an exactly matched-height experiment.

| Geometry | Belly clearance | Femur / knee angles | Peak at 0.10 m/s | Peak at 0.20 m/s | 0.20 m/s screen |
|---|---:|---:|---:|---:|---|
| f050_t060 | 107.5 mm | 40 / 120° | 1.277 N·m | 1.387 N·m | Pass |
| f050_t080 | 117.0 mm | 60 / 120° | 1.509 N·m | 1.650 N·m | Fail |

![Same geometry in taller and lower poses](/Users/andreboufama/Documents/CUPI/HEXAPOD/artifacts/length_study_2026-09-09/stance_audit_all_low_poses/same_lengths_lower_stance.png)

Lower body height can reduce the overturning moment from horizontal acceleration at a given support footprint. The actual benefit depends on whole-robot COM height, foot placement, contact forces, available joint travel and ground clearance. A visually crouched pose alone is not a stability or efficiency certificate.

Next mechanical comparison should explicitly vary body height and stance width, preserve motion headroom before pruning poses, and present the torque/work/workspace tradeoffs at common absolute heights. Extend that comparison to the remaining geometries before using the original shortlist as a final ranking. Keep production hardware fit and real joint/linkage limits separate from the mock sensitivity study. No GPU trial has started.
