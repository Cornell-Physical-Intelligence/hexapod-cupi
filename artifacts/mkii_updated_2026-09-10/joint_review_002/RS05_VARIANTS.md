# RS05 mass variants — reviewed joint-travel successor

The raw CAD and nominal RS05 mass lineages retain exactly the preceding per-link mass, COM, inertia and 18 housing additions. Raw mass is 5.147603654203134 kg; nominal-motor-mass total is 7.466088235225788 kg. No new mass allocation was introduced.

All three URDF variants now use the current joint review: femur [-120, +80] degrees and tibia [-5, +180] degrees about unchanged pitch zeros; coxa zero and bounds use each reviewed standoff interval midpoint. These are travel limits, not a collision-free motion envelope. See BUILD_REPORT.md and joint_review.json for current coordinate definitions.

The RS05 URDF variants retain sourced 5.5 N·m peak effort and 50.26548245743669 rad/s maximum speed; the inspection-only file retains disabled physical actuation. Runtime torque-speed, continuous-duty and hardware protection requirements remain.

rs05_mass_correction.json identifies current corresponding raw-file hashes and separately preserves the original correction calculation hashes. Its full per-link before/after tables and additions are unchanged. previous_RS05_VARIANTS.md is frozen prior provenance, including its old narrow envelope; it is not current travel documentation. No Isaac admission is claimed.
