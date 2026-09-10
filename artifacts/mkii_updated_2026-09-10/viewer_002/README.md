# Full-range viewer review

The final viewer loads the canonical nominal motor corrected model. It visits zero → lower → upper → zero for all18 joints, holds every endpoint, highlights the moving rigid group and descendants, and supports pause/resume/reset and1×/2×/4×. At1× it takes199.202 seconds and peak kinematic angular speed is60°/s; animation speed is not a hardware command or validated actuator model. Render time steps are bounded so a background/slow frame cannot skip an endpoint hold even at4×.

`test_motion_tour.mjs` independently checks all36 exact endpoint values, all54 holds, +180° preservation, travel bounds and completion at zero under the largest allowed render interval at each speed. Run it with Node from any working directory; it loads the final corrected model relative to the script. `motion_tour_test.json` is the passing receipt. `qa_receipt.json` binds the actual browser checks to exact final source and model hashes. The UI shows original CAD mass for each part and corrected aggregate link totals.

The user visually approved the movement and grouping. See the sibling main_selection_001 activation receipt. The animation is an inspection aid and provides no physics or combined-pose collision qualification.
