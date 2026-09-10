# Bounded independent directional frame review

No concrete frame or direction-projection blocker was found in the frozen directional adapter `1ffc294c64d3e3eb47a13710115ae94c9033da398e015854c9d6645138387032`. This is a late read-only code review; the owner remains unchanged and root already owns the physical run.

Navigation forward/left maps to body through `[left, -forward, 0]`; the implementation uses this vector and rotates it by measured body-to-world orientation. Each world displacement is projected along the adjacent measured heading directions. Heading is independently extracted from the body forward axis `-Y`. Twelve synthetic covariance cases (four commands × three global yaw offsets, also translated in world) preserve displacement, direction, yaw and motion verdicts to numerical tolerance.

Pure turning requires signed measured heading change and reported yaw rate, with at most 10 mm planar drift; it does not require translation. Arc cases require both signed translation and yaw. Requested and admitted commands must match the exact 4–28 second schedule. Filtered reference and measured executed twist remain separate telemetry, with no silent command derating. The runtime capture retains pre-reset samples, uses the unchanged wave005 targets, and stops on existing torque, contact or support failures. The final score retains six measured leg events, original 5 mm position/rate consistency, and at least 10 seconds of original quiet criteria after reference quiet plus the excluded settling interval.

Reported body twist uses the legacy COM channel, while world position/rate evidence uses matching root-link quantities. Body gyro Z is not a general exact Euler heading derivative under roll/pitch; the report keeps both signed heading and reported yaw checks plus their discrepancy. No native velocity fidelity, new physical result, omnidirectional coverage, or walking PPO claim is made here. Existing owner tests were not broadly repeated.

Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tmp/reference_directional_frame_independent_review_001/review.py` from the repository root for the read-only hash and synthetic covariance check.
