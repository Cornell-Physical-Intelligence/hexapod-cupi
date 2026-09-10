# Independent contact and stop review of actual009

The complete 48 s trace contains 11 measured flight/landing events across all six legs. Independently reconstructed events require at least two consecutive absent-contact samples, at least 2 mm measured rise, and three consecutive returned-contact samples. They agree with the controller's 11 confirmed landings. Short unloading episodes remain separate in the report.

All 32 retrieved raw payload hashes match. The trace has zero residual actions, zero reference execution lag, at least five post-settle distal supports, no terminations or nonfoot contacts, and peak post-settle requested torque 1.2542 N·m. The unchanged source reports 115.24 mm actual forward displacement over its 24 s movement window; the original displacement/integrated-velocity difference is 4.373 mm, below its existing 5 mm gate.

Stopping is finite but slow: the command becomes zero at 28.00 s, and the reference reaches quiet hold at 33.54 s, a **5.54 s latency**. After another 2 s excluded settling, 12.46 s pass the existing quiet scorer. This is not a prompt-stop qualification and cannot be covered by extrapolating a 250 ms terrain observation lease.

This is a bounded zero-residual stepping-reference result. It is not PPO, a useful all-direction Stage2 policy, terrain traversal, or hardware qualification. A progress recording must run fresh physics with the exact source/admission and record its own outcome; it is not a movie of this already completed run.

[report.json](report.json) contains the raw-event reconstruction and source metric values. [review.py](review.py) reproduces the review without editing its inputs.
