# Independent short device-host review

The corrected host002 has no remaining concrete blocker in this focused CPU review. All 11 tests independently pass. This is a review of guarded dispatch/result checks; it does not report a CUDA run, fresh quiet admission, walking or PPO performance.

Both immutable inventories are verified: host001 freeze `139f9680aa5a92f5694fbd0350d22945e36da3d715e78ee113b220985d519a80` and successor002 freeze `adbb60b3d77b8d6b268dcf3060232fcf3c493361259a0997582e694d62a35658`, eight files each. Exact runtime file hashes and the source009 supervisor hash are in [verification.json](verification.json).

## Finding and correction

Host001 compared actual/current, last-update and exported expected sensor-clock arrays, but it did not independently verify advancement between successive rows. Joint duplication or reordering of all three arrays could evade that extra host evidence check. The already frozen device bridge itself checks its clock recurrence at runtime; this was an independent raw-evidence validation gap, not an observed CUDA defect.

Host002 closes the gap by reconstructing each next actual row from the previous actual float32 clock, using eight additions of the float32-rounded 2.5 ms increment. It rejects duplicate/reordered rows even when the three exported arrays agree. The first clock epoch remains unspecified; subsequent advancement, nonnegative finite values and source-bound runtime checks remain mandatory. I independently ran all 11 tests, including the joint-corruption regression; see [successor_tests.log](successor_tests.log). Parent host001's ten tests also passed independently and are retained in [tests.log](tests.log).

The only successor launcher changes are importing and calling that independent recurrence helper. [verify_review.py](verify_review.py) checks this exact tiny delta. No physics, source009 runtime, observation schema, controller, GPU supervisor or gate was altered.

## Other reviewed guards

The two-phase loop checks source/input identities before allocation, then waits for the exact owned job and validates all declared and numeric results before it records acceptance. A rejected N1 result or changed inputs prevents an N32 allocation. Tests exercise both cases. The host imports the exact source009 supervisor and overrides only its container command callback; the original locks, resource checks, AppReady/deadline handling, contact-log audit and exact owned cleanup remain in that pinned source. The callback is restored on exit.

Source009, the completed physical run/assets, the 18-file adapter and 160-file observation bundle are read-only mounts; output is fresh and outside those inputs. Host self-inventory is checked against its starting freeze before, between and after phases. Each result requires the exact identity, 264 controls, 846/849 outputs, original 2,113-sample observer completion, finite/ordered times and requested/applied postsettle torque values under 1.6 N·m. The host reads complete arrays, including interior substeps; it checks all sensor-clock/contact-validity rows and preserved final raw SDK joint velocity separately from the interval-angle channel. No actor/checkpoint or automatic walking successor exists.

The underlying bridge still performs the source-bound runtime checks of every physical/control row; this compact host does not reconstruct every observation/reference operation from raw arrays. Final state/receipt checks complement those pinned executable checks. Neither reviewed phase is a replacement for the full standing or ten-second quiet gate, and measured GPU timing is still pending at the time of this receipt.
