# Qualified-flight source review

Eight targeted independent CPU tests passed after the owner corrected the fixture baseline. The source distinguishes unloading/rebounds from measured flight without changing the target path: two contiguous off-contact samples and measured 2 mm lift are required before flight latches. An unqualified return clears count/height, and the existing planned apex is a finite qualification deadline. After qualification, early obstacle return still fails the original apex/descent/landing checks; stop completes the current swing without new liftoffs. Support, torque, 12 mm landing and physical clearance thresholds remain unchanged.

Actual008 prefix targets remain identical, but its final rebound is unqualified unloading, not an accepted touchdown. Old003 insufficient lift still rejects (earlier at apex); old004 12 mm failure still rejects; old002 remains provisional landing. Counterfactual replay does not establish physical continuation.

The additional unloading mode, raw off-contact runs/samples and optional unqualified-return state require a new observation/schema binding. The frozen wave004 tensor state is not compatible.

Independent command: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tmp/omni_reference_wave_005 -p test_liftoff_qualification.py` — 8 tests, 2.907 s, PASS.
