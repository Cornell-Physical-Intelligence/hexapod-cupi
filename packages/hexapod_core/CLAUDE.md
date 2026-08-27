# hexapod_core

This package owns the frozen interface contracts every other package builds on:
the 66-dimensional observation layout in concatenation order (`observation.py`),
the 18-dimensional action interface with its clip bounds, action scale, slew
budget, stand scale and 50 Hz/200 Hz/decimation-4 timing (`action.py`), the
three-scalar velocity command that is the planner/policy boundary plus the
Stage2C and Phase-2 envelopes (`command.py`), the runtime joint/action order
(`joints.py`), the RobStride RS05 actuator numbers and termination thresholds
(`actuator.py`), and the anatomical coordinate contract (`frames.py`). Every
value here was read out of `packages/hexapod_env/hexapod_env/` and is bound to
its source literal by `isaaclab/tests/test_core_contracts.py`; documentation is
corroboration, never the source, with the single documented exception of the
runtime joint order, which the training Python never declares. Don'ts: no
dependencies at all — stdlib only, not even torch or numpy, so this package
imports on a bare system interpreter as readily as inside the Isaac container;
never import any other hexapod package (everything depends on this, it depends
on nothing); never edit a v1 value or reorder a v1 field — a changed interface
is a v2 constant or module standing beside v1, because a deployed checkpoint's
meaning is frozen against these numbers; and never add behavior here beyond
plain validation and unit conversion, since pipelines belong in
`hexapod_runtime` and producers in `hexapod_nav`.
