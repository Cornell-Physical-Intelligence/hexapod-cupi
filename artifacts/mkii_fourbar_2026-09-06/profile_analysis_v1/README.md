# Profiler attempt 001: output-publication failure

No bottleneck estimate can be established from this attempt. The actual
32-environment, 800 Hz standing workload completed all 100 controls and 1,600
physics captures, and its short validation report passed. However, neither
`profile.json` nor `trace.json` was published. The owned container exited with
code zero; the supervisor correctly marked the attempt failed when it could
not find the required profile and removed the exact owned container.

[failure_metadata.json](failure_metadata.json) records remotely calculated
SHA-256 values and sizes for all surviving original output files, the exact
source/tool identities, final supervisor/validation status, and installed SDK
launcher path/hash. Root retrieves the original evidence separately; this
analysis does not replace or rewrite it.

The inspected `launch_simulation` implementation calls the application's close
function from its `finally` block at lines 505–507. Its source SHA-256 is
`2d0d5c11e594273576639e0e5895b9360a1d762ce83565c95ff70dcca679de94`.
The validator writes its final report before leaving that context. Version 1
exported the profiler only after `validator.main()` returned. The observed log
ends at simulation shutdown without the wrapper's final-result marker, and
there is no Python traceback explaining a normal wrapper return. This is
consistent with Kit terminating the process during app shutdown before the
outer export code can run.

The version 2 diagnostic therefore observes the validator's exact final report
write: it calls the original writer first, then exports the stopped profiler
and joins the actual validation result while still inside the live app context.
It preserves every original report byte, physical capture, SDK call, and
shutdown action. A subprocess test deliberately calls `os._exit(0)` immediately
after the final validation write and verifies durable profiler output before
exit. Version 1 tools and the failed attempt remain unchanged.

There is no recoverable host-region or CUDA trace breakdown here. Container
wall time includes startup, simulation, profiling, and shutdown and must not be
reported as a measured control-step throughput or used to choose a sensor/solver
optimization. A subsequent version 2 profile is required for that decision.
