# CUDA metrics comparison preparation

The [arithmetic proposal](../runtime_efficiency_review_v1/README.md) passed
sampled CPU equivalence. This bounded runner will check the actual Spark CUDA
backend in the same container image and Python environment used by the current
physical validator. It does not start Isaac Sim, qualify physics, or train a
policy. The frozen training source remains read-only.

`run_probe.py` defaults to a CPU-only dry run and requires `--execute` for the
experiment. It verifies the exact probe hash and functional source identity,
acquires both normal per-job GPU locks, and uses the frozen supervisor's
resource, coordination, and immutable-container ownership checks. The CUDA
process stays behind the existing CPU admission barrier until those checks
pass. The job has a 900-second timeout, stops/removes only its own container,
and releases its locks at exit. It creates no reservation or scheduled task.

Local and Spark dry runs passed without starting CUDA. The tools are staged at
`/home/orionh/HEXAPOD_runs/mkii_cuda_metrics_v1/tools/`; the planned fresh result
directory is `/home/orionh/HEXAPOD_runs/mkii_cuda_metrics_v1/cuda_001/`.
Execution must follow completion and cleanup of the active refined validation.

The benchmark reports synthetic arithmetic timings, not simulator throughput.
Any adoption requires review of CUDA output/threshold equivalence, complete
capture-row checks, a new source release, and fresh physical qualification.
No runtime optimization has been installed by this preparation.

## Execution evidence

Attempt001 failed before creating a container or output directory: the shared
`/opt/wx/gpu.lock` is readable but cannot be opened read/write by this account.
The original supervisor, launch and host log are preserved. `run_probe_v2.py`
opens that existing file read-only and takes the same exclusive Linux flock;
it does not change the file's contents, permissions or ownership. This matches
the existing command-line flock's ability to use the shared file.

Attempt002 launched at23:53:36UTC and completed with exit0, followed by verified
owned-container removal. It used actual **PyTorch2.10.0+cu130 / CUDA13.0 / GB10**.
The original result is [cuda_002/report.json](cuda_002/report.json).

| Arithmetic subset | Existing median | Batched median |
| --- | ---: | ---: |
| 32 environments, float32 | 1.420 ms | 0.135 ms |
| 512 environments, float32 | 1.495 ms | 0.133 ms |

The32-environment float32 comparison was bitwise equal. The512-environment
case had point/pin-velocity differences up to4.768e-7 in their respective
units; point error differed by up to2ULP and pin velocity by64ULP. Axis,
passive-velocity and force-norm outputs matched exactly. Float64 differences
were at most4.441e-16. The probe's `pass=true` means its sampled tolerance and
classification tests passed; it does **not** mean all CUDA outputs were
bitwise equal. Highest matmul precision does not ensure identical accumulation
for differently batched contractions.

Consequently the optimization remains uninstalled pending complete-row and
actual-CAD threshold checks. Retaining the legacy closure arithmetic is an
available conservative alternative. This subset speedup removes roughly20ms
per16-substep control at these measured timings; it cannot explain or eliminate
the entire approximately800ms observed validation-control duration. Native
sensor/physics profiling is a separate experiment.

[retrieval.json](retrieval.json) records the six remote hashes matched after
retrieval, including both host logs. No PPO checkpoint or simulated robot
trajectory was produced by this arithmetic experiment.
