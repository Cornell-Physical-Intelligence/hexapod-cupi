# Persistent HEXAPOD Spark reservation

The user instructed: “make sure our job takes full control of the spark at all times”. The reservation applies between jobs and supersedes earlier weather-only and sharing-on-request policies. This bundle records actual operations, not a proposed installation.

Five forecast timers and their five controller services have persistent systemd condition drop-ins requiring the reservation marker to be absent before they can start. The timers were active before deferral; all controller services were already inactive or failed, so no running forecast controller was stopped. The idle NVIDIA PAIR Ollama service was subsequently stopped through its exact verified invocation, PID start time and executable. Its source unit and outputs were preserved. The same reservation condition prevents its user service from restarting.

Read-only verification confirms all 11 units inactive or failed, all owned drop-ins loaded, no pending manager reload and no CUDA compute process at the recorded instant. The marker, original scheduler states, exact drop-in bytes and shared coordination SHA are retained. SSH and operating-system services remain available. This is persistent admission control for these known systemd producers, not hardware isolation: manual or privileged CUDA launches can still bypass it. Existing job resource checks remain truthful. No process suspension is misrepresented as freeing CUDA memory.

Per-job cleanup restores only that job's snapshot and must not release this reservation. Release requires an explicit later user instruction: verify these exact owned files and current unit identities, remove only the owned reservation controls, reload the manager and restore only the recorded prior active/enabled states. Do not replay expired transient jobs or remove another contributor's changes.

`verification_loaded_001.json` contains the final readback; the scripts and before/after receipts preserve the operations. No hardware partition, MPS quota, GPU clock change, policy-training result or Stage 2 completion is claimed. Run `python3 -B -S verify_bundle.py` for an offline integrity check; it performs no remote action.
