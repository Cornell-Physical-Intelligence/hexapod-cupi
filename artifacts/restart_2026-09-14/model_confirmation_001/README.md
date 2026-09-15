# Confirmed model and authorized qualification restart

On 14 September 2026 the user visually confirmed the selected mass-corrected robot and requested restarting standing, walking/stopping, terrain and survey qualification on that exact model. MODEL_CONFIRMATION.json binds the unchanged URDF and viewer model hashes. The nominal motor mass is 191 g; its spatial and inertia allocation remains an estimate. Visual confirmation does not admit physics, training or hardware.

The single-robot standing pass and 32-robot standing rejection remain historical evidence. Walking training on this model has not started. Fresh qualification must use matching source/configuration and the unchanged gates.

The local restart branch and CPU checks do not establish a native run. SPARK_READONLY_PREFLIGHT.json records a read-only host inspection: dsv41-inference.service owns the DeepSeek inference processes, uses about 107 GB of GPU memory and holds the shared workload lock. No competing process was stopped and no native allocation was launched.

The next diagnostic proposed by the existing saved analysis compares a fresh single robot at the origin with one at (14, 4), preserving source005 dynamics and all standing gates. That is a proposed investigation, not a diagnosed fix or an admitted training run. Its placement must be explicit before warmup and at reset, with a successor source identity and review. The current workflow remains blocked on live compute allocation and subsequent standing admission.
