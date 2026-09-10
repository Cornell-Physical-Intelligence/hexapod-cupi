# Qualified-flight reference and observation pipeline

This CPU prototype combines the complete batched wave005 reference with a new 846-value actor and 849-value critic interface. It preserves explicit unloading, qualified flight, landing, stop, bounded residual and reset-safe history state. It is the next integration candidate; no policy is loaded and no CUDA, walking or Stage 2 admission follows from these tests.

The [independent review](independent_review/README.md) verifies all 86 tensor files and all 160 observation files, the exact copied reference source, and the complete actual009 replay. All 13 tensor tests and 18 observation tests pass independently; root also passed the 13 tensor tests. The batched reference reproduces all 2,200 post-settle controls from the complete slow physics screen, including eleven landings, the unqualified RR rebound and final quiet hold. The maximum emitted position difference is 2.7e-15 rad. Physical feasibility belongs to that separately audited scalar screen.

The new [schema and interface](owner/README.md) expose controller state and five valid-masked history frames without pretending earlier checkpoints are compatible. Raw SDK joint rates remain separate from an additional 20 ms interval-average angle-derived rate and its validity. Unknown contact points retain raw NaNs and masks. Measured sensor freshness and original pre-reset terminal flags are still required from a separately reviewed real Isaac bridge. No favorable rate substitution, gate change or hidden simulator-truth deployment claim is made.

The original 160-payload owner freeze is preserved under `owner/`; its embedded reference tree is the exact 86-payload tensor005 package. The seven-payload independent receipt remains separate. Keeping publication prose outside those frozen trees preserves their strict manifest inventories. The next bounded device test checks actual sensor timestamps, packing, sequencing, shapes and execution cost on 1 and 32 robots before any PPO allocation. A short hold test cannot qualify sustained standing or walking.

Local CPU test invocation:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run python -B -m unittest discover -s owner -p 'test_*.py'
```

Run from this publication directory. The tests require the workspace dependencies but do not start Isaac or use a GPU.
