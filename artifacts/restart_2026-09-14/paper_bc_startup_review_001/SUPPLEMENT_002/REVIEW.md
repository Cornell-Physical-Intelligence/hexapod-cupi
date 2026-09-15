**Verdict:** Worthwhile as a bounded, non-gating diagnostic. **Hypothesis under test:** fit004's forward failure is partly onset-state mismatch between qualification (BC from raw reset) and demonstrations (onsets after 4s settled neutral). Because memory is feed-forward over a five-sample window, W changes physical/contact state and the previous-held-target input at handoff—not 4s "history." It cannot test quiet or stop: scripted neutral is not BC quiet, and with no moving-to-stop demos, stop failure is expected regardless.

**Measured evidence:** cold BC bobs; no forward/quiet/stop; PPO200 saturated; 120 forward onset rows, all post-settle. Nothing measured yet about settled-onset BC.

**Proposed diagnostic — what results support:**
- C must reproduce the recorded cold fit003/fit004 signature; otherwise the harness is invalid—stop.
- W passes the unchanged forward scorer, and its handoff state (proprio, contacts, held target) lies inside the 120 onset-row distribution → supports onset mismatch as a contributing cause. Not history-only causality: confounds are held-target input, contact settling, solver state; n=1 deterministic.
- W fails similarly → onset state is not the dominant deficiency; fit004 does not walk even under demo-matched onset, pointing at fit/data/command conditioning rather than protocol.
- Partial → read per-control actor mean vs issued action, limiter clipping, saturation, contacts before further runs.

**Minimum constraints:**
- Identical model bytes (hash-equal to fit003), seed, config, camera; reset readback recorded for both arms; one loader for both (original v2 loader, or a verified migration with hash-equal outputs on a fixed observation batch—no identity bypass).
- Per-control `action_source ∈ {scripted_neutral, bc}`; actor-mean==issued enforced on every bc row and skipped only by label on scripted rows—never via a zero-returning wrapper. Log post-limiter target and readback proving zero normalized action holds the existing neutral through the actuator.
- W prefix (200 controls/1600 substeps) preserved, labeled, never scored; both BC portions scored with the unchanged 1000-row scorer and 100-row exclusion; additionally report, unscored, the first 100 BC rows of both arms, since exclusion hides the onset.
- New non-gating case family; 96 cases/13 probes untouched; cold qualification unchanged.
- Separate fresh apps initially to exclude solver-cache order effects; same-app sequential only as a later order-swapped replicate with compared readbacks.

**Adoption:** none; root decision pending.

**Conditional next decision:** If C reproduces and W passes in-distribution → one replicate pair on a second seed before discussing any added settled-onset case; if W fails → close this line and return attention to fit/data.
