The rejected left-strafe trace supports a specific lost-preload hypothesis. RR force declines from7.75 N at confirmed landing to0.946 N while LM swings, despite a large geometric COM support margin. The fixed virtual toe target loses about0.492 mm of downward offset as the pad changes orientation. Contact remains; the unchanged1 N/five-support gate still rejects the run.

![Measured load and target geometry](owner/diagnostic.png)

[The frozen diagnosis](owner/README.md) separates actual50 Hz force evidence,400 Hz pose/torque evidence, reconstructed mesh geometry and a CPU-only target sensitivity. The proposed single trial lowers RR's anchor0.5 mm with a C2 quintic during the existing0.3 s first-landing hold. Prefix IK/target bounds pass; the small remaining torque margin means physics must still test feasibility. It is not an adopted gait or a repair for every direction.

`owner/` preserves all14 original frozen payloads. [Root review](ROOT_REVIEW.json) verifies every one of the930 source files and reproduces all five numeric result files exactly; the figure was visually inspected. No GPU trial occurred in this review.

For numeric reproduction, restore the unchanged contents of `owner/` to the fresh repository path `tmp/reference_strafe_support_review_001/`, restore directional002 source using [its source reconstruction](../reference_directional_002/preparation/RECONSTRUCTION.json) at `tmp/reference_directional_adapter_002/source_directional_002/`, and restore [the original raw bundle](../reference_directional_002/raw/) plus [remote audit](../reference_directional_002/remote_audit.json) at `tmp/reference_directional_results_002/`. Verify the manifest identities in `owner/INPUT_SHA256.json`, then use the frozen README commands with fresh output paths. The preserved scripts intentionally retain their original repository-relative working paths; they do not run directly inside `owner/`.

`python3 -B verify_payload.py` provides portable read-only publication verification without restoring those working inputs.
