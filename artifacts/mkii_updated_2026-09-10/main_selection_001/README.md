# Prepared main model selection

The user requested the detailed robot as the main URDF after inspecting an animation of every joint. This is the exact prepared switch; it is not an activation receipt.

`activate_main_model.patch` adds `robot/active_model.json` with hash-pinned nominal motor corrected URDF/model references and points the repository Vite quickstart at the detailed part/joint inspector. The inspector displays original per-part CAD masses, while the dynamics selection uses the explicitly labeled nominal motor correction. There is no synthetic mock gait in that new default viewer.

Apply only after the user's requested visual sign-off, then update STATUS, the contributor guide, README, the plan and the poster record with the actual selection. Verify both manifest file hashes against the selected files, build the Vite source, and check the rendered default. Source meshes, old task IDs, checkpoints and historical assets are preserved. No existing policy becomes compatible by changing this pointer. Native SDF cooking, the new coordinate/actuator adapter and a separate exact-model admission remain required before training on it.

The switch does not run or stop a Spark job. Current execution coordination remains in STATUS and the hardened launchers.

## Applied selection

The user subsequently approved the animation and explicitly instructed: “save this URDF as the ground truth URDF for all training with the motor weight overrides.” `activation_receipt.json` records the final canonical selector identity and exact URDF/model/USD hashes. The applied selector adds that stronger all-training scope and the actual visual approval to the earlier prepared patch. The generic viewer now opens the detailed inspector. Native physics admission remains false; no training was launched or existing GPU job interrupted by this asset-selection action. Historical task IDs were not repurposed.
