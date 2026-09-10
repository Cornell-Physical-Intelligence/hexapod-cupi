# Breadth-first legged policy review

Read the [integrated recommendation](REPORT.md), then the detailed reviews of [training recipes](reviews/training_recipes/REVIEW.md), [hexapod gait and actuation](reviews/hexapod_actuation/REVIEW.md), and [terrain, sensing and navigation](reviews/terrain_perception/REVIEW.md).

The [source index](SOURCES.json) contains 32 distinct core works, a separate tentative Swift comparison, and explicit access limitations. The specialist reviews are copied verbatim with their original manifests; workstation paths inside those frozen records document their original evidence access. The integrated report provides portable repository links.

The recommendation is a matched cold evaluation of the preserved omnidirectional PPO, followed by command-balanced standing/stop training and a separate smoothness ablation. A learnable phase/placement controller remains the independent architecture alternative. Terrain control, privileged teaching and sensor replay proceed in parallel. Literature is not a performance result and changes no acceptance gate.

Run `python3 verify_bundle.py` from this directory to verify all published payloads. Full copyrighted papers and download caches are excluded.
