# Initial inspection preparation integration

Integration base: `6908b48911c150fa4cf34fc8a7ce0141a8615614`.
This receipt covers the initial conservative inspection envelope, before the user supplied wider joint travel in the viewer review. It preserves that checked step rather than relabeling its 133-pose result as evidence for different limits.

Root reviewed the recovery, fusion and USD authoring code, reread all source-bound manifests, verified 59 original STL hashes and checked that both USD source snapshots exactly match the delivered URDF/model bytes. Root independently ran 5 mass, 3 builder and 5 USD regression tests; both saved USD audits pass. The actual browser control review is in `../viewer_001/`. Source geometry and the original screw penetrations are fully reported in `../import_001/`.

`tools/project_site.py check --base6908…` and `build` pass (the actual full command is recorded in `site_check.log`). `tools/check_pipeline_lineages.py current` verifies the existing 426-file runtime manifest unchanged. No policy, runtime, task, gate, checkpoint or Spark process was modified.

The public deployment will be verified after final integration. The new user-supplied travel ranges and calculated coxa plate-clearance endpoints belong to a separate review successor; these initial bounds and tests remain historical evidence.
