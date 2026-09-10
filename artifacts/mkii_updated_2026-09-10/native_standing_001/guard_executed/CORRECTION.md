# Exact NVIDIA query flag correction

The actual first standing dispatch failed in `nvidia-smi` before any pause, output or unit creation, as reported by root. Its command used the invalid `--standing-compute-apps=pid,process_name`, introduced by an overbroad phase-name replacement. The original63-payload guard and actual traceback are preserved. Earlier mocked tests did not inspect this argv literal.

The sole runtime change restores `--query-compute-apps=pid,process_name`. The new test inspects the actual runtime AST argv, compares it with the correct query guard parent and proves the failed001 literal differs. Existing tests separately compare the protected CUDA ancestry loop and embedded restorer byte-for-byte with that parent. No ownership, pause, locks, source, asset, host, controller, timeout or restoration behavior is otherwise changed.

Root will independently confirm that the original output/unit/pause were never created before reusing approved names. Only the deployed guard path changes to `standing_guard_002`. The source40 f81f61... and host18 302358... remain identical. Preparation performed no remote or GPU action.

Logs inherited from001 are historical original tests; `tests_correction.log` is the actual successor22-test rerun. The new full local bound-input receipt is `bound_inputs_correction.json`; neither is a native standing result. Root owns central evidence publication under `docs/PROJECT_SITE.md`.
