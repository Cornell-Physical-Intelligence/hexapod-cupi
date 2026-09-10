# Matched-origin diagnostic002: origin-storage correction

Origin001 passed fresh32×1000standing but stopped before its first single-case control. Every initial joint/root/target/command/ground check passed except Python object identity between scene and manually-created terrain origins. It produced no rate-comparison trajectory. Its source, raw failure and restoration evidence remain unchanged.

The pinned task creates `self._terrain` manually and never registers it as `scene._terrain`. Installed `scene.env_origins` can therefore return `_default_env_pose[:, :3]`, a new Python view of separate storage. Object identity cannot prove shared tensor storage. Source002 corrects only this reset-origin metadata handling and its protocol/lineage; host, rollout, solver, startup, controller, observer, metrics, commands, five cases, initial-state payload and all physical/quiet gates are unchanged.

Inside cold reset only, both terrain and public scene origin views receive the declaredXY. Original Z values and scene default-pose Z/quaternion are preserved. Each written view must reappear with the same underlying storage pointer, offset, shape, stride, device and dtype, and with exact requested values. The written views remain retained for the independent initial readback. A copying getter fails even at(0,0); a storage swap fails even if values match. Distinct terrain/default-scene storage is allowed only because both actual stores are explicitly synchronized and checked. No body pose is written after the first physics substep; inherited terminal handling remains unchanged.

`PINNED_SETUP_SCENE.json` binds the task setup excerpt; copied `sdk_*_api.json` binds installed getters/writers. Runtime readback now reports whether scene terrain is registered, separate view descriptions, before/after origin values and preserved default pose. The existing exactZ0 and collisionPlane checks still fail closed. This is not a solver, rate-fidelity or physical-success result.

The source is `source_origin_002`:933hashes, exactly three changed paths from origin001 (`tools/matched_origin.py`, `tools/origin_contract.py`, `source_origin.json`). `SOURCE_BUILD.json` names both manifests and all deltas. All930other parent payloads are byte-identical. Source001's prior standing receipt is rejected; source002 requires fresh32standing and all32unchanged quiet checks before the five cold cases.

Nineteen focused CPU tests pass, including four new regressions for distinct stores with public slicing views, copying getters, Z preservation plus exact-plan rejection, and the actual001 alias-only failure. Independent PPO-owner review repeated all19 successfully. Exact-source preflight verifies pinned16runtime/all933files, rejects actual001 admission and rejects failed-quiet synthetic admission. Synthetic receipts only exercise CPU logic.

Root-owned host invocation is unchanged except fresh paths:

```sh
python3 /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_origin_source_002/tools/launch_origin_physics_spark.py \
  --source /home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_origin_source_002 \
  --output /home/orionh/HEXAPOD_runs/mock_length_study_20260909/origin_rate_002
```

Pause038 and the new outer unit/guard are root-owned. Retain3900s maximum,180s stop allowance and75min restoration fallback for the six possible600s phases. Source001 cleanup/restoration must be verified first. No retries, wave/PPO continuation, quiet-miss relabelling, production adoption or changed velocity metric are introduced. The next project-motion work remains the separately proposed opposing-pair load-transfer proof; this experiment is a bounded discriminator.
