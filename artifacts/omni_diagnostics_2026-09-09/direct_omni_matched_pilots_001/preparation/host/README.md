# Direct-PPO bounded training host002

This is a pin-only successor to frozen host001 (`332fad29e0a0f88ba07012700f2a21c3a4f0453f3d71224da84fcc477e48394f`). Its launcher changes only `SOURCE_MAP` and `CONTRACT_FREEZE`. The external entry wrapper and legacy16 inventory remain byte-identical. Every function, phase, mount, original-checkpoint selection, per-phase deadline, cleanup path and result check is unchanged.

The complete 598-file source is `BASE/direct_omni_train_source_002`, map `64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e`. The contract is `BASE/direct_omni_train_preparation_002`, native00326 freeze `20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb`. Plan `9fb7d0b01480523972b4f1c17e717756d7dcf1c54b9caa436245671937de538c` is unchanged. `SOURCE_BUILD.json` preserves the root's actual new-source build receipt; no simulator allocation is part of this host preparation.

Native003 addresses the prior smoke's finalizer exception after two actual PPO updates: live observation-normalizer buffers created inside inference mode must receive value-preserving ordinary-tensor clones before strict checkpoint reload. The PPO owner tests and binds that separate consumer change. This host does not modify the learner, normalize weights, relax reload checks, recover a failed campaign or continue old trained weights. The original smoke failure and host001 receipts remain immutable historical evidence; host002 CPU tests are a separate run.

The same immutable host and source support three independent allocations:

| Allocation | Branch | Training | Ordered phases |
|---|---|---|---|
| smoke | caps | 32 environments × 24 controls × 2 updates | fresh standing, train, final constant commands, final moving-to-stop |
| pilot | curriculum | 1024 × 24 × 50 | fresh standing, original constant commands, original moving-to-stop, train, final constant commands, final moving-to-stop |
| pilot | caps | 1024 × 24 × 50 | same complete six-phase comparison |

Each pilot requires a separately completed matching CAPS smoke. It starts from original checkpoint `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`; one pilot never resumes the other. Every output directory must be fresh. No next allocation launches automatically. Training and diagnostic completion are unqualified; requested saturation, nonfoot contacts and termination/quiet failures remain evidence for a separate decision.

## Contracts and protection

`verify_inputs(args)` is standard-library only and checks this host, the exact complete native source, native contract, full926-file supervisor and canonical legacy16 tree. It then calls the native contract for plan, original checkpoint, selector and prior-smoke validation. `validate_inputs` is an alias for outer guards.

The original source009 `run_owned` and `owned_container` functions are imported unchanged, including their locks, competitor/memory checks, 90-second actual AppReady deadline, 600-second per-phase limit, contact-buffer audit and exact-name/recorded-ID cleanup. Namespace adapters change only the container command, full native source verification, legacy runtime identity and truthful job metadata. Their original supervisor error text sometimes says “standing” for other phases; the explicit phase/allocation/branch fields remain authoritative.

The source, original checkpoint, external entry adapter and optional prior smoke mount read-only. Completed phase trees are hashed and mounted read-only at their `/output/<phase>` aliases after the containing `/output:rw` mount. Standing also mounts at `/admission:ro`. Evaluations get their exact selected checkpoint at `/checkpoint/evaluated.pt:ro`; final evaluations additionally receive the completed training folder read-only at `/output/train`. Full input and completed-tree hashes are checked before and after every phase and during finalization.

The external `run_train_entry.py` keeps the cold wrapper's two metadata-only insertions: emit readiness only after real AppLauncher construction, and wrap the original state writer. Every original native entry function/class AST remains unchanged. All imported `hexapod_rl` modules must resolve under the source's verified legacy package. Canonical legacy tree hash remains `abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280`.

An unverified policy phase records `no_policy_loaded: null`. A completed training phase binds original warm-start loading through its actual repair-initialization receipt; its state checkpoint identifies newly trained weights. Evaluation phases bind their selected checkpoint separately. Malformed optional policy metadata is recorded without preventing the inherited cleanup receipt from being written.

Campaigns record accepted update counts separately from producer-reported updates. Missing producer counts remain unavailable, not zero. A later failed evaluation does not erase completed training. Per-update checkpoints saved by the native learner remain unqualified diagnostics; host completion does not select or promote them.

## Invocation

```sh
python3 launch_train_spark.py --source /path/to/native-source --checkpoint /path/to/original.pt --contract /path/to/native-contract --supervisor-source /path/to/source009 --output /path/to/fresh-smoke --allocation smoke --branch caps --preflight-only
python3 launch_train_spark.py --source /path/to/native-source --checkpoint /path/to/original.pt --contract /path/to/native-contract --supervisor-source /path/to/source009 --output /path/to/fresh-pilot --allocation pilot --branch curriculum --smoke /path/to/completed-smoke --preflight-only
```

Only the root agent owns actual allocation and the external pause/restore guard. This host never signals unrelated work. The `--preflight-only` path creates no output directory or simulator process.

Focused tests exercise both phase lists, original/final checkpoint mounts, read-only nested aliases, exact unchanged supervisor functions and the actual precontainer source-verification seam, failure-preserving campaign finalization, wrong selectors, source metadata insertions, and standard-library host imports. They provide CPU orchestration evidence, not a physical smoke result.

## Successor verification

`test_pin_delta.py` verifies the original eight-payload host freeze, exact complete launcher AST equality after normalizing only the two version pins, and byte equality of the entry wrapper and legacy map. The original 18 orchestration tests run against native003, for 21 focused tests total. They do not rerun native003's real-RSL regression and do not imply an actual simulator, smoke or pilot pass.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tmp/direct_omni_train_host_002 -p 'test_*.py'
```

Only root may choose and dispatch the new smoke with its separately bound pause/restore guard. The original preview adapter remains pinned to native002 and cannot accept this new source until a separately reviewed successor is prepared.
