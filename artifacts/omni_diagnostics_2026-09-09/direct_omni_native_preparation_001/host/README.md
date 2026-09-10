# Direct-PPO bounded training host

This is a new successor to the frozen cold host. It binds the complete 598-file native source `37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6` and the 21-payload native002 contract `9e591b93ae99ca3a76b7f7500ea20cc778da76c7f6b9b2c86fd22d31aceab3d7`. The source uses the preserved 16-file direct-PPO environment and the explicitly versioned native curriculum/CAPS overlay; source009 supplies only the host supervisor, not robot physics or a reference controller.

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
