# Work on the Spark

You develop in your own checkout, freeze a source copy for each attempt, and
launch that copy through `locomotion.launch`. You can keep editing your
checkout while the attempt runs. Retain the frozen copy with its results.
Stop on a command error before you continue to the next step.

Use [CONTRIBUTING](CONTRIBUTING.md) for code review and CPU checks.
[OPERATIONS](docs/OPERATIONS.md) owns runtime versions and recovery procedures;
[TRAINING](docs/TRAINING.md#frozen-flat-pilot-issues-50-and-51) owns the pilot
budget and acceptance rules.

## Connect with your account

Connect through Tailscale. Choose the command for your account:

| Account | Command |
| --- | --- |
| James | `ssh james@spark-e26c.tailf4bbf2.ts.net` |
| Shaurya | `ssh shaurya@spark-e26c.tailf4bbf2.ts.net` |
| Julian | `ssh julian@spark-e26c.tailf4bbf2.ts.net` |

The laptop alias `spark` selects James on the setup used for the migration.
Use your account name instead of assuming that alias selects you.

Run these checks on Spark:

```sh
id -nG
test -w "/srv/cupi/hexapod/runs/$(id -un)" && echo 'Run directory: writable'
docker version --format '{{.Server.Version}}'
```

Your groups must include `cupi` and `docker`. Your run-directory check must
succeed. Ask the workspace maintainer to fix access if either check fails;
reconnect after a group change. A new contributor needs their own SSH account
and run directory with the same owner and inherited access rules. Do not
borrow another contributor's account or change shared permissions.

Start a persistent terminal before the steps below:

```sh
tmux new-session -s hexapod
```

If you have that session, use `tmux attach-session -t hexapod` instead.
Press `Ctrl-b`, then `d` to detach; reattach after an SSH disconnect. Keep the
launcher in this session so it can supervise its container and record cleanup.
The shell variables below remain in that terminal session.

## Know where your files belong

```text
/home/<user>/src/hexapod/                 Personal clone
/home/<user>/src/hexapod-worktrees/       Personal task checkouts

/srv/cupi/hexapod/
├── inputs/                              Admitted bundles; CUPI reads
├── evidence/                            Retained captures and unchanged manifests
├── runs/<user>/<attempt>/               Your frozen package and outputs
└── maintenance/                         Inventories and migration receipts
```

You can write your own run directory and read other CUPI runs. James owns the
shared input and evidence directories. The maintainer publishes admitted
bundles. Preserve the inherited permissions so CUPI can read your results
and you can write container outputs.

The shared Isaac runtime stays at `/home/orionh/IsaacLab`. The launcher uses
its Docker image; your checkout supplies the frozen project source. Leave the
runtime and its `docker/.env.base` file in place. Keep credentials out of Git
and run receipts.

The administrator completed cleanup. Use
`maintenance/cleanup_20260930_001/relocation.json` under the CUPI root to resolve
old evidence paths. Some protected or unreadable
legacy items remain in place. [OPERATIONS](docs/OPERATIONS.md#prepared-flat-pilot)
links the completed cleanup receipt and the prepared pilot packages.

## Use a worktree for each task

Run this setup in your Spark terminal. Enter a branch name such as
`jcc463/reward-audit`, using your own NetID and task name:

```sh
cd "$HOME/src/hexapod"
git fetch origin
git worktree list
read -r -p 'Branch name (netid/task): ' task_branch
git check-ref-format --branch "$task_branch"
task_checkout="$HOME/src/hexapod-worktrees/$task_branch"
mkdir -p "$(dirname "$task_checkout")"
```

For a new task, create a branch from current `main`:

```sh
git worktree add -b "$task_branch" "$task_checkout" origin/main
cd "$task_checkout"
```

For a branch you pushed from your laptop, use this block in place of the
preceding block. The provisioned shallow clones fetch `main`; add your task
branch to their fetch list:

```sh
git remote set-branches --add origin "$task_branch"
git fetch --depth 1 origin
git worktree add --track -b "$task_branch" "$task_checkout" "origin/$task_branch"
cd "$task_checkout"
```

If the branch exists in a local worktree, return to that worktree. Use a
different branch for a separate experiment; do not force a branch into two
worktrees. [Git's worktree guide](https://git-scm.com/docs/git-worktree)
describes these commands.

You can edit and run CPU checks on your laptop or Spark. For Spark development,
install [uv in your account](https://docs.astral.sh/uv/getting-started/installation/)
if it is absent, then follow [CONTRIBUTING](CONTRIBUTING.md). Run `uv sync
--locked` inside your worktree. This environment serves CPU development;
native simulation uses the shared Isaac image. Do not copy a laptop `.venv`
to Spark or install project dependencies into the shared runtime.

Commit your source and run the required checks before packaging. Confirm that
`git status --porcelain` prints nothing. Preparation copies the files in your
worktree; it does not reject uncommitted edits or record a Git commit for you.

## Prepare a fresh attempt on Spark

The current admitted input declaration is:

```text
/srv/cupi/hexapod/inputs/flat_pilot_20260930_001/admission_128/inputs.json
```

It covers the recorded model and physics with one-robot and 128-robot standing
admission. A physics or model change needs matching native admission before
training. Ask the run lead to review that change; do not reuse old admission
by changing its paths or hashes. Pass `--inputs`: the default configuration
retains historical paths that the cleanup relocated.

The current pilot has four prepared packages in James's run directory;
use [their recorded bindings](docs/OPERATIONS.md#prepared-flat-pilot) for that
dispatch. For a new assigned attempt, the following example prepares a PPO
control from your committed worktree and starts no training:

```sh
umask 027
attempt="/srv/cupi/hexapod/runs/$(id -un)/ppo_mlp_$(date -u +%Y%m%dT%H%M%SZ)"
admitted_inputs=/srv/cupi/hexapod/inputs/flat_pilot_20260930_001/admission_128/inputs.json

python3 -B -m locomotion.prepare \
  --output "$attempt" --remote-root "$attempt" \
  --inputs "$admitted_inputs" --mode train \
  --learner ppo --networks mlp --num-envs 128 \
  --updates 2000 --seed 20260917 --reward-version 2 \
  --allocation-profile flat_pilot_v1 --max-wall-seconds 21600 \
  --logger wandb --wandb-project hexapod-amp --wandb-mode offline

git rev-parse HEAD > "$attempt/SOURCE_COMMIT"
cp docs/TRAINING.md "$attempt/TRAINING.md"
sha256sum "$attempt/TRAINING.md" > "$attempt/TRAINING.sha256"
binding_sha=$(sha256sum "$attempt/binding.json" | cut -d ' ' -f 1)
printf '%s\n' "$binding_sha" > "$attempt/BINDING_SHA256"
printf 'Attempt: %s\nBinding SHA256: %s\n' "$attempt" "$binding_sha"
```

The host Python can prepare and verify a package without importing Isaac.
Using the same path for `--output` and `--remote-root` removes the transfer
step. Preparation refuses an existing destination. Keep each retry under a
fresh name, including retries after a timeout or learner fix.

For AMP, choose a fresh `amp_mlp_...` or `amp_paper_...` attempt name and replace
the learner/network options with `--learner amp --networks mlp` or
`--learner amp --networks paper`. Keep the other learner budget options. The
tripod reference uses its separate prescribed-controller options in
[TRAINING](docs/TRAINING.md#contact-and-motor-comparison).

Record the attempt path and binding hash in its issue with the source commit.
Keep `source/`, `PACK.json` and `binding.json` unchanged after review. Put
additional receipts beside them, outside `source/`. If you reconnect in a new
shell, restore `attempt` and `binding_sha` from the reviewed record; do not
compute a new hash to accept an edited package.

## Coordinate and launch

Agree on the assigned arm and dispatch order with the Spark run lead before
using the GPU. Contributors can edit separate worktrees at the same time.
One owner dispatches each native allocation and checks its cleanup.

GeoData owns the installed Slurm setup. CUPI uses the guarded Isaac/Docker
launcher and its two GPU locks. Those locks coordinate CUPI launchers; they
provide no host-wide queue or GPU exclusivity. Leave GeoData jobs and Qwen to
their owners. Follow [compute coordination](docs/SPARK_COMPUTE_COORDINATION.md)
for reservation changes and preserve the recorded recovery state.

From your frozen source directory, check the package and live host:

```sh
cd "$attempt/source"
python3 -B -m locomotion.launch \
  --bindings "$attempt/binding.json" --bindings-sha256 "$binding_sha" \
  --preflight-only
```

Host preflight checks file identities and available memory, and reads current
GPU workloads. It creates no run output. It does not run the learner's CPU
entry checks or grant standing admission; the prepared pilot receipts record
those separate checks. Stop on a failed check or a busy CUPI lock. Do not
remove lock files or bypass the launcher.

After the run lead confirms dispatch, start the allocation in the same tmux
session:

```sh
python3 -B -m locomotion.launch \
  --bindings "$attempt/binding.json" --bindings-sha256 "$binding_sha"
```

The launcher verifies its frozen source and holds both locks through container
cleanup. Keep `-B` so Python imports cannot add files to the frozen source.
Keep editing in your worktree while the run uses its source copy.

## Inspect results and recover your attempt

You find these files under each attempt; training paths retain the `standing` name:

| Path within the attempt | Contents |
| --- | --- |
| `run/jobs/standing.json` | Launcher status and exact container identity. |
| `run/logs/standing.log` | Native console output and startup errors. |
| `run/standing/state.json` | Native status and completed training updates. |
| `run/standing/metrics.jsonl` | Per-update training measurements. |
| `run/standing/checkpoint_update*.pt` and matching `.json` | Checkpoint bytes and identity records. |
| `run/standing/force_metrics.json` | Recorded contact-force and motor-torque summaries. |
| `run/standing/wandb/` | Offline W&B records for learner arms. |
| `run/cleanup.json` | Final owned-container cleanup receipt. |

In a second SSH terminal, set `attempt` to the recorded path and use
`tail -f "$attempt/run/logs/standing.log"` to follow its output.

To request a stop, run `touch "$attempt/run/stop.request"` for your active
attempt. Keep the launcher running until it records cleanup. After a wrapper
failure, return to its frozen `source/` and use the same reviewed binding:

```sh
python3 -B -m locomotion.launch \
  --bindings "$attempt/binding.json" --bindings-sha256 "$binding_sha" \
  --cleanup-only
```

Inspect `run/cleanup.json` and the recorded container identity before another
allocation. If cleanup fails, retain the logs and involve the run lead. Avoid
global Docker pruning, broad process kills and changes to shared services.
The runner has no training resume; preserve an interrupted attempt and prepare
a fresh one if the lead approves a retry.

Follow the frozen comparison order in [TRAINING](docs/TRAINING.md#frozen-flat-pilot-issues-50-and-51).
Record contact force and motor torque before the next learner sequence. Keep
checkpoint and capture bytes, including failed attempts. A completed process
does not qualify walking or pass the pilot decision rule.

Use [TRAINING's tracking instructions](docs/TRAINING.md#run-tracking) to sync
offline W&B records. Put the attempt path and result hashes in the assigned
issue; publish selected evidence through [PROJECT_SITE](docs/PROJECT_SITE.md).
Keep full run payloads outside Git. Remove a finished personal worktree with
`git worktree remove` after preserving its commits; keep the run archive.
