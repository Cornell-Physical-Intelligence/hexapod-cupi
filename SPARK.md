# Work on the Spark

You develop in your own checkout and run a frozen copy of your code. Use
[CONTRIBUTING](CONTRIBUTING.md) for dependencies, tests and PRs. Read
[OPERATIONS](docs/OPERATIONS.md) before a GPU run; it owns runtime setup,
permissions and recovery.

## Connect with your account

Connect through Tailscale with your Spark username:

```sh
printf 'Spark username: '
read spark_user
ssh "$spark_user@spark-e26c.tailf4bbf2.ts.net"
```

On Spark, check access and start a persistent terminal:

```sh
id -nG
test -w "/srv/cupi/hexapod/runs/$(id -un)"
docker version --format '{{.Server.Version}}'
tmux new-session -s hexapod
```

You need the `cupi` and `docker` groups and a writable run directory. Ask the
workspace maintainer to fix access if a check fails. Use `tmux attach -t hexapod`
to return to the session; press `Ctrl-b`, then `d` to detach.

## Know where your files belong

| Location | Use |
| --- | --- |
| `$HOME/src/hexapod` | Your clone. |
| `$HOME/src/hexapod-worktrees/` | Your task worktrees. |
| `/srv/cupi/hexapod/runs/<user>/<attempt>/` | Your frozen source and run output. |
| `/srv/cupi/hexapod/inputs/` and `evidence/` | Shared inputs and retained results. |

You edit your own checkout. Preserve shared inputs and the Isaac installation
at `/home/orionh/IsaacLab`. Follow the [workspace access rules](docs/OPERATIONS.md#cupi-workspace)
for container permissions and new input bundles.

## Develop and prepare

From your Spark clone, fetch your branch and create a task worktree:

```sh
cd "$HOME/src/hexapod"
read -r -p 'Your pushed branch (netid/task): ' task_branch
git check-ref-format --branch "$task_branch"
git remote set-branches --add origin "$task_branch"
git fetch origin
task_checkout="$HOME/src/hexapod-worktrees/$task_branch"
mkdir -p "$(dirname "$task_checkout")"
git worktree add --track -b "$task_branch" "$task_checkout" "origin/$task_branch"
cd "$task_checkout"
uv sync --locked
```

For an existing worktree, return to it. Commit your changes and run the checks
in CONTRIBUTING before packaging. Agree on the experiment configuration and
matching standing admission with the run lead. Use
`python3 -m locomotion.runs.prepare --help` to prepare a fresh attempt; keep its
source commit and reviewed binding hash with the run. Preparation starts no GPU job.
Use [OPERATIONS](docs/OPERATIONS.md#before-a-run) for pack and preflight requirements.

## Launch and inspect

In your tmux session, set the attempt path and its reviewed binding hash:

```sh
read -r -p 'Frozen attempt path: ' attempt
read -r -p 'Reviewed binding SHA-256: ' binding_sha
cd "$attempt/source"
python3 -B -m locomotion.runs.launch \
  --bindings "$attempt/binding.json" --bindings-sha256 "$binding_sha" \
  --preflight-only
```

After preflight passes and the run lead confirms dispatch, repeat that command
without `--preflight-only`. Keep the launcher running through container cleanup.
You can edit your worktree while the frozen copy runs.

From another terminal, inspect the log or request a stop:

```sh
tail -f "$attempt/run/logs/standing.log"
# Run this command to request a stop, after leaving tail with Ctrl-C:
touch "$attempt/run/stop.request"
```

Set `attempt` in that terminal to the same recorded path. Read
`run/standing/state.json` for progress and `run/cleanup.json` for cleanup.
Use the [recovery procedure](docs/OPERATIONS.md#ownership-and-cleanup) after a
launcher failure. Preserve checkpoints and failed attempts; record results in
the assigned issue. Use [TRAINING](docs/TRAINING.md#acceptance) to assess locomotion.

The [previous workflow](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/df551b9018505daef56f48e0078282b608d9db45/SPARK.md)
retains the pilot-specific preparation commands and tracking instructions.
