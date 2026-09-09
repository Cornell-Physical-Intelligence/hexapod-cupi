# One campaign → one policy video

Prepared 2026-09-06; CPU-tested, not remotely launched. This is a bounded server
process, not a recurring Codex automation. It has no notification integration and
does not create GPU containers itself. Only the existing guarded capture launcher
can start a render, after full training succeeds.
It never creates an exclusive reservation or changes the shared coordination
file. The capture launcher requires the existing canonical control to permit use;
`HEXAPOD_SHARE_STATUS=REQUESTED` prevents capture and remains untouched. Preparing
these files does not authorize or dispatch a campaign, resume, or capture.

Use the corrected frozen training release and **policy_capture_tools_v2**. The
earlier campaign/source identity must not be substituted after a source change.
Arguments explicitly bind the exact `campaign.json`, campaign PID/start ticks,
source path, functional SHA-256 and commit. Capture-tool file hashes and this
follower's hash are also bound in the durable state.

```sh
/usr/bin/python3 /ABSOLUTE/FOLLOWUP/follow_campaign.py \
  --campaign /ABSOLUTE/EXACT_CAMPAIGN/campaign.json \
  --campaign-pid CAMPAIGN_PID \
  --campaign-start-ticks PROC_STAT_START_TICKS \
  --source-dir /ABSOLUTE/CORRECTED_FROZEN_SOURCE \
  --source-sha256 FUNCTIONAL_SOURCE_SHA256 \
  --source-commit EXACT_COMMIT \
  --capture-tools /ABSOLUTE/policy_capture_tools_v2 \
  --state-dir /ABSOLUTE/NEW_FOLLOWUP_STATE \
  --wait-seconds 64800
```

For the eventual authorized launch, detach this one process on the Spark and
redirect stdin/stdout/stderr to server files so the Mac's connection is irrelevant.
The example deliberately contains placeholders: the corrected training campaign
will provide the actual binding. Do not launch it against the earlier incompatible
runner source. The state directory must be outside frozen source.

The default wait is 18 hours, capped at 24 hours. It reads campaign status and
verifies the exact Linux process identity every ten seconds, rechecks source bytes
at least every minute, and preserves the original deadline across restarts. PID
identity includes start ticks, complete argv and Linux boot ID. Campaign failure,
pause, missing process, changed source or deadline expiry records `no_video`.

Capture requires campaign `state=complete`, `pass=true`, no errors, verified
separate-process resume, and exactly one passed **full** phase requesting and
completing **1,000 iterations with 512 environments**. The primary report hash,
successful supervisor/cleanup, own-campaign admission hash and exact final
checkpoint location are checked. Existing capture validation then checks the full
source/admission/training/checkpoint lineage again. The three-update scratch
checkpoint cannot qualify.

The follower invokes the existing capture launcher for 15 seconds/750 frames at
1280×720 with its existing 1,800-second timeout. It writes the exact argv and an
fsynced `launch_reserved` state **before** creating the capture subprocess. The
capture output directory must not exist before dispatch, including as a dangling
symlink; prior successful evidence cannot cover a failed new launch. The follower
allows 2,160 seconds for the capture supervisor, including owned-container cleanup
and independent video decoding after the bounded render interval. The
recorded post-exec process identity includes PID, start ticks, argv and boot ID.
Only this exact capture supervisor may receive a forwarded SIGTERM; Linux pidfds
prevent a reused PID from being signaled. It never signals the campaign or other
jobs. The capture host retains GPU/coordination/ownership guards and cleanup.

Restart the identical command with the identical state directory. Waiting resumes
with its original deadline; an already recorded capture is monitored without a
second launch. If a crash falls between launch reservation and confirmed PID
persistence, state becomes `launch_uncertain` and no retry is attempted. Preserve
that state and inspect it manually; do not create another state directory to
circumvent duplicate protection. SIGKILL/power loss cannot provide reliable
exactly-once process creation, so this tool deliberately guarantees at-most-once
automatic dispatch and fails closed in that narrow uncertainty window.

`state.json` is the status record. Capture outputs appear below
`STATE_DIR/capture_outputs/`; the launcher log is `capture_launcher.log`. A final
`video_complete` requires the capture supervisor's success, exact owned cleanup,
decoded MP4 verification and matching final capture artifacts. Source, input and
external capture-tool hashes are checked again independently of the successful
supervisor flag. A live follower also requires its child process to exit zero;
after restart, the bound final supervisor and primary report provide completion
evidence. Terminal states
are never automatically relaunched. Existing partial output is preserved.

CPU checks:

```sh
uv run python -m unittest discover \
  -s artifacts/mkii_fourbar_2026-09-06/campaign_capture_followup -p 'test_*.py' -v
```

Twelve tests cover full-phase selection, failure/pause/scratch rejection, report
corruption, supervisor failure, checkpoint/symlink escape, fsynced reservation and
restart duplicate rejection, fresh capture-output enforcement, complete source/
input/tool lineage, campaign PID arguments, and pidfd signal ownership and exit
races.
They do not launch subprocesses, Docker, or a GPU workload.
