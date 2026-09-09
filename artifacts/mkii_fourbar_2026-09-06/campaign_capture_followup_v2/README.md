# One-shot campaign video follower v2

This version preserves the original full 512-environment/1,000-update completion,
lineage, PID/start-ticks/boot-ID checks and at-most-once dispatch. It adds two
explicit deployment bindings needed by the new 1600 Hz campaign:

- `--campaign-wrapper PATH --campaign-wrapper-sha256 HASH` binds the reviewed
  external campaign entrypoint. Exact source/commit/output argv checks remain.
  Wrapper bytes are checked at startup, every campaign-header poll, immediately
  before launch reservation, and before accepting the final video. A changed or
  absent wrapper rejects; the frozen original campaign entrypoint remains the
  default when both options are omitted.
- `--capture-shared-lock /opt/wx/gpu.lock` starts only the actual capture child
  with `/usr/bin/flock --nonblock --no-fork /opt/wx/gpu.lock`. Waiting for the
  campaign never owns this lock. Busy admission fails closed, with no automatic
  retry. The reserved argv contains the flock prefix; exact process identity is
  recorded after exec as the Python capture-supervisor argv with the same PID.
  The existing capture launcher still checks the canonical coordination file,
  GPU/resource inventory, source/inputs and exact container ownership.

The published original follower and policy_capture_tools_v2 remain unchanged.
The latter already writes encoder/NPZ/metadata/report outputs before Kit's hard
shutdown boundary and verifies the video from its separate host supervisor.
No task automation is created. No persistent GPU reservation is made.

Seventeen focused CPU tests pass, retaining the original twelve and adding
wrapper binding/tampering/terminal-state checks, paired CLI validation, and the
nonblocking no-fork capture argv/reservation test. The independent Linux CPU
probe in `../1600hz_campaign_host_v1/linux_flock_cpu_probe.json` establishes
no-fork PID preservation and lock lifetime, using a temporary lock rather than
the real GPU lock. These checks do not constitute actual policy-video evidence.

Exact deployment parameters for the current c2af43c campaign are recorded in
`../capture_followup_1600hz_deployment_v1/deployment_config.json`. Restart only
with the identical arguments and state directory; an existing ambiguous launch
reservation must never be bypassed with a fresh directory. The follower waits
at most 18 hours and bounds its one capture supervisor to 2,160 seconds; the
capture's existing render timeout is 1,800 seconds.
