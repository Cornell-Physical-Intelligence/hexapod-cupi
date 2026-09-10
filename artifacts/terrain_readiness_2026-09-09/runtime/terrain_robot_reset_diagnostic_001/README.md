# Full-C terrain reset diagnosis001

**Five pre-reset controls confirmed the quaternion convention fault. This is a
diagnostic, not a standing admission.** A fresh exact-plan32×1000 flat standing
gate passed first.

The unchanged bad adapter requested XYZW `[0.70710677,0,0,0.70710677]`, which is
a90-degree roll. The simulator read that orientation back. Immediately after
reset its projected gravity was `[0,-0.99999994,3.42e-8]`, already satisfying
the unchanged upside-down predicate. Before the first automatic reset, gravity
Z was+0.0287853, upside-down and base-contact predicates were true, too-low was
false and torque-excess duration was zero. The intended root translation and
zero environment origin were correct. The hook called the original done
function once per step and returned its termination flags unchanged.

- [All five pre-reset records](run/terrain/reset_diagnostic.json), [diagnostic state](run/terrain/state.json), [raw terrain log](run/logs/terrain.log).
- [Fresh flat admission](run/flat/admission.json), [campaign](run/campaign.json), [exact original result map](full_result_SHA256SUMS.json).
- [661-file source manifest](campaign_source_hashes.json), SHA-256 `93ab3251b736f7828491648c837390ad5fafe664218a3581906f20402fe4a7dc`; [original source provenance](source_origin.json).
- [Installed XYZW definition](installed_rotation_contract.json), [post-run hash/cleanup verification](post_run_verification.json), [pause016 restoration](forecast_pause_016/restored.json).

The source and all550 admitted asset files remained unchanged; both owned
containers were absent, CUDA empty and both locks free at the recorded exit
check. Both previously active StormScope timers were restored. This is an
exit-time observation, not a claim that Spark stays idle afterward.

[Provenance audit](provenance_audit.json) identifies one preserved bookkeeping
issue: `source_origin.json` inherited the parent's convenience
`bounded_host_launcher_sha256` value. The authoritative661-file source manifest
correctly records the modified diagnostic launcher as
`9368bcf8b1d91edbf3446d0dcd1cdd8bcae3f7652021caa5eef2e66b700f5425`.
The immutable origin was not rewritten. A selected executing-source subset is
included under `frozen_source/`; the complete source remains at the remote path
recorded by the campaign. [Method and launch details](METHOD.md) preserve the
bounded preparation. No traversal, derived-fixture, policy, sensing or physical
four-bar qualification follows from this diagnostic.
