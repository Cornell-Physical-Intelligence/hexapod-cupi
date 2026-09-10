# Exact reference002 → reference003 landing delta

This preserves a **CPU-reviewed preparation**, not new physics evidence. It contains one changed runtime file, complete target source identity, the frozen landing-owner tests/report/actual-prefix inputs, and integration verification. It does not duplicate the924-file parent source.

Parent source manifest: `34390c6162f462baa4c531f1d5aff346ffbc98d4d0266a68ea797f5e5b992ee0`. Target924-file source manifest: `7c75f0372abcea9eace3a280a2c204164179b8c433fc9d6280f60dd189737e24`. Only `tools/wave_reference.py` and `source_origin.json` differ. Startup, host, physical configuration, assets, residual, telemetry and physical acceptance gates remain unchanged. New source requires its own full fresh standing admission.

The new contact-triggered C2 landing retains measured flight/descent and original endpoint/3D preload bounds. It requires three stable contact samples after the blend and rejects contact loss. Actual002-prefix replay includes a **1.6968 mm planned horizontal overshoot and return**; this remains explicit in `wave002/report.json` and the diagnostic image. It is not evidence of physical friction, slip, completed landing, successful walking or stopping. The synthetic continuation uses prescribed measured support, labeled accordingly.

Independently rerun tests:24 unchanged adapter tests and16 wave/landing tests, all passed. Original owner evidence and all15 owner file hashes are retained under `wave002/`; the executed integration's source/preflight is under `integration_review/`. Source geometry/FK inputs remain identical to the prior wave; no CAD or mass change was made.

To verify the exact virtual merged source without duplicating it:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 reconstruct_source.py --parent-source /path/to/reference_source002
```

Add `--output /fresh/reference_source003` only when needed. The recipe verifies every parent file/noextras, validates the exact one-file runtime overlay plus source origin against target identity, and refuses overwrite. Obtain the parent through the already-published reference002 reconstruction recipe. Nothing here launches a simulator or changes compute ownership. Root supplies the separate fresh result, cleanup and pause029 restoration evidence after dispatch.
