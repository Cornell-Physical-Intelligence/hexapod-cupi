# Detailed canonical CAD showcase

**Selected video:** [hexapod_detailed_cad_showcase_final.mp4](hexapod_detailed_cad_showcase_final.mp4).

This24-second,1920×1080,H.264 video shows the exact canonical motor-mass-corrected detailed model:1,753original CAD part instances,59source meshes,19rigid bodies and18direct-drive joints. It uses original CAD colors and geometry, a clean orbit and a modest front-left femur/tibia articulation. Playback is25fps at the declared kinematic time scale1×. There is no audio.

**This is a CAD kinematic preview. It contains no learned policy, simulated gait or physical qualification.** The robot root is fixed in its inspection pose. Only LF femur and tibia move, using a9-second smooth sine-squared envelope with maxima0.22rad(12.61°) and0.28rad(16.04°), then return to zero. These modest angles stay inside the reviewed visual joint bounds; that is not a collision, support or actuator test. Joint travel and hardware limits remain distinct.

`INPUTS.json` pins the canonical selector, URDF `9492fde5…`, model `7126935b…`, original viewer sources and all59meshes. The renderer uses the model's exact named joint hierarchy and original instance transforms, with no geometry simplification. `capture/showcase.js` is an isolated capture adaptation of the existing inspector; repository viewer/model/URDF files were not edited. Camera, lighting and labels are presentation only. `capture_trace_final.json` records every frame's named angles, model identity, camera and projected assembly bounds.

The final camera keeps the complete articulated assembly inside a caption-free region throughout the clip. The opening,8s,16s and22s frames are retained under `qa_final/`; the final MP4 was decoded in full. `VIDEO_VERIFICATION.json` records exact size/hash,600frames,25fps,24s, source/angle/bounds checks and visual QA. The final poster is `poster_final.png`.

## Drafts and capture history

The earlier `hexapod_detailed_cad_showcase.mp4` and its trace/QA are **draft evidence**, retained after root noticed cropped limbs in its close/return framing. Do not select that draft for the poster. The final uses a distinct filename and corrected full-assembly framing. A short rejected intermediate camera-alias attempt and the slow software-renderer attempt are explicitly named as rejected/draft files; they carry no model or scientific conclusion. No frozen or previously published video bytes were reused.

The first headless renderer selected software SwiftShader and was too slow. Capture then used the installed full Chromium with requested native Metal rendering. Rendering was local, from the actual repository CAD assets served on loopback, with deterministic frames piped to FFmpeg. No Spark resource or native simulation was started. Native model admission and PPO work continued independently.

## Verify and reproduce

Run the standard-library bundle verifier:

```sh
python3 -S artifacts/mkii_updated_2026-09-10/cad_showcase_001/verify_bundle.py
```

`capture/capture.mjs` records the actual local Playwright path, Chromium selection, loopback URL and FFmpeg arguments used. Rerender into a **new** directory after adapting those machine-local paths; never overwrite this frozen artifact. The HTML import map references the unchanged repository Three.js/vendor inspector modules and exact model meshes. Source hashes and all frame records are included; the original CAD mesh files are bound rather than duplicated.

Root owns final integration/publication. The new bounded `site/updates` record covers this artifact under `docs/PROJECT_SITE.md`; shared STATUS, plan and presentation-registry edits remain root-owned. This video illustrates the approved detailed robot and does not advance a training or physical-admission gate.
