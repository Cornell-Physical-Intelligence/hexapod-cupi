# PhysX build and TGS velocity-iteration warning review

**The actual profiler used Omni PhysX 110.1.13, internal build `c38f7d1`
(June 4, 2026), with a compiled PhysX 5.9.0 serialization-version marker.**
The warning about more than four TGS velocity iterations refers to the change
introduced in PhysX 5.3. It is not a new diagnostic proving why this robot
overturned. NVIDIA's matching **110.1** guide recommends approximately one TGS
velocity iteration, so a controlled lower-velocity-iteration recipe is a
reasonable next investigation if the smaller outer timestep remains unstable.
No solver, model, runtime or acceptance parameter was changed here.

## Exact installed build evidence

| Item | Observed identity |
| --- | --- |
| Runtime image | `isaac-lab-base:latest`, Linux ARM64 |
| Image ID | `sha256:8ddc1623d70d5dd622fd728ce4eb3f59dea6ce1ea5cfbe858353dab15e8d0ef8` |
| Isaac Sim VERSION | `6.0.1-rc.7+release.42383.32955d8d.gl` |
| PhysX extension | `omni.physx-110.1.13+110.1.2.la64.r.cp312.u7f4` |
| Embedded integration build | `110.1.13`, `c38f7d1`, `HEAD`, `omniverse/physics`, `Jun-04-2026` |
| Core plugin SHA-256 | `a66e7338758473c3361e7d81730a2fab3c1f746329e4e99541bf7e7834ab4b35` |
| Extension TOML SHA-256 | `36fecddd676f1e5de19157684113afe94a4a90b8fbedd59827c164abaf7b2f9e` |

The core plugin is
`/isaac-sim/extscache/omni.physx-110.1.13+110.1.2.la64.r.cp312.u7f4/bin/libomni.physx.plugin.so`.
`installed_image_metadata.json` records the file hashes, extension metadata,
installed core changelog and relevant binary strings with byte offsets. The
binary contains the exact warning and the `5.9.0` string adjacent to PhysX's
`RepXCollection::create` serialization code. The corresponding public code builds
that string from the SDK major/minor/bugfix macros, whose published values are
5/9/0. This is static compiled-file evidence; no version API or GPU initializer
was executed. [Version header](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/include/foundation/PxPhysicsVersion.h#L48),
[serialization version construction](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/source/physxextensions/src/serialization/Xml/SnXmlSerialization.cpp#L719).

The evidence is tied to the actual running profiler, not just another installed
copy: `live_profiler_image_binding.txt` records container
`631cfc7d4a18cddf24136f3b29f9ae03c4027a5785f5f7a830f0f62aefb9c04e`,
the same image and hashes, and **PID 15's loaded-library mappings** pointing at
that exact plugin path. The read did not inspect process memory or import any
SDK module.

Before that live binding became available, `inspect_image_files.py` created a
unique metadata container from the image, copied the selected files, and removed
that exact container. It was **never started**: before/after state was `created`,
PID 0, with an unset start time. No simulation or GPU job was launched. One
optional bundle-changelog path did not exist. A subsequent optional header
query found that the profiler owner had already removed its container; those
limitations are retained in the evidence and do not negate the captured binding.

The official public tag `110.1-omni-and-physx-5.9.0` resolves to
`517a0073715120e114ee055b63b26c95e00d9039`, published July 14, 2026. Its hash is
**not** the installed internal build hash. It establishes the matching release
family and a reproducible source reference, not an exact private-to-public
commit mapping or proof that every public-tag fix is in the installed binary.
Earlier 5.6.1/5.8 citations should not be presented as the installed version.

## What changed at four velocity iterations

The SDK changelog's **v5.3.0-105.1** entry explains that older TGS converted each
requested velocity iteration above four into a position iteration on both CPU
and GPU. Newer TGS uses the requested counts. For example, the old behavior of
64 position / 16 velocity iterations corresponds to 76 / 4, not 64 / 16.
This conversion is historical arithmetic, **not a proposed setting change**.
[Pinned SDK changelog](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/CHANGELOG.md#L1207).

The matching Omni Physics 110.1 limitations page explicitly connects that change
to the warning and lists few/zero velocity iterations or PGS as alternatives.
Its simulation guide recommends around one TGS velocity iteration and notes
that applying external forces every iteration can help convergence. Neither
page claims that the warning means every affected articulation is unstable.
[110.1 limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/guides/current_limitations.html),
[110.1 solver guidance](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/simulation_control/simulation_control.html#physics-solver).

Both the 64/16 near-pass and the 128/16 overturn used sixteen velocity iterations.
The warning therefore does not isolate their difference. More position
iterations also change TGS's effective integration; they are not simply a
monotonic quality dial for this contact/constraint system. That is an inference
from the run comparison and documented timestep behavior, not a proved root
cause. [110.1 mimic tuning](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/rigid_bodies_articulations/articulations.html#tuning-a-compliant-mimic-joint).

## Contact and mimic consequences relevant to this robot

- The matching articulation guide documents instability when hard mimic
  constraints compete with contact and stiff actuation, particularly on light
  links. It supports smaller outer timesteps for closed loops. It also explains
  that mimic compliance changes the constraint's physical response; adding it
  would be a new model assumption, not a free numerical fix.
  [110.1 articulation guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joint-compliance).
- The D6-drive/TGS warnings concern implicit joint drives. Our current
  `RS05V2Actuator` derives from `IdealPDActuator` and supplies limited explicit
  effort. Those warnings do **not** directly establish an actuator bug here.
  The separate closed-loop/contact caveats remain relevant.
  [110.1 limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/guides/current_limitations.html).
- The SDK history contains older fixes for TGS nonzero-velocity timesteps on
  CPU, PGS/CPU mimic/limit velocity handling, and D6 drive-force behavior. They
  must not be cited as newly diagnosed bugs in this TGS/GPU run. The 110.1.8
  release also documents improved incoming joint-force readback with external
  forces every iteration. Incoming-joint-force readback is distinct from our
  raw/applied motor telemetry and net-contact measurements.
  [SDK history](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/physx/CHANGELOG.md),
  [110.1 bundle changelog](https://github.com/NVIDIA-Omniverse/PhysX/blob/517a0073715120e114ee055b63b26c95e00d9039/omni/ovexts/extensions/ux/source/omni.physx.bundle/docs/CHANGELOG.md).

## Bounded next experiment, if needed

Keep the currently selected outer timestep, position iterations, explicit motor
model, physical asset, initial state, commanded motion and all existing gates
fixed. Compare **16 versus 1 velocity iteration** as separately versioned
numerical recipes, after the current experiment finishes. Preserve every
physics-substep measurement and the before-reset guard. Record closure,
passive/pin velocities, support/contact history, attitude, motor demand and
applied effort; a small closure gap alone does not establish stability.

This is a proposed diagnostic, not an instruction to weaken the numerical
contract. A promising result still needs the full admission sequence and the
required resolution comparison under its own source identity. Only after
stability at a smaller timestep should fewer position iterations be evaluated
for throughput, as NVIDIA recommends. Do not add unmeasured inertia, damping,
friction, compliance or speed limits merely to suppress a failure.
[110.1 stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/guides/articulation_stability_guide.html#reduce-simulation-timestep-for-complex-systems-and-closed-loops).

A collateral CAD-review caveat: 110.1 documents a GPU convex-hull limit of both
64 vertices **and faces**. The uninstalled pad proposal's 64 vertices and 124
triangles do not establish its cooked polygon-face count or GPU compatibility.
It remains unqualified; no collider work was performed here.
[110.1 limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/guides/current_limitations.html).

## Evidence verification

`public_source_manifest.json` pins the downloaded primary source files and their
hashes; their BSD license is included. `public_tag_commit.json` records the tag
commit date. Run `python3 verify.py`, then `shasum -a 256 -c SHA256SUMS` in this
directory. The binary itself is not redistributed: its image identity, recorded
hash, static markers and live mapped path provide the audit trail.
