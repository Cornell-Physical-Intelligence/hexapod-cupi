# Hand-off: Hexapod MKII product-launch showcase film

Written 2026-09-04 for the next agent (GPT Astra). Read `AGENTS.md` and `CLAUDE.md` first for the
repo rules; this file covers only the showcase-film work stream, which is unfinished.

## 1. Where everything is

Repo: `/Users/andreboufama/Documents/CUPI/HEXAPOD`, branch `codex/isaaclab-training-artifacts`
(PR #12 on GitHub carries this branch). Nothing from this work stream is committed yet.

| What | Path |
|---|---|
| The film (source of truth, UNCOMMITTED) | `robot/hexapod_mkii_assy/preview/showcase_template.html` |
| Shared viewer engine (small UNCOMMITTED change: `materialOverride` hook, `key/hemi/PO` exported on the api) | `robot/hexapod_mkii_assy/preview/hexapod_core.js` |
| Packer that inlines core + model + meshes into one HTML | `robot/tools/pack_urdf_viewer.py` |
| Packed film, latest (take 9), 9.7 MB, kept out of git | `tmp/hexapod_mkii_showcase.html` |
| Take 9 master mp4, 1920x1080, 30 fps, 92 s, 89 MB | `tmp/showcase_handoff/hexapod_mkii_showcase_take9_1080p.mp4` |
| 720p previews sent to the user (takes 8 and 9) | `tmp/showcase_handoff/hexapod_mkii_showcase_720p_take{8,9}.mp4` |
| Review sheets and 4-second key frames of take 9, extracted from the mp4 | `tmp/showcase_handoff/review_v9/` |
| Frame-save server (browser POSTs PNG/JPEG data URLs to it) | `tmp/showcase_handoff/save_server.py` |
| Encode + extraction script | `tmp/showcase_handoff/make_video.sh` (paths inside point at the old session scratchpad; edit `SP`, `SRC`, `OUT`) |
| Interactive viewer template (committed, separate page) | `robot/hexapod_mkii_assy/preview/standalone_template.html` |
| Published showcase artifact (older live-render version, NOT updated with the film) | https://claude.ai/code/artifact/8d388907-ee3f-46a3-88c9-ea96b01a25b8 |

`tmp/` is ignored by git on purpose (CLAUDE.md: `tmp/` and packed viewer HTML stay out of git).

## 2. Goal and acceptance rule (from the user, verbatim in spirit)

Produce a website-ready showcase animation of the Hexapod MKII (rotations, exploded views,
fly-throughs, walk) with no debug UI, a plain white (or any solid) background, and:

- the film must START in the URDF default (rest) pose, not a chosen pose;
- everything must flow smoothly into everything else (no jerks);
- quality "comparable with an official product launch of an R&D product";
- **"until an independent judge rates the animation higher or equal to an official animation
  product launch, do not approve it"** - i.e. do not publish, send as final, or commit until an
  independent judge agent returns PASS (overall >= its own benchmark, ~8.2/10);
- the user later asked that the export and the rating be based on an offline-rendered mp4, not on
  live rendering in a browser. That is now the pipeline.

Hard constraints to keep (tell every judge not to penalise them): rest pose at start and end;
the standing/walking pose is the validated control stance from `stance.json` and must not be
made taller; 92 s loop; one continuous take; plain page background; real-time engine rendered
frame by frame (no path tracing / DOF / motion blur); short captions.

## 3. Score history (all FAIL so far, benchmark 8.2)

| Take | Overall | Notes |
|---|---|---|
| 2 | 5.5 | live-render contact sheets |
| 5 | 5.3 | live-render sheets |
| 7 | 6.0 | first offline mp4 |
| 8 | 5.7 | offline mp4 |
| 9 | not scored | judge was stopped mid-review at the user's request |

Take 9 contains every fix from the four reviews. It is the cut to have judged next.
The judge prompt used every time (keep it identical for comparability) is in section 7.

## 4. Pipeline (how to make a take)

1. Edit `robot/hexapod_mkii_assy/preview/showcase_template.html`.
2. Pack:
   ```bash
   python3 robot/tools/pack_urdf_viewer.py --template robot/hexapod_mkii_assy/preview/showcase_template.html --out tmp/hexapod_mkii_showcase.html
   ```
   (a harmless numpy "divide by zero in matmul" warning is printed; ignore it.)
   Quick syntax check of the module script:
   `node -e "const h=require('fs').readFileSync('tmp/hexapod_mkii_showcase.html','utf8');const js=h.split('<script type=\"module\">')[1].split('</script>')[0];new Function(js.replace(/^\s*import\b.*$/m,''));console.log('ok')"`
3. Serve the repo root over HTTP (any static server, e.g. `npx http-server . -p 8321`; the
   `.claude/launch.json` entry `hexapod-preview` does exactly this) and start the save server:
   `python3 tmp/showcase_handoff/save_server.py <output-dir>` (listens on 8322, CORS `*`,
   `POST /save?name=<n>` with a base64 data-URL body; it always writes `<n>.png` even when the
   body is JPEG, so rename to `.jpg` before encoding).
4. Open the packed page in a browser with export parameters. It renders deterministically
   (no wall clock) and POSTs each frame:
   - full render for the mp4: `tmp/hexapod_mkii_showcase.html?export=frames&fps=30&fmt=jpg&post=http://localhost:8322/save&name=v10&w=1920&h=1080`
     (2760 frames, ~3 min on the user's Mac, ~420 MB of JPEGs)
   - quick checks: `...?export=frames&fps=1&from=12&to=30&...` (any range), or
     `...?export=sheets&every=1&...` (6x4 contact sheets straight from the page)
   Other page params: `bg=`, `face=`, `motor=`, `accent=`, `edge=` (hex without #), `captions=0`,
   `speed=`, `vignette=0`, `mark=0`. Without `export=` the page just plays the loop (click pauses).
   Note: a hidden browser tab pauses requestAnimationFrame; the export path uses timers, so it
   still works hidden, but the interactive loop does not.
5. Encode and extract review material from the mp4 (edit the three path variables first):
   `zsh tmp/showcase_handoff/make_video.sh v10 30` -> `hexapod_mkii_showcase.mp4` plus
   `sheet01..04.png` (1 tile per second, stamped top-right), `key0001..0023.png` (exact frames at
   t = 0,4,8,... s) and `sec0001..0092.png` (one per second, 480 px). It uses
   `select='not(mod(n\,30))'` on purpose: ffmpeg's `fps=` filter picks frames 2 s late.
6. Motion check (per-second mean abs frame difference; the loop seam 91->0 must be within the
   neighbours; no "quiet" seconds < 3):
   see the python snippet in the transcript of this session, or reimplement (PIL + numpy).
7. Spawn an INDEPENDENT judge agent with the prompt in section 7 pointing at the new folder.
   Fix its "TOP FIXES", re-render, repeat until PASS.
8. After PASS only: republish the artifact at the URL above (pass `url=` so the link is kept;
   the page is the packed HTML), send the 1080p mp4 to the user, commit ONLY
   `hexapod_core.js` and `showcase_template.html` (plus the package README if features changed),
   push the branch. Never commit `tmp/`.

## 5. How the film is built (inside `showcase_template.html`)

- `createHexapodCore(...)` from `hexapod_core.js` loads the packed model (linkage URDF, 31 links,
  mimic joints) and gives `V` (renderer, scene, camera, `view` = {theta, phi, r, target},
  `meshes` with `userData {part, link, leg, pclass}`, `setExplode`, `gait`, `frameVisible`, ...).
  `V.materialOverride = m => material` is honoured by `applyMode()` (added this session).
- Materials: `MAT` (hero leg + body) and `OTHER` (the five non-featured legs, so they can ghost):
  structure warm aluminium (`face`, default `#cdc4b6`, metalness 0.5), motor anthracite physical
  with clearcoat, fastener grey, cover charcoal, foot rubber. `MAT.femurGhost/coverGhost` make
  the featured femur translucent during the hip/knee beats. Highlight sets `HI.yaw/hip/knee`
  are the actuator parts clustered to the nearest of the three RS05 housings of the `lf` leg
  (yaw = housing in `lf_coxa`; hip = femur housing nearest the yaw one; knee = the other), plus
  `HI.linkage` (pushrod + push-lever structure). `hiMats[...]` colour-lerps to amber.
- Light rig follows the camera azimuth (`rig(theta)`: key +0.7 rad, rim opposite), VSM soft
  shadows, contact-shadow decals under each foot (vertex-accurate lowest point), a dot-grid floor
  (0.125 m pitch, 30 %) that fades in for the stance/walk and scrolls like a treadmill
  (`gaitDist`), a background wipe disc that sweeps from the bottom-left corner (`placeWipe`).
- Timeline: `T = 92`. `CAM` = keyframes `[t, theta, phi, distance, targetFn, shiftRight, shiftUp]`
  where distance is a factor of `r0` (rest-pose fit), `{m: metres}`, `"exp"` (exploded fit) or
  `"st"` (stance fit); fits are computed per shot over the azimuth range that shot sweeps,
  ignoring fasteners (`fitAll`). Interpolation is a closed Catmull-Rom/Hermite spline; theta is
  wrapped by 2 pi at the loop so the spline is C1 across 92 -> 0. Theta DECREASES through the
  loop (one full turn). Held shots sit between the measured rest-pose leg directions
  (lf -0.93, rf -1.69, rm -2.82, rr -4.07, lr -4.83, lm -5.95 rad) so no leg points at the lens.
- `E` = smoothstep envelopes (links/parts explode, edges, faceOp, occlude, wipeIn/Out, edgeCol,
  shadow, fastOp, stand, gait, floor, legdemo, kneedemo, ghost, focus, hiYaw/hiHip/hiKnee).
  `TITLES`/`SUBS` = captions with independent fades (`capAt`). `renderAt(t, tAbs)` =
  `applyPose` -> `applyLook` -> `applyCamera` -> decals -> captions -> render.
- Beats (take 9): 0-7 reveal (continuous orbit + dolly, knee ripple 0.8-2 s); 7-13 approach;
  13-27 left-front leg with the other legs ghosted: yaw drive amber 15-18, hip 19-22 (femur
  translucent), knee + four-bar 22-26 with the knee flexing; 26.5-29.5 pull back; 31-43 exploded
  (assemblies then parts, breathing drift); 43.6-44.9 wipe to navy blueprint; 45-50 implode;
  50.6-51.9 wipe back; 52-57 x-ray; 57-63 stand up (root lifted from foot contact); 63-84 tripod
  gait with body bob; 84-88.5 settle; 85.5-90.8 closing caption; loop.
- Authoring helpers exposed on `window.__show` (page loaded without `export=`):
  `V, renderAt, camera, env, E, CAM, applyPose/applyLook/applyCamera, partW(name), jointW(name),
  computeFramings, probe(setName)` (fraction of an actuator visible + subject bounds +
  caption-zone occupancy, via magenta render passes) and `searchBeat(setName, t, grid)` (grid
  search of theta/phi/r/shift/lift). Stop the animation loop first
  (`V.renderer.setAnimationLoop(null)`) and call `V.resize(1600, 900, false)` so the aspect is
  16:9 before probing. Run long searches inside `setTimeout` and poll a global; the browser tool
  call times out at ~45 s.

## 6. Known open issues / what the next review will probably flag

- The aluminium at `#cdc4b6` reads slightly khaki; a judge asked for sRGB 205/196/182, so it is
  by request, but check it against the page `#fafbfd`.
- ~88 s the camera passes the lf leg direction while the robot is settled (leg points at lens
  briefly). Options: finish the walk at theta -6.95 instead of -6.75, or start the loop at -1.45.
- The yaw macro (15-18 s) is very low (phi 1.66); verify the yaw can is not 60 % hidden by the
  coxa plate in the mp4 frames (judge complaint on take 8).
- The exploded hold: verify the lowest foot stays above 92 % of frame height at 31-50 s (both
  judges flagged cropping before; take 9 shortened the explode travel to links 0.6 / parts 0.4 and
  fits framing without fasteners, but it has not been re-judged).
- Single-frame pops were removed by turning toggles into fades (shadows off for the whole ghost
  beat, fasteners fade via `fastOp`, depth-write flips only when faces are opaque again); confirm
  with the per-frame difference trace that nothing exceeds ~1.5x the local median outside the
  two wipes.
- A judge suggested a "closing caption" and an earlier approach; both are in take 9.

## 7. Judge prompt (use verbatim, only change the folder/take names)

Spawn a general-purpose agent with no other context and this prompt:

> You are an independent reviewer of launch films for R&D robotics products (the standard you
> compare against: official reveal/launch animations from companies like Boston Dynamics,
> ANYbotics, Agility, DJI or Apple product films). You have not seen this project before and you
> do not report to the people who made it. Be strict, specific and honest; a generous review is
> useless.
>
> WHAT TO REVIEW: a 92-second looping film of a hexapod robot (the "Hexapod MKII", a six-legged
> research platform with 18 actuators), rendered offline frame by frame and encoded as an mp4:
> `<path to mp4>` (1920x1080, 30 fps, 92.0 s). You cannot watch video directly, so review material
> was extracted FROM THAT MP4 at exact frame indices into `<folder>`: contact sheets, one tile per
> second, 24 tiles per sheet, each tile stamped top-right with its time: sheet01.png (t = 0..23),
> sheet02.png (24..47), sheet03.png (48..71), sheet04.png (72..91); full-resolution key frames
> every 4 seconds: key0001.png = t 0 s, key0002.png = 4 s, ... key0023.png = 88 s (read at least
> key0001, 0003, 0005, 0006, 0007, 0010, 0012, 0014, 0016, 0019, 0021, 0023); one frame per second
> at 480 px wide: sec0001.png = 0 s ... sec0092.png = 91 s. You may extract any exact moment with
> `ffmpeg -y -loglevel error -ss 27.5 -i <mp4> -frames:v 1 /tmp/f.png` (ffmpeg is at
> /opt/homebrew/bin/ffmpeg); use it for transitions at sub-second resolution (13-15 s, 27-30 s,
> 43-45 s, 50-52 s, 57-59 s, 91-0 s loop point). Use the Read tool to look at the images. Judge
> frame-to-frame coherence of the camera path and the animation from the 1-second material.
>
> WHAT THE FILM INTENDS: <paste the beat list from section 5, with the wipe, ghosting, amber
> highlights, stance lift, treadmill grid, closing caption, one continuous camera spline that
> circles once per loop, velocity-continuous loop point, camera-relative key+rim light, materials>.
>
> HARD CONSTRAINTS YOU MUST NOT PENALISE: the film starts and ends in the robot's URDF rest pose
> (legs splayed flat, spider-like); the standing/walking pose is the robot's validated control
> stance and must not be changed (body low, legs splayed ~45 degrees; do not ask for a taller
> stance); real-time-engine render exported frame by frame (no path tracing, DOF, motion blur);
> one continuous take without cuts; plain near-white page background; short captions; total
> length fixed at 92 s.
>
> RUBRIC, score each 1-10 (10 = indistinguishable from the best official launch films):
> 1 Composition and framing; 2 Camera motion and continuity; 3 Pacing and structure (loop point
> invisible); 4 Lighting, materials and visual quality; 5 Typography and captions; 6 Clarity of
> story; 7 Polish (no glitches, pops, odd colours, half-transitions). Then an Overall score, and
> separately the score you would give a typical OFFICIAL launch animation from an R&D robotics
> company on the same rubric (usually ~8.0-8.5). Verdict: PASS only if Overall >= benchmark.
>
> OUTPUT FORMAT (exactly): a JSON block
> `{"scores": {"composition","camera","pacing","visual","typography","story","polish","overall"},
> "benchmark_overall": n, "verdict": "PASS"|"FAIL"}`, then a 150-250 word prose review naming the
> strongest and weakest moments with time stamps, then "TOP FIXES": the 5 most impactful,
> concrete, implementable changes with time stamps and numbers.

## 8. Gotchas learned the hard way

- three.js `alphaMap` reads the GREEN channel: paint alpha textures as luminance on black.
- Mesh origins sit at the part frame, not the part centroid: use
  `m.localToWorld(m.geometry.boundingSphere.center.clone())` for part centres.
- The RS05 yaw and hip drives are only ~50 mm apart; cluster parts to the nearest housing, never
  by a fixed radius.
- Adjacent legs are 60 degrees apart, so a clean single-leg side view is impossible without
  ghosting the other five legs; that is why the "focus" ghost exists.
- In the stance the legs extend below the rest-pose floor; the root must be lifted from the lowest
  foot each frame (`measureFeet`), otherwise feet sink through the shadow plane.
- `frameVisible` fits depend on azimuth and on flying fasteners; fit per shot and without
  fasteners.
- The shell cwd in the tool sandbox resets between calls; always use absolute paths.
- The save server names every file `.png`; ffmpeg needs the JPEG frames renamed to `.jpg`.

## 9. Not part of this stream (already done and committed)

The CAD URDF package (`robot/hexapod_mkii_assy/`), the Isaac Lab re-pointing, validation tooling
and the agent runbook are complete and pushed; see `AGENTS.md`, `HANDOFF.md` and the package
README. The interactive viewer (`standalone_template.html`) is committed and published as its own
artifact; only the film is outstanding.
