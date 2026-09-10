# Actual browser inspection of the updated CAD viewer

The root agent loaded the final viewer in the Codex in-app browser at 1280×720 and inspected actual rendered screenshots. All 1,753 parts loaded. The joint selector, knee −3°…+20° bounds, animated sweep/pause, part search, overlap filter, selection, link isolation, 15 mm separation, top camera, X-ray and original/neutral pose controls were exercised. A downloaded settings JSON was checked against the source hash and all 18 joint entries.

`qa_receipt.json` binds the exact page/model files and distinguishes the observed controls from untested performance, mobile and native-physics behavior. `exported_preview_settings.json` is the actual downloaded file at restored neutral. Screenshots were inspected in the task UI; they are not invented rollout evidence. The viewer uses the real exported meshes and forward kinematics, and is not an Isaac recording.

The robot loads at neutral with animation paused. `python3 -m http.server 8347 --bind 127.0.0.1 --directory robot` is the repository-root local preview command; the canonical readable command is in `docs/UPDATED_CAD_IMPORT.md`.
