# BC dataset 002 — native rows only

Exactly 3,760 distinct raw control/replica pairs: the original 1,860 accepted steady rows remain byte-exact as the first block; 1,500 screened onset rows cover 15 commands and 25 replicas; 400 actual zero-command controls from replicas 0 and 21 cover reset and four seconds of settling. The final one second alone is labeled settled. Zero command has 640 rows (17.02%).

SELECTION.json preserves every inclusion/exclusion and model/physics binding. ROW_PROVENANCE.jsonl maps every row to source bytes and control/physics counters. RECONSTRUCTION.json verifies all five native proprioceptive history frames, command, previous applied action, requested target, and native q/dq/AMP endpoints. No cycle-3 rows enter the dataset. The temporal holdout still shares the same trajectories and prior selection; it is not independent validation.

The original realized_prior.npz (22f7b04b…) remains the AMP dataset. This new NPZ is BC-only; its velocity label is actual pre-hold body-origin navigation velocity, never an actor input. No walking-to-stop trajectory is invented or included. The four-second zero prefix does not establish the 20/32-second quiet requirement. No native execution or qualification occurred in this artifact.
