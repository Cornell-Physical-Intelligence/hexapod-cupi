# Local clearance arithmetic cross-check

Root observed NumPy matrix-multiply warnings while rerunning CPU tests. This separate read-only check leaves the frozen source unchanged and does not suppress warnings. It evaluates the15 actual reset/captured/failed poses from standing001 across all19 bodies and both full and clipped non-toe clouds:675,510 plane dot products.

Every result is finite. Matrix and elementwise outputs agree exactly; maximum difference from independent Python `math.fsum` dot products is2.7756e-17m. All full geometry minima agree within1e-14m. Local stderr was empty in this run. This verifies these actual finite inputs; it does not establish the source of root's warning, guarantee all possible native states, or claim a native standing pass. Root may retain its warning log alongside this evidence.

`check.py` reproduces `report.json` using exact frozen source002 and unchanged actual001 raw. No source, raw, remote or GPU state was modified. Root owns central publication.
