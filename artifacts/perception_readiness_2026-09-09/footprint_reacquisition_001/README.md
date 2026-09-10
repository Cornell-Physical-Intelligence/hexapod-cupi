# Causal evidence for a proposed landing footprint

This CPU prototype checks sensor evidence for the original planned landing region using actual004's recorded motion and the frozen hypothetical six-camera visibility cache. It keeps measured planted contacts separate from optical evidence for new footholds. Root independently passed all 16 tests, verified frozen inputs and inspected the figure; a separate review checks every reference/capture/receipt alignment. It makes no walking, hardware, mount or safe-abort claim.

Across 377 active-foot queries, complete footprint coverage is fresh now in 169 cases, but existing evidence covers the original planned landing time in none. Added registration drift reduces fresh-now cases to 156. A synthetic missing frame is detected without renewing old map timestamps; a longer outage makes previously observed cells stale. The figure and [full report](owner/README.md) distinguish map coverage, whole-footprint decisions, and time remaining to planned landing.

![Coverage and landing time](owner/footprint_lease.png)

The result uses a 15 mm footprint proxy, 20 mm grid, inferred camera intrinsics and synthetic registration/noise assumptions on flat ground. The 250 ms capture-age lease has only 210 ms left after a 40 ms receipt delay, while the source reference swing lasts two seconds. No future frame is assumed. Missing observations remain unknown, observed pits remain ineligible, and a planted contact cannot fill adjacent terrain cells. These are conservative evidence decisions under a declared prototype contract, not a recommendation to buy six cameras.

The next integration requires a physically executable decision, landing and stop interface, plus a measured contact-patch and registration model. Merely extending the lease would not establish those facts. The admitted slow reference's 5.54-second stop latency is retained as context and is not used to invent a safe abort trajectory.

Original 24-payload owner freeze and independent review remain unchanged under separate subdirectories. The latter verifies all 426 reference rows, 85 capture/receipt pairs at exactly 40 ms, and raw contact/motion-array equality. The owner documents exact source inputs and reproducible tests; its recorded-data replay does not run Isaac or create new physical evidence.
