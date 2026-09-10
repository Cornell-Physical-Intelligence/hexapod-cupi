# Caps 50-update pilot

The six acquisition phases completed and the saved 50-update checkpoint passed strict reload. Final quiet passed **0/48 replicas**. This is not Stage 2 admission. See [derived results](analysis/REPORT.md), [training receipt](raw/run/train/training_receipt.json), and [final stop diagnostics](raw/run/final_stop/stop_diagnostics.json).

The curated raw folder has all 72 selected files, including initial/final constant and moving-to-stop traces, training event/joint traces and immutable decisions 10/25/50 plus final weights. The complete 122-file remote inventory and all 50 intentionally omitted intermediate native autosaves are retained in [FETCH_PLAN.json](audit/FETCH_PLAN.json). Source, accepted phase tree aliases, original/final checkpoint identity, all 12 exact container names/IDs and timer restoration were verified in [the terminal audit](audit/remote_audit.json).

Do not run the host's full-tree admission verifier against this curated subset: it intentionally omits intermediate autosaves that remain in the original immutable remote tree. Analysis ran on that complete original tree. Historical partial-fetch receipts, when present, document the first disk-limited attempt; FETCH_COMPLETE.json and LOCAL_SHA256.json establish the completed curated fetch.
