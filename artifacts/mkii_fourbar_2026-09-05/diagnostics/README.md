# Preserved incomplete diagnostics

These runs do not admit training and have no complete primary report.

- `053455Z-88fcc531`: original v3 LF knee diagnostic completed its trace, but the old outer report writer did not execute before native Kit shutdown. Supervisor correctly rejected the missing report. The lifecycle fix is in6734539; raw traces are retained here.
- `060625Z-e3d6aee3`: D6 v4 group diagnostic was interrupted because the host detector matched the Python command in a blocked weather `flock` wrapper. The GPU was not occupied by weather. Logs/source hashes/supervision are retained here; partial NPZ traces remain in the corresponding Spark run directory.

Spark root: `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/runs/hexapod-fourbar-diagnose-<timestamp-id>/`.
`admitted` is the host CPU/resource startup barrier, not physical training admission.
The raw file inventory records bytes only. No substitute primary report has been created.
