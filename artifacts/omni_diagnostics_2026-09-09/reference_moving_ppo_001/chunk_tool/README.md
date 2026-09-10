# Lossless publication chunks for terminal telemetry

This preparation splits an original binary artifact into ordered raw chunks of at most **64 MiB (67,108,864 bytes)**. It does not decompress, recompress, modify, truncate or reinterpret NPZ contents. Original remote and local files remain intact. Each chunk carries its byte count and SHA256; the manifest also binds the original total size and whole-file SHA256. The original logical path is metadata only.

Use only after a job is terminal and the complete original file has been retrieved and matched to its remote hash. For a large terminal artifact:

```sh
python3 chunk_artifact.py split --input /preserved/run/smoke/trace.npz --chunks /new/publication/chunks/smoke_trace --logical-path smoke/trace.npz
python3 chunk_artifact.py verify --chunks /new/publication/chunks/smoke_trace
python3 chunk_artifact.py reconstruct --chunks /publication/chunks/smoke_trace --output /new/reconstructed/trace.npz
```

The terminal wrapper should record the raw remote whole-file hash, the chunk manifest path/hash, and all chunk paths/hashes in its own immutable manifest. Its portable verifier can call `consume()` to stream every chunk and verify the original hash without reconstructing or allocating large memory. Reconstruction verifies each chunk and the whole original during streaming, then installs the result atomically without replacing an existing file. Neither mode imports Isaac, NumPy or Torch.

Fresh chunk/output paths are required. Changed, missing, reordered, truncated, additional or symbolic chunk files reject. Invalid sizes and traversal paths reject. A changing or truncated original rejects; a partial failed split directory must not be published. The original-file hash must additionally match the independently recorded remote hash before the wrapper freezes. No active campaign has been fetched or changed by this preparation.

Eight focused tests pass with standard-library Python: boundary/empty-file round trips, exact content preservation, chunk and whole-hash corruption, missing/reordered/traversal/extra entries, overwrite refusal, failed reconstruction cleanup, size limits, source truncation during read, and symbolic/output-path guards. Small chunk sizes exercise boundaries without fabricating large sensor data. A terminal artifact test remains required when actual large bytes arrive.
