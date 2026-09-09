# Exact profiler v2 output capture

Original output from `/home/orionh/HEXAPOD_runs/mkii_profile_v2/profile_001` on `spark-e26c` was copied without rewriting its bytes. All 8 selected files match SHA-256 values computed on the remote host from the transmitted original bytes. `retrieval_metadata.json` records hashes, sizes, timestamps, the exact read-only retrieval script and the SSH result.

The validation report records `pass=true`. This is a profiler capture, not physical training admission or a completed PPO run. The `admitted` marker records launcher/resource admission; it is not a passing physical qualification. Interpret timing using the corresponding analysis artifact and its measurement limitations.

The trace remains remote at `/home/orionh/HEXAPOD_runs/mkii_profile_v2/profile_001/trace.json`: **82,890,837 bytes**, SHA-256 `52d0cd58363f1c538468b161c5e44075066627e45ef7630b2c68f785b82b41fa`. It was streamed through a remote SHA-256 calculation only. It has not been copied or independently hashed locally. The remote size and modification timestamp were unchanged across hashing.

Run `python3 verify.py` and `shasum -a 256 -c SHA256SUMS` from this directory to check the retrieved originals and this capture manifest. Existing output, source, runtime manifests and Spark processes were not altered. No GPU task was launched.
