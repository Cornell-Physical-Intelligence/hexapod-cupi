# Exact profiler v1 output capture

Original output from `/home/orionh/HEXAPOD_runs/mkii_profile_v1/profile_001` on `spark-e26c` was copied without rewriting its bytes. All 6 selected files match SHA-256 values computed on the remote host from the transmitted original bytes. `retrieval_metadata.json` records hashes, sizes, timestamps, the exact read-only retrieval script and the SSH result.

The validation report records `pass=true`. This is a profiler capture, not physical training admission or a completed PPO run. The `admitted` marker records launcher/resource admission; it is not a passing physical qualification. Interpret timing using the corresponding analysis artifact and its measurement limitations.

No trace file was present in this original output directory at retrieval. This capture does not fabricate one.

Run `python3 verify.py` and `shasum -a 256 -c SHA256SUMS` from this directory to check the retrieved originals and this capture manifest. Existing output, source, runtime manifests and Spark processes were not altered. No GPU task was launched.
