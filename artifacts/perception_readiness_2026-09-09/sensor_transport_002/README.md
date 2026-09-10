# Causal sensor transport fixes integrated

The reviewed successor fixes delayed older packets overwriting newer captures, inference-created state failing on reset, and invalid clocks producing future or negative-age observations. Per-row valid capture order, equal-time sequence ties and reset epochs are explicit. Defaults, noise, latency, dropout and the250ms map lease are preserved.

The [preparation](preparation/README.md) retains failing-before evidence and15 CPU tests. The independent review adds four tests and seeded queue checks. The [integration receipt](integration/INTEGRATION_RESULT.json) binds the adopted runtime, portable tests and original-source fixture;21 focused tests pass, including401 exact nominal reads and six existing sensor contracts. Source and all original evidence freezes remain unchanged inside this wrapper.

This is software preparation for Stage3. It is not a terrain traversal result, actor integration, physical sensor measurement or qualification. The current Spark walking experiment uses its separately frozen source. Current execution belongs in STATUS.md; the complete robot still requires terrain and sensing qualification.
