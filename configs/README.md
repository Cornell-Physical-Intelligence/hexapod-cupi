# Configuration

`locomotion_spark.json` declares the target for a portable canonical input pack.
The destination remains unpopulated until you transfer that pack. Use a fresh
name and an explicit `--inputs` file if that destination exists.
Use `locomotion.inputs pack` for a separate immutable input allocation.
`source_inventory.json` lists maintained code. `archive.json` resolves historical
references. `releases/` holds current source manifests and the frozen Stage 2
manifest needed by the explicit historical verifier.
