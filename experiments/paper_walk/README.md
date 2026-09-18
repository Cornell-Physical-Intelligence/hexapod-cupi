# Retired paper-walk implementation

You run the maintained native environment, stock PPO and evaluation through
[`locomotion/`](../../locomotion/README.md). The cleanup removes the custom PPO/AMP
learner, BC refit/startup tools and the old launcher from active source.

You can retrieve the original [source and documentation](https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/tree/33ec6f16d70c8b0e74a9608d69be7c563c11bfbb/experiments/paper_walk)
from the pinned Git revision. The [inventory](../../configs/source_inventory.json)
records hashes for each retired source and test. Saved result directories,
checkpoints and frozen source packs retain their bytes and original conclusions.
Use those original packs for historical reproduction; the cleanup transfers no
admission or checkpoint identity.
