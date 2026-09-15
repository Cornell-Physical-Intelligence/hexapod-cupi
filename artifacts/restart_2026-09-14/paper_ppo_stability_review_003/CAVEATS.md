# Adopted scope and retained caveats

Root explicitly adopted the fresh successor in DECISION_001.json, including separate BC Adam at 3e-4 discarded before PPO, velocity coefficient1 with BC-only detach, PPO LR1e-4 and analytic targetKL0.02. This supersedes the narrower optimizer suggestion in RECOMMENDATION.md without rewriting that advice or the raw Fable response.

The crossing shared-model step remains accepted; only later model steps stop. Discriminator scheduling continues. This is not a hard KL cap or a gait/physics guarantee. Final analytic KL need not exceed an earlier mean of sampled minibatch estimates. Target/LR values are engineering proposals. Loss magnitudes alone cannot establish velocity-gradient dominance. Between-rollout normalizer drift is observed separately and never gates/freezes statistics. Existing source/checkpoints and every Stage2 gate remain unchanged. Root owns native dispatch after code review and meaningful CPU verification.
