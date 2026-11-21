"""Main's walking-requirements judge on the surrogate, with the five leg-stand fitted axes drawn.

Same draws as records/distilled_bm0_u1000_20261112/scripts/fitted_eval.py (its FIT table and
--motor-check drawn), ported to the merged code: the fitted ranges go to SurrogateEnv(dynamics=...)
and are drawn per episode, i.e. per judge case; the touchdown model is off; the capture's
motor-law check uses each episode's drawn damping. The judge, its cases and every bound are
main's, unchanged. The contact consensus is skipped, because it varies friction, which the
draws set.

Usage (from the hexapod-cupi checkout, merged code):
  .venv/bin/python fitted_judge.py --checkpoint CKPT --output DIR --cases all --video none --seed N
Wrap with records/distilled_bm0_u1000_20261112/scripts/body_mass.py run ... for bm0's own body.
"""

import json
import sys
from dataclasses import asdict, replace

import locomotion.surrogate.capture as surrogate_capture
import locomotion.surrogate.env as surrogate_env
from locomotion.surrogate import evaluate

FIT = {
    "action_delay_controls": (1, 2),
    "kd_scale": (0.13, 0.37),
    "friction_scale": (0.45, 0.6),
    "joint_friction_nm": ((0.0335, 0.0622), (0.1387, 0.2576), (0.018, 0.0335)),
    "armature_kg_m2": ((0.0, 0.0001), (0.0, 0.00058), (0.0, 0.00049)),
}
DRAWS = []
OriginalEnv = surrogate_env.SurrogateEnv


class FittedEnv(OriginalEnv):
    """The merged surrogate with the fitted ranges drawn per episode and the touchdown model off."""

    def __init__(self, *args, **kwargs):
        contact = kwargs.get("contact") or surrogate_env.ContactModel()
        kwargs["contact"] = replace(contact, touchdown_overshoot=False)
        kwargs["dynamics"] = surrogate_env.DynamicsRandomization(**FIT)
        super().__init__(*args, **kwargs)

    def _randomize(self, rows):
        super()._randomize(rows)
        DRAWS.append(
            {key: float(value[0]) for key, value in (self.randomization_draws or {}).items()}
        )


surrogate_env.SurrogateEnv = FittedEnv
OriginalCapture = surrogate_capture.SurrogateCaptureSource


class FittedCapture(OriginalCapture):
    """The stock capture; its motor-law oracle uses the damping this episode drew."""

    def __init__(self, env):
        super().__init__(env)
        if getattr(env, "_kd_rows", None) is not None:
            self.kd = env._kd_rows.detach().cpu().numpy().copy()
            self.metadata["motor_snapshot"]["kd_nm_s_per_rad"] = self.kd.tolist()


surrogate_capture.SurrogateCaptureSource = FittedCapture


def main(argv):
    if "--no-contact-consensus" not in argv:
        argv = [*argv, "--no-contact-consensus"]
    code = 0
    try:
        code = evaluate.main(argv)
    except KeyError as error:
        # main's evaluate.py writes surrogate_summary.json, then its closing printout reads a
        # "metrics" field the judge no longer writes; the summary is already complete.
        if error.args != ("metrics",):
            raise
    output = argv[argv.index("--output") + 1]
    with open(f"{output}/fitted_draws.json", "w") as handle:
        json.dump(
            {
                "ranges": asdict(surrogate_env.DynamicsRandomization(**FIT)),
                "touchdown_overshoot": False,
                "motor_check": "drawn kd per episode",
                "draws_per_episode": DRAWS,
            },
            handle,
            indent=1,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
