"""Surrogate run of the evaluator's 96 required cases, without changing the repository.

Adds each case of locomotion.evaluation.runner.case_manifest() as a learning probe ("learning:" + its
id with the colon as "__"), keeping its profile, command, length and seed, gives every case the native
evaluator's 90 s episode timeout, then calls locomotion.surrogate.evaluate.main. The surrogate is design evidence; native runs are acceptance evidence.
Usage: uv run python full96_suite.py --checkpoint ... --output ... [evaluate options]
"""

import sys

import locomotion.evaluation.runner as runner
import locomotion.surrogate.env as surrogate_env
from locomotion.surrogate import evaluate

original = runner.learning_probe_cases


def required():
    return [{**c, "case_id": "learning:" + c["case_id"].replace(":", "__")} for c in runner.case_manifest()]


def cases():
    return original() + required()


runner.learning_probe_cases = cases
# The surrogate builds a 60 s episode for every case (evaluate.main imports SurrogateConfig when it
# runs); the 70 s transition program needs the native evaluator's 90 s timeout (EvaluationEnvConfig),
# which leaves the shorter cases unchanged.
surrogate_config = surrogate_env.SurrogateConfig


def evaluation_config(*args, **kwargs):
    return surrogate_config(*args, **{**kwargs, "episode_seconds": 90.0})


surrogate_env.SurrogateConfig = evaluation_config

if __name__ == "__main__":
    if "--cases" not in sys.argv:
        sys.argv += ["--cases", ",".join(c["case_id"] for c in required())]
    evaluate.main(sys.argv[1:])
