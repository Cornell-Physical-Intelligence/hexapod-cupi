"""Run the standing diagnostic or the noise probe on the admitted canonical simulator."""
from __future__ import annotations

import argparse
import time

from locomotion.scripts.cli import (add_app_arguments, add_common_arguments, add_learner_arguments, print_progress,
    resolve_options, run, save)


def recorded_mode(args):
    """Return the mode that state.json records: probe for the noise probe, diagnostic for the standing run."""
    return 'probe' if args.noise_probe else 'diagnostic'


def parse_args(argv=None):
    """Parse the command line and resolve the learner defaults of the selected run."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--noise-probe', action='store_true',
                        help='Hold fixed action means under graded exploration noise. Without this flag the script '
                             'holds a zero command and scores the standing robot.')
    add_common_arguments(parser)
    add_learner_arguments(parser)
    add_app_arguments(parser, argv)
    args = parser.parse_args(argv)
    return resolve_options(args, recorded_mode(args))


def run_diagnostic(env, session):
    """Hold a zero command for 1000 controls, capture the standing robot and score it."""
    import torch
    from locomotion.evaluation.capture import DiagnosticCapture, score_diagnostic
    args = session.args
    env.commands.zero_()
    capture = DiagnosticCapture(env, args.output/'standing', args.geometry_extrema)
    try:
        for control in range(1000):
            env.step(torch.zeros((env.num_envs, 18), device=env.device))
            if (control+1) % 100 == 0:
                print(f"STANDING controls={control+1} replicas={env.num_envs}", flush=True)
            if time.monotonic()-session.started >= args.max_wall_seconds:
                raise TimeoutError('Standing allocation exceeded its deadline')
    finally:
        capture.close()
    report = score_diagnostic(args.output/'standing')
    save(args.output/'standing'/'standing_report.json', report)
    session.state['standing_gate_pass'] = report['all_pass']


def run_probe(env, session):
    """Run the noise probe inside the remaining allocation and record its summary without the cells."""
    from locomotion.evaluation import noise_probe
    args = session.args
    summary = noise_probe.run(env, args.output/'probe', seed=args.seed,
        max_wall_seconds=max(.001, args.max_wall_seconds-(time.monotonic()-session.started)),
        progress=print_progress)
    session.state['probe'] = {key: value for key, value in summary.items() if key != 'cells'}


def main(argv=None):
    args = parse_args(argv)
    return run(args, recorded_mode(args), run_probe if args.noise_probe else run_diagnostic)


if __name__ == '__main__':
    raise SystemExit(main())
