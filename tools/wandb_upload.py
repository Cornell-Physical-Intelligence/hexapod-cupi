"""Upload a finished locomotion run directory to Weights & Biases from the host.

The Spark image does not install wandb, so native runs keep writing metrics.jsonl,
state.json and force_metrics.json. This tool sends those files to W&B after you
retrieve the run. W&B holds curves and media; Git keeps acceptance summaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

BOUND_FILES = ('state.json', 'metrics.jsonl', 'force_metrics.json', 'ppo_config.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix=''):
    """Keep finite numeric leaves under slash-separated keys; drop text and missing values."""
    if isinstance(value, bool):
        return {prefix: int(value)} if prefix else {}
    if isinstance(value, (int, float)):
        return {prefix: value} if prefix and math.isfinite(value) else {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return {}
    result = {}
    for key, item in items:
        result.update(flatten(item, f'{prefix}/{key}' if prefix else str(key)))
    return result


def records(run):
    """Read one finished run directory without contacting W&B."""
    run = Path(run)
    state = json.loads((run/'state.json').read_text())
    if state.get('status') not in ('completed', 'failed'):
        raise ValueError('Upload a finished run: state.json status must be completed or failed')
    metrics = run/'metrics.jsonl'
    rows = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.is_file() else []
    updates = [row['update'] for row in rows]
    if any(b <= a for a, b in zip(updates, updates[1:])):
        raise ValueError('Metric rows need strictly increasing updates')
    loads = run/'force_metrics.json'
    config = {'identity': state.get('identity'), 'run_mode': state.get('mode'), 'status': state.get('status'),
              'files_sha256': {name: sha(run/name) for name in BOUND_FILES if (run/name).is_file()}}
    summary = flatten({'updates': state.get('updates'), 'transitions': state.get('transitions'),
                       'wall_seconds': state.get('wall_seconds'), 'evaluation': state.get('evaluation'),
                       'force_metrics': json.loads(loads.read_text()) if loads.is_file() else None})
    steps = [(row['update'], flatten({k: v for k, v in row.items() if k != 'update'})) for row in rows]
    videos = sorted(p for p in run.rglob('*.mp4') if 'wandb' not in p.relative_to(run).parts)
    return config, steps, summary, videos


def upload(run, project, *, mode='online', videos=False, wandb=None):
    """Create one W&B run whose files stay inside the run directory."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', project):
        raise ValueError('Invalid W&B project name')
    if mode not in ('online', 'offline'):
        raise ValueError('W&B mode must be online or offline')
    if wandb is None:
        import wandb
    run = Path(run).resolve()
    config, steps, summary, found = records(run)
    # W&B otherwise writes ./wandb into the current directory, which can be the checkout.
    active = wandb.init(project=project, name=run.name, dir=str(run), mode=mode,
                        job_type=str(config['run_mode']), config=config)
    try:
        for step, values in steps:
            active.log(values, step=step)
        if videos:
            for path in found:
                active.log({'video/'+path.relative_to(run).as_posix(): wandb.Video(str(path))})
        active.summary.update(summary)
    finally:
        active.finish()
    return {'run': str(run), 'project': project, 'mode': mode, 'steps': len(steps),
            'videos': len(found) if videos else 0, 'files_sha256': config['files_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Retrieved run directory that holds state.json')
    parser.add_argument('--project', required=True)
    parser.add_argument('--mode', choices=('online', 'offline'), default='online')
    parser.add_argument('--videos', action='store_true', help='Also upload rollout videos under the run')
    args = parser.parse_args(argv)
    print(json.dumps(upload(args.run, args.project, mode=args.mode, videos=args.videos), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
