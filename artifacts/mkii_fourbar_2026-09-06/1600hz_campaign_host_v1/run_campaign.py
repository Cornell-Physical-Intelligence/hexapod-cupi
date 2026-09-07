#!/usr/bin/env python3
"""Pinned physical campaign with a shared flock owned by each actual job only."""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

SOURCE_COMMIT = 'c2af43ca0f384a4c2c7ab8f1d627f309dc78a683'
SOURCE_SHA256 = 'c33b4591e99f287579181469dcc132f659d1980200708475442dbe3ffcb7ecfc'
ENTRYPOINT_SHA256 = {
    'run-mkii-fourbar': '60dc7803aa962c2a1152fa00bb20ccafe5348890509a903d65bfb850fb42bfbc',
    'run-mkii-fourbar-campaign': '4e3e5f3407f780c5d28df35e72ab141b1b745f367b9e1d53f761f9952666ed44',
}
FLOCK_PREFIX = ['/usr/bin/flock', '--nonblock', '--no-fork', '/opt/wx/gpu.lock']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_campaign(source, commit):
    if commit != SOURCE_COMMIT:
        raise ValueError('Wrapper requires the exact pinned source commit')
    for name, expected in ENTRYPOINT_SHA256.items():
        if digest(source/'isaaclab/deploy'/name) != expected:
            raise ValueError('Frozen entrypoint bytes differ: ' + name)
    path = source/'isaaclab/deploy/run-mkii-fourbar-campaign'
    loader = importlib.machinery.SourceFileLoader('_pinned_1600hz_campaign', str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    campaign = importlib.util.module_from_spec(spec)
    loader.exec_module(campaign)
    contract = campaign.identity(source)
    if contract['sha256'] != SOURCE_SHA256:
        raise ValueError('Frozen source functional identity differs')
    return campaign, contract


def install_per_job_flock(campaign):
    original = campaign.phase_argv
    @functools.wraps(original)
    def phase_argv(*args, **kwargs):
        return [*FLOCK_PREFIX, *original(*args, **kwargs)]
    campaign.phase_argv = phase_argv
    return lambda: setattr(campaign, 'phase_argv', original)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args, _ = parser.parse_known_args(argv)
    source, output = args.source_dir.resolve(strict=True), args.output_root.resolve()
    if output.is_relative_to(source):
        raise ValueError('Campaign output must be outside the immutable source')
    campaign, contract = load_campaign(source, args.source_commit)
    binding = {'schema': 'hexapod.per_job_campaign_host.v1', 'source': str(source),
        'source_commit': SOURCE_COMMIT, 'source_sha256': contract['sha256'],
        'entrypoint_sha256': ENTRYPOINT_SHA256, 'wrapper_sha256': digest(__file__),
        'phase_argv_prefix': FLOCK_PREFIX, 'persistent_reservation': False,
        'output_root': str(output)}
    print('FOURBAR_CAMPAIGN_HOST_BINDING ' + json.dumps(binding, sort_keys=True), flush=True)
    if args.dry_run:
        print('FOURBAR_CAMPAIGN_HOST_DRY_RUN_PASS', flush=True)
        return 0
    restore = install_per_job_flock(campaign)
    try:
        # Frozen parsing, resource gates, timeout bounds, phase order, admission,
        # checkpoint verification, signals and exact-container cleanup remain.
        return campaign.main(argv)
    finally:
        restore()


if __name__ == '__main__':
    raise SystemExit(main())
