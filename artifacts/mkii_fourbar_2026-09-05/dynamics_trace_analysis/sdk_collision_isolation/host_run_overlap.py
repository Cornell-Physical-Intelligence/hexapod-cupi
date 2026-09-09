#!/usr/bin/env python3
"""Explicit, bounded host runner for the external collision-overlap fixture.

This diagnostic never admits training or hardware. Default operation prints
the planned command; only --execute starts the two separately owned cases.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import re
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
FIXTURE_MOUNT = '/workspace/overlap_fixture'
FIXTURE_SCRIPT = 'live_robot_pair.py'
CASES = ('filtered', 'unfiltered_negative')


def load_supervisor(source):
    path = source / 'isaaclab/deploy/run-mkii-fourbar'
    loader = importlib.machinery.SourceFileLoader('overlap_frozen_supervisor', str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def tree_identity(root):
    """Include all regular fixture files, including this external host driver."""
    records = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Fixture snapshots must not contain symlinks')
        if path.is_file():
            records[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return records


def compose_argv(host, source, fixture, output, name, owner, case, steps, solver_multiplier=2):
    barrier = 'while [ ! -f /workspace/validation_artifacts/admitted ]; do sleep 0.1; done; exec "$@"'
    return ['docker', 'compose', '--env-file', 'docker/.env.base', '-f', 'docker/docker-compose.yaml',
            '--profile', 'base', 'run', '--no-deps', '-d', '-T', '--name', name,
            '--label', f'{host.OWNER_LABEL}={owner}', '-w', host.CONTAINER_OUTPUT,
            '-e', f'PYTHONPATH={host.CONTAINER_SOURCE}/isaaclab', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-v', f'{source}:{host.CONTAINER_SOURCE}:ro', '-v', f'{fixture}:{FIXTURE_MOUNT}:ro',
            '-v', f'{output}:{host.CONTAINER_OUTPUT}:rw', '--entrypoint', '/bin/bash',
            'isaac-lab-base', '-c', barrier, 'hexapod-overlap-barrier',
            '/workspace/isaaclab/_isaac_sim/python.sh', f'{FIXTURE_MOUNT}/{FIXTURE_SCRIPT}',
            *shlex.split(host.TELEMETRY_ARGUMENT), '--case', case, '--physics-steps', str(steps),
            '--solver-multiplier', str(solver_multiplier), '--source-dir', host.CONTAINER_SOURCE,
            '--report', f'{host.CONTAINER_OUTPUT}/report.json', '--viz', 'none', '--device', 'cuda:0']


def require_report(path, *, case, steps, fixture_hash, contract, solver_multiplier):
    if not path.is_file() or path.is_symlink():
        raise ValueError('Fixture exited without a regular final report')
    result = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    expected = {'schema': 'hexapod.live_collision_pair_fixture.v1', 'case': case,
                'pass': False, 'fixture_complete': True, 'control_expectation_met': True,
                'simulation_training_admission': False, 'hardware_admission': False,
                'physics_steps_requested': steps, 'errors': [], 'source_identity': contract,
                'fixture_source_sha256': fixture_hash}
    if not isinstance(result, dict) or any(type(result.get(key)) is not type(value)
            or result[key] != value for key, value in expected.items()):
        raise ValueError('Incomplete, mismatched, non-diagnostic or incorrectly identified fixture report')
    paths = [f'/World/envs/env_{i}/Robot/Geometry/body' for i in range(2)]
    bindings = [{'source_body': paths[i], 'target_filter': paths[1-i]} for i in range(2)]
    if result.get('native_pair_bindings') != bindings:
        raise ValueError('Native contact views did not bind the two exact robot bodies')
    metrics = result.get('metrics', {})
    if not isinstance(metrics, dict):
        raise ValueError('Fixture metrics must be an object')
    count, force = metrics.get('max_pair_contact_count'), metrics.get('max_pair_force_n')
    if (type(metrics.get('samples')) is not int or metrics['samples'] != steps
            or metrics.get('finite') is not True or metrics.get('contact_buffer_capacity_reached') is not False
            or type(count) is not int or count < 0 or type(force) not in (int, float)
            or not math.isfinite(force) or force < 0):
        raise ValueError('Fixture metrics are invalid, incomplete or contact capacity was reached')
    if case == 'filtered':
        support = metrics.get('ground_support_observed_per_robot')
        if count != 0 or force > 1e-6 or support != [True, True] or any(type(v) is not bool for v in support):
            raise ValueError('Filtered case has cross-robot contacts or lacks independent ground support')
    elif count < 1 or force <= 1e-3:
        raise ValueError('Negative control does not prove actual cross-robot contact')
    trace = result.get('trace', {})
    if not isinstance(trace, dict):
        raise ValueError('Fixture trace metadata must be an object')
    trace_path = path.parent / 'trace.npz'
    if (trace.get('file') != 'trace.npz' or type(trace.get('samples')) is not int
            or trace['samples'] != steps or not trace_path.is_file() or trace_path.is_symlink()
            or hashlib.sha256(trace_path.read_bytes()).hexdigest() != trace.get('sha256')):
        raise ValueError('Trace bytes or sample count do not match the completed fixture')
    return {key: result[key] for key in expected}



def cleanup_owned(host, container, name, owner, output, report):
    """Never stop/remove by name; a name is used only to recover creation's ID."""
    if container is None:
        container = host.inspect_container(name)
    if container is None:
        report['cleanup'] = 'not_required'
        return
    if not host.check_identity(container, name, owner):
        raise host.Blocked('Cleanup refused a container ownership mismatch')
    identifier = container['id']
    current = host.inspect_container(identifier)
    if not host.check_identity(current, name, owner):
        raise host.Blocked('Cannot prove final owned container identity')
    if current['running']:
        host.command(['docker', 'stop', '--time', '25', identifier], timeout=30, check=False)
    with (output / 'container.log').open('w') as log:
        logged = subprocess.run(['docker', 'logs', '--timestamps', identifier], stdout=log,
                                stderr=subprocess.STDOUT, timeout=20)
    report['container_log_exit_code'] = logged.returncode
    after = host.inspect_container(identifier)
    if not host.check_identity(after, name, owner) or after['running']:
        raise host.Blocked('Owned container did not stop or changed identity')
    report['final_container_exit_code'] = after['exit_code']
    removed = host.command(['docker', 'rm', identifier], timeout=15, check=False)
    if removed.returncode or logged.returncode:
        raise host.Blocked('Exact-ID removal or mandatory log capture failed')
    report['cleanup'] = 'removed_exact_id'


def run_case(host, args, source, fixture, output, contract, fixture_identity, shared_digest, case):
    owner = uuid.uuid4().hex
    name = f'hexapod-overlap-{case}-{owner[:12]}'
    output.mkdir(parents=True, exist_ok=False)
    argv = compose_argv(host, source, fixture, output, name, owner, case, args.steps, args.solver_multiplier)
    report = {'schema': 'hexapod.collision_overlap_supervisor.v1', 'case': case,
              'pass': False, 'simulation_training_admission': False, 'hardware_admission': False,
              'execution': 'preflight', 'source_commit': args.source_commit, 'contract': contract,
              'fixture_files_sha256': fixture_identity, 'coordination_sha256': shared_digest,
              'container_name': name, 'owner': owner, 'gates': [], 'cleanup': 'not_required',
              'timeout_seconds': args.timeout_seconds, 'argv': argv, 'observed_owned_gpu_pids': []}
    host.atomic_json(output / 'argv.json', argv)
    host.atomic_json(output / 'supervisor.json', report)
    container, success = None, False
    deadline = time.monotonic() + args.timeout_seconds

    def unchanged():
        host.require_unchanged_source(source, contract, 'during external overlap diagnostic')
        if tree_identity(fixture) != fixture_identity:
            raise host.Blocked('External fixture bytes changed during the diagnostic')
        if host.coordination_snapshot() != shared_digest:
            raise host.Blocked('Coordination changed; owned overlap diagnostic must yield')

    try:
        unchanged()
        report['gates'].append(host.resource_gate())
        created = host.command(argv, cwd=host.LAB, timeout=60, check=False)
        container = host.inspect_container(name)
        if not host.check_identity(container, name, owner):
            raise host.Blocked('Cannot establish new container ownership')
        report['container_id'] = container['id']
        if created.returncode:
            raise host.Blocked(f'Compose creation failed (exit {created.returncode})')
        report['gates'].append(host.resource_gate(owned_container=container))
        time.sleep(1)
        report['gates'].append(host.resource_gate(owned_container=container))
        unchanged()
        if time.monotonic() >= deadline:
            raise host.Blocked('Overlap deadline expired before admission')
        (output / 'admitted').touch(exist_ok=False)
        report['execution'] = 'running'
        host.atomic_json(output / 'supervisor.json', report)
        print(f'OVERLAP_STARTED case={case} container_id={container["id"]} output={output}', flush=True)
        next_gate = time.monotonic()
        while True:
            current = host.inspect_container(container['id'])
            if not host.check_identity(current, name, owner):
                raise host.Blocked('Owned container identity disappeared or changed')
            if not current['running']:
                report['exit_code'] = current['exit_code']
                if current['exit_code'] != 0:
                    raise host.Blocked(f'Fixture exited with status {current["exit_code"]}')
                success = True
                break
            if time.monotonic() >= deadline:
                raise host.Blocked(f'Overlap exceeded {args.timeout_seconds} second timeout')
            if time.monotonic() >= next_gate:
                unchanged()
                report['last_runtime_gate'] = host.resource_gate(owned_container=current, allow_owned_gpu=True)
                report['observed_owned_gpu_pids'] = sorted(set(report['observed_owned_gpu_pids']) |
                                                           set(report['last_runtime_gate']['gpu_pids']))
                host.atomic_json(output / 'supervisor.json', report)
                next_gate = time.monotonic() + 5
            time.sleep(1)
    except (host.Blocked, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        report['error'] = str(exc)
    finally:
        old_handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            cleanup_owned(host, container, name, owner, output, report)
        except (host.Blocked, OSError, ValueError, subprocess.TimeoutExpired) as exc:
            report['cleanup'] = 'FAILED'; report['cleanup_error'] = str(exc); success = False
        try:
            unchanged()
            report['source_identity_unchanged_at_finish'] = True
            report['fixture_identity_unchanged_at_finish'] = True
            if success:
                if not report['observed_owned_gpu_pids']:
                    raise ValueError('No live GPU process was observed with verified container ownership')
                report['fixture_summary'] = require_report(output / 'report.json', case=case,
                    steps=args.steps, fixture_hash=fixture_identity[FIXTURE_SCRIPT],
                    contract=contract, solver_multiplier=args.solver_multiplier)
                host.validate_numerical_recipe_report(json.loads((output / 'report.json').read_text()), args.solver_multiplier)
                host.require_requested_asset(json.loads((output / 'report.json').read_text()), 'mkii_fourbar_v5', contract)
        except (host.Blocked, OSError, ValueError) as exc:
            report['report_error'] = str(exc); success = False
        report['execution'] = 'diagnostic_complete' if success else 'failed'
        report['supervisor_exit_code'] = 0 if success else 1
        host.atomic_json(output / 'supervisor.json', report)
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
    print(f'OVERLAP_FINISHED case={case} status={report["execution"]} output={output}', flush=True)
    return success


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--fixture-dir', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--case', choices=(*CASES, 'all'), default='all')
    parser.add_argument('--physics-steps', dest='steps', type=int, default=256)
    parser.add_argument('--solver-multiplier', type=int, choices=(1, 2), default=2)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not 30 <= args.timeout_seconds <= 600 or not 32 <= args.steps <= 512 or args.steps % 16:
        parser.error('Cases require 30..600 seconds and 32..512 physics steps divisible by16')
    if not re.fullmatch(r'[0-9a-f]{7,40}', args.source_commit):
        parser.error('--source-commit must identify the frozen source commit')
    source, fixture, output_root = (path.expanduser().resolve() for path in
                                   (args.source_dir, args.fixture_dir, args.output_root))
    if not (source / 'isaaclab/deploy/run-mkii-fourbar').is_file() or not (fixture / FIXTURE_SCRIPT).is_file():
        parser.error('Frozen source supervisor and external live_robot_pair.py are required')
    if fixture.is_relative_to(source) or source.is_relative_to(fixture):
        parser.error('Fixture and frozen production snapshots must be separate directories')
    if any(output_root.is_relative_to(path) for path in (source, fixture)):
        parser.error('Outputs must be outside both read-only input snapshots')
    if any(any(c in str(path) for c in ('\n', '\r', ':')) for path in (source, fixture, output_root)):
        parser.error('Bind paths cannot contain newlines or colons')
    host = load_supervisor(source)
    contract, fixture_identity = host.identity(source), tree_identity(fixture)
    if Path(__file__).resolve().parent != fixture:
        parser.error('Run the host driver stored in the exact external fixture snapshot')
    cases = CASES if args.case == 'all' else (args.case,)
    run = output_root / ('overlap-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8])
    if not args.execute:
        print(json.dumps({'execution': 'not_started', 'output': str(run), 'contract': contract,
              'fixture_files_sha256': fixture_identity, 'cases': {case: compose_argv(host, source,
              fixture, run / case, f'hexapod-overlap-{case}-DRYRUN', 'DRYRUN', case, args.steps, args.solver_multiplier)
              for case in cases}, 'simulation_training_admission': False, 'hardware_admission': False}, indent=2))
        return 0
    for relative in ('docker/.env.base', 'docker/docker-compose.yaml'):
        if not (host.LAB / relative).is_file():
            raise host.Blocked(f'Required Compose file missing: {relative}')
    lock = os.open(host.LOCK, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    if lock != 9:
        os.dup2(lock, 9); os.close(lock)
    try:
        fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(9); raise host.Blocked('Another hexapod supervisor owns the GPU lock')
    def interrupted(signum, _frame):
        raise host.Blocked(f'Overlap supervisor interrupted by signal {signum}')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        shared_digest = host.coordination_snapshot()
        run.mkdir(parents=True, exist_ok=False)
        archived = {}
        for label, root in (('source', source), ('fixture', fixture)):
            manifest = host.snapshot_source(root, run / f'{label}.SHA256SUMS')
            archived[label] = host.archive_source(root, run / f'{label}.SHA256SUMS', run / f'{label}.tar.gz',
                                                  expected_manifest_sha256=manifest['sha256'])
        host.atomic_json(run / 'inputs.json', {'contract': contract, 'fixture_files_sha256': fixture_identity,
                         'source_commit': args.source_commit, 'archives': archived})
        for case in cases:
            if not run_case(host, args, source, fixture, run / case, contract, fixture_identity, shared_digest, case):
                return 1
        if len(cases) == 2:
            compare_argv = [sys.executable, str(fixture / FIXTURE_SCRIPT), '--source-dir', str(source),
                            '--compare', str(run / cases[0] / 'report.json'), str(run / cases[1] / 'report.json'),
                            '--physics-steps', str(args.steps), '--report', str(run / 'pair_report.json')]
            host.atomic_json(run / 'compare_argv.json', compare_argv)
            compared = host.command(compare_argv, timeout=30, check=False)
            (run / 'compare.log').write_text(compared.stdout + compared.stderr)
            if compared.returncode or not (run / 'pair_report.json').is_file():
                raise host.Blocked('CPU comparison failed or omitted its durable pair report')
            pair = json.loads((run / 'pair_report.json').read_text(),
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if (not isinstance(pair, dict) or pair.get('pair_control_success') is not True or pair.get('errors') != []
                    or any(pair.get(key) is not False for key in
                           ('pass', 'simulation_training_admission', 'hardware_admission'))):
                raise host.Blocked('The positive/negative pair did not prove the expected collision response')
        host.require_unchanged_source(source, contract, 'after the completed overlap pair')
        if tree_identity(fixture) != fixture_identity or host.coordination_snapshot() != shared_digest:
            raise host.Blocked('External fixture or coordination changed during comparison')
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        fcntl.flock(9, fcntl.LOCK_UN); os.close(9)


if __name__ == '__main__':
    raise SystemExit(main())
