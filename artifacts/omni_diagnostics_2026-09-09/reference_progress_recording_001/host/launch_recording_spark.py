"""Bounded native rendering of the admitted009 reference; no PPO or pose forcing."""
from pathlib import Path
import argparse
import hashlib
import json
import signal
import sys
import time

sys.dont_write_bytecode = True
SOURCE_MAP = '04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
CAMPAIGN_SHA = 'cdcaa80c1156509d60e336042172dc6651755767e08ecbb66ef31d8e788bcb26'
ADAPTER_FREEZE = '2913d9c0f38f850e408a721bc98f73330ef11a7243c157b0528435196f120e99'
RUNTIME_TREE = 'abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def tree(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_inputs(args):
    if sha(args.source / 'campaign_source_hashes.json') != SOURCE_MAP:
        raise ValueError('Exact reviewed009 source required')
    if sha(args.run / 'campaign.json') != CAMPAIGN_SHA:
        raise ValueError('Exact completed009 campaign required')
    c = read(args.run / 'campaign.json')
    if (c.get('status') != 'completed' or c.get('bounded_wave_physics_passed') is not True
            or c.get('source_manifest_sha256') != SOURCE_MAP
            or c.get('source_unchanged') is not True or c.get('admitted_asset_unchanged') is not True
            or c.get('stage2_complete') is not False or c.get('policy_training_started') is not False):
        raise ValueError('Completed reference evidence required; no policy qualification inferred')
    for key, path in [('standing_admission_sha256', 'standing/admission.json'),
                      ('wave_state_sha256', 'wave/state.json')]:
        if c.get(key) != sha(args.run / path):
            raise ValueError('Qualified phase bytes changed: ' + path)
    standing = read(args.run / 'standing/admission.json')
    wave = read(args.run / 'wave/state.json')
    if (standing.get('identity') != wave.get('identity') or standing.get('identity') != c.get('identity')
            or standing.get('gate', {}).get('passed') is not True
            or standing.get('gate', {}).get('all_replica_quiet', {}).get('passed') is not True
            or standing.get('control_steps') != 1000 or standing.get('gate', {}).get('num_envs') != 32
            or wave.get('gate', {}).get('passed') is not True or wave.get('control_steps') != 2400):
        raise ValueError('Exact standing, quiet and completed wave admission required')
    asset_map = read(args.run / 'inputs/study_before.sha256.json')
    if len(asset_map) != 550 or tree(args.run / 'inputs/study') != asset_map:
        raise ValueError('The550-file admitted study package changed')
    if sha(args.adapter / 'FREEZE_SHA256.json') != ADAPTER_FREEZE:
        raise ValueError('Exact independently reviewed recording adapter required')
    manifest = read(args.adapter / 'FREEZE_SHA256.json')
    manifest = manifest.get('files', manifest)
    actual = tree(args.adapter)
    actual.pop('FREEZE_SHA256.json')
    if manifest != actual:
        raise ValueError('Recorder files changed or unlisted files were added')
    return standing['identity']


def command(args, name):
    return ['docker', 'compose', '--env-file', 'docker/.env.base', '-f', 'docker/docker-compose.yaml',
        '--profile', 'base', 'run', '--rm', '--no-deps', '--name', name, '-w', '/outputs',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONUNBUFFERED=1',
        '-e', 'PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab',
        '-v', f'{args.source}:/workspace/hexapod:ro', '-v', f'{args.output}:/outputs:rw',
        '-v', f'{args.run}:/qualified:ro', '-v', f'{args.adapter}:/recording:ro',
        '--entrypoint', '/workspace/isaaclab/_isaac_sim/python.sh', 'isaac-lab-base',
        '/recording/record_reference_video.py', '--source-root', '/workspace/hexapod',
        '--package', '/qualified/inputs/study', '--campaign', '/qualified/campaign.json',
        '--admission', '/qualified/standing/admission.json',
        '--study-tree-receipt', '/qualified/inputs/study_before.sha256.json',
        '--geometry-reference', '/workspace/hexapod/robot/hexapod_mkii_length_study/candidate_c_reference.json',
        '--output', '/outputs/recording', '--headless', '--enable_cameras', '--device', 'cuda:0', '--info',
        '--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']


def validate_result(output, identity):
    s = read(output / 'state.json')
    v = read(output / 'video.json')
    if (s.get('status') != 'completed' or s.get('identity') != identity
            or s.get('gate', {}).get('passed') is not True or s.get('control_steps') != 2400
            or s.get('runtime_binding', {}).get('runtime_tree_sha256') != RUNTIME_TREE):
        raise ValueError('Fresh recording failed the unchanged physical source contract')
    expected = dict(complete=True, frames=1200, fps=25, recorded_control_steps=2400,
                    planned_control_steps=2400, source_manifest_sha256=SOURCE_MAP,
                    source_and_inputs_reverified_after_recording=True,
                    stage2_complete=False, policy_training_started=False, pose_forcing=False)
    if any(v.get(k) != value for k, value in expected.items()):
        raise ValueError('Incomplete or mismatched fresh reference recording')
    if not (output / 'rollout.mp4').is_file() or v.get('video_sha256') != sha(output / 'rollout.mp4'):
        raise ValueError('Video bytes do not match the receipt')
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'run', 'adapter', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--isaaclab', type=Path, default=Path('/home/orionh/IsaacLab'))
    args = parser.parse_args()
    for name in ('source', 'run', 'adapter', 'output', 'isaaclab'):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() or any(args.output == p or p in args.output.parents
                                  for p in (args.source, args.run, args.adapter)):
        raise ValueError('Fresh output outside every frozen input is required')
    sys.path.insert(0, str(args.source / 'tools'))
    import launch_reference_physics_spark as host
    if Path(host.__file__).resolve() != args.source / 'tools/launch_reference_physics_spark.py':
        raise RuntimeError('Supervisor imported from the wrong source')
    runtime = host.check_source(args.source)
    identity = validate_inputs(args)
    args.coordination_sha256 = host.digest(host.COORDINATION)
    args.output.mkdir(parents=True)
    for name in ('jobs', 'logs'):
        (args.output / name).mkdir()
    signal.signal(signal.SIGTERM, lambda *_: (args.output / 'stop.request').touch())
    # Reuse the exact reviewed supervisor/cleanup body. Only the container
    # command is supplied by this separately frozen rendering-only adapter.
    host.command = lambda source, output, name, phase: command(args, name)
    report = dict(status='preparing', source_manifest_sha256=SOURCE_MAP, runtime_binding=runtime,
                  recording_adapter_freeze_sha256=ADAPTER_FREEZE, admitted_identity=identity,
                  started_unix=time.time(), stage2_complete=False, policy_training_started=False,
                  scope='fresh native reference physics recording, not the earlier009 raw trajectory')
    try:
        save(args.output / 'campaign.json', report)
        host.run_owned(args, 'recording')
        video = validate_result(args.output / 'recording', identity)
        report.update(status='completed', video_sha256=video['video_sha256'])
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        try:
            host.check_source(args.source)
            validate_inputs(args)
            report['source_and_inputs_unchanged'] = True
        except Exception as integrity_error:
            report.update(status='failed', source_and_inputs_unchanged=False,
                          integrity_error=repr(integrity_error))
            raise
        finally:
            report['finished_unix'] = time.time()
            save(args.output / 'campaign.json', report)


if __name__ == '__main__':
    main()
