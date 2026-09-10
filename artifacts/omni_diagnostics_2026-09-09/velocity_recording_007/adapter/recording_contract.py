"""CPU-only contracts for recording immutable candidate003 checkpoints."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

SOURCE_SHA256 = 'fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e'
SOURCE_MANIFEST_SHA256 = '00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25'
PROBE_CAMPAIGN_SHA256 = 'ca436a5439526cfd9b742143650fa1a6b9f132e812fa02f38415bf0324327155'
PLAN_SHA256 = '356b72f0e82b207ba6105cb42057643d21a5b911d0517661465dcfe57745d00f'
URDF_SHA256 = 'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c'
IDENTITY = dict(variant='f050_t060', urdf_sha256=URDF_SHA256,
                plan_sha256=PLAN_SHA256, stance_index=0, source_sha256=SOURCE_SHA256)
DT = .02
FPS = 25
SEQUENCE = (
    ('QUIET START', 3., (0., 0., 0.)),
    ('FORWARD', 3., (.10, 0., 0.)),
    ('STRAFE LEFT', 3., (0., .10, 0.)),
    ('REVERSE', 3., (-.10, 0., 0.)),
    ('STRAFE RIGHT', 3., (0., -.10, 0.)),
    ('TURN LEFT', 3., (0., 0., .20)),
    ('TURN RIGHT', 3., (0., 0., -.20)),
    ('FORWARD LEFT ARC', 4., (.10, 0., .15)),
    ('STRAFE RIGHT ARC', 4., (0., -.10, -.15)),
    ('QUIET END', 5., (0., 0., 0.)),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_digest(root):
    """Exactly the frozen candidate_runner source_digest algorithm, CPU-only."""
    root = Path(root)
    paths = set()
    for folder in ('tools', 'isaaclab', 'packages', 'experiments/c_length_study/runtime'):
        paths.update(p for p in (root / folder).rglob('*.py') if '__pycache__' not in p.parts)
    paths.update((root / 'experiments/c_length_study/runtime').glob('*.json'))
    if not paths:
        raise ValueError('No frozen candidate source found')
    encoded = {str(p.relative_to(root)): digest(p) for p in sorted(paths)}
    return hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_source(root):
    root = Path(root).resolve()
    if source_digest(root) != SOURCE_SHA256:
        raise ValueError('Recorder requires the byte-exact candidate003 source; no identity substitution')
    if digest(root / 'campaign_source_hashes.json') != SOURCE_MANIFEST_SHA256:
        raise ValueError('Recorder requires the exact root-dispatched source003 campaign manifest')
    return root


def verify_checkpoint_bytes(path, expected_sha256):
    path = Path(path)
    if len(expected_sha256) != 64 or digest(path) != expected_sha256:
        raise ValueError('Explicit recording checkpoint SHA-256 mismatch')
    meta = json.loads(path.with_suffix(path.suffix + '.json').read_text())
    if meta.get('checkpoint_sha256') != expected_sha256:
        raise ValueError('Checkpoint sidecar byte identity mismatch')
    if any(meta.get(key) != value for key, value in IDENTITY.items()):
        raise ValueError('Checkpoint was not produced by this exact candidate003 source/plan/asset')
    # Full controller, embedded contract and tensor checks remain in the actual
    # frozen candidate_runner. This helper never changes its loading rules.
    return meta


def verify_pilot_prerequisites(calibration_path, smoke_path):
    calibration = json.loads(Path(calibration_path).read_text())
    smoke = json.loads(Path(smoke_path).read_text())
    if (calibration.get('passed') is not True or smoke.get('passed') is not True
        or calibration.get('initial_std') != .005
        or any(calibration.get(k) != v or smoke.get(k) != v for k, v in IDENTITY.items())
        or smoke.get('calibration_sha256') != digest(calibration_path)):
        raise ValueError('Recording requires matching passed calibration and runner smoke')


def verify_pilot_run(state_path, checkpoint_sha256, label):
    state = json.loads(Path(state_path).read_text())
    if (state.get('status') != 'completed' or state.get('mode') != 'train'
        or state.get('iterations') != 50
        or any(state.get(k) != v for k, v in IDENTITY.items())):
        raise ValueError('Recording labels require a completed exact-source50-update scratch pilot state')
    if label == 'scratch_50_update_final' and state.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('Final recording checkpoint is not the completed pilot checkpoint')
    return state


def tree_hashes(root):
    root = Path(root)
    paths = sorted(root.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('Admitted package must not contain symlink substitutions')
    return {str(path.relative_to(root)): digest(path) for path in paths if path.is_file()}


def verify_admitted_package(args):
    """Use the exact post-validation package, separate from executable source.

    The host additionally mounts probe, pilot, recorder and source read-only.
    This independently verifies every file, including USD payloads and meshes.
    """
    probe = args.probe_campaign.parent
    pilot = args.pilot_state.parent.parent
    if (args.package != pilot / 'inputs/study'
        or args.study_tree_receipt != probe / 'inputs/study_after_flat.sha256.json'
        or args.pilot_inputs_receipt != pilot / 'inputs_before.sha256.json'):
        raise ValueError('Use exact pilot admitted package and named probe/pilot tree receipts')
    if digest(args.probe_campaign) != PROBE_CAMPAIGN_SHA256:
        raise ValueError('Accepted probe campaign receipt mismatch')
    campaign = json.loads(args.probe_campaign.read_text())
    if (campaign.get('status') != 'completed' or campaign.get('source_unchanged') is not True
        or campaign.get('admitted_asset_unchanged') is not True
        or campaign.get('source_manifest_sha256') != SOURCE_MANIFEST_SHA256):
        raise ValueError('Probe receipt does not bind the exact admitted source/asset')
    for field, argument, relative in (
        ('flat_admission_sha256', 'admission', 'flat/admission.json'),
        ('calibration_sha256', 'calibration', 'probe/calibration.json'),
        ('runner_smoke_sha256', 'runner_smoke', 'probe/runner_smoke.json')):
        path = getattr(args, argument)
        if path != probe / relative or campaign.get(field) != digest(path):
            raise ValueError('Probe evidence does not match accepted campaign: ' + argument)
    expected = json.loads(args.study_tree_receipt.read_text())
    if len(expected) != 550 or tree_hashes(args.package) != expected or tree_hashes(probe / 'inputs/study') != expected:
        raise ValueError('Complete550-file post-validation study tree mismatch')
    inputs_expected = json.loads(args.pilot_inputs_receipt.read_text())
    if tree_hashes(pilot / 'inputs') != inputs_expected:
        raise ValueError('Complete pilot inputs tree mismatch')
    study_subset = {key[len('study/'):]: value for key, value in inputs_expected.items() if key.startswith('study/')}
    if study_subset != expected:
        raise ValueError('Pilot input receipt does not contain the identical admitted study tree')
    for name, original in (('admission.json', args.admission), ('calibration.json', args.calibration),
                           ('runner_smoke.json', args.runner_smoke)):
        if inputs_expected.get(name) != digest(original):
            raise ValueError('Pilot copied evidence differs from accepted probe: ' + name)
    pilot_campaign = json.loads((pilot / 'campaign.json').read_text())
    if (pilot_campaign.get('status') != 'completed_needs_review' or pilot_campaign.get('identity') != IDENTITY
        or pilot_campaign.get('accepted_probe_campaign_sha256') != PROBE_CAMPAIGN_SHA256
        or pilot_campaign.get('source_unchanged') is not True or pilot_campaign.get('admitted_inputs_unchanged') is not True
        or pilot_campaign.get('comparison_sha256') != digest(pilot / 'comparison.json')):
        raise ValueError('Completed pilot does not bind this admitted probe/source/comparison')
    return dict(package_path=str(args.package), study_file_count=len(expected),
                probe_campaign_sha256=digest(args.probe_campaign),
                study_tree_receipt_sha256=digest(args.study_tree_receipt),
                pilot_inputs_receipt_sha256=digest(args.pilot_inputs_receipt),
                pilot_campaign_sha256=digest(pilot / 'campaign.json'),
                complete_admitted_package_tree_verified=True,
                package_separate_from_frozen_executable_source=True)


def schedule():
    rows = []
    start = 0
    for index, (name, seconds, command) in enumerate(SEQUENCE):
        steps = round(seconds / DT)
        for step in range(steps):
            rows.append(dict(segment_index=index, segment=name,
                             time_start_s=(start + step) * DT,
                             requested_command=list(command)))
        start += steps
    return rows


def xyzw_matrix(quaternion, *, convention):
    """Explicit installed Isaac Lab XYZW convention; no guessed conversion."""
    if convention != 'xyzw':
        raise ValueError('Explicit raw Isaac Lab xyzw quaternion convention required')
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1.) > 1e-4:
        raise ValueError('Finite unit XYZW quaternion required')
    x, y, z, w = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def actual_pose_xy_heading(position, quaternion, *, convention):
    position = np.asarray(position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Finite world body position required')
    forward = xyzw_matrix(quaternion, convention=convention) @ np.array([0., -1., 0.])
    if np.linalg.norm(forward[:2]) < .1:
        raise ValueError('Body heading is undefined near vertical forward axis')
    return np.r_[position[:2], math.atan2(forward[1], forward[0])]


def assert_event_snapshot(snapshot, terminated, truncated):
    for key, actual in [('terminated', terminated), ('truncated', truncated)]:
        if not np.array_equal(snapshot[key], actual):
            raise ValueError('Recording pre-reset event mismatch: ' + key)
    raw = np.asarray(snapshot['quaternion_world_xyzw'])
    converted = np.asarray(snapshot['quaternion_world_wxyz'])
    if raw.shape != (1, 4) or not np.array_equal(converted, raw[..., [3, 0, 1, 2]]):
        raise ValueError('Raw XYZW and explicit WXYZ diagnostic conversion disagree')
    xyzw_matrix(raw[0], convention='xyzw')
    return bool(np.asarray(terminated).any() or np.asarray(truncated).any())


def check_observations(obs):
    for name, width in [('policy', 495), ('critic', 498)]:
        value = obs[name]
        if tuple(value.shape) != (1, width):
            raise ValueError('Wrong candidate recording observation schema')
        if not bool(value.isfinite().all()):
            raise ValueError('Nonfinite actor/critic observation')


def external_source_hashes(root):
    root = Path(root)
    return {p.name: digest(p) for p in sorted(root.glob('*.py'))}
