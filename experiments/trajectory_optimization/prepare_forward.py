"""Freeze the forward-example pilot and its matched Spark allocations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .forward_experiment import ARMS, COMMAND, SCHEMA, load_protocol, sha
from locomotion.prepare import ROOT, prepare, save

REFERENCE = Path('artifacts/ppo_reference_comparison_20260917/replay_pack_001/replay_001/standing/evaluation')
REFERENCE_REPORT_SHA = '128f43977987afb35139b319ed61052e047f6470576930a03de88c7be94279e4'


def create_protocol(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    reference = root/REFERENCE
    if sha(reference/'report.json') != REFERENCE_REPORT_SHA:
        raise ValueError('Use the recorded passing forward reference')
    report = json.loads((reference/'report.json').read_text())
    force = json.loads((reference/'force_metrics.json').read_text())
    if (not report['acquisition_complete'] or len(report['results']) != 1
            or not report['results'][0]['pass'] or force['status'] != 'available'
            or sha(reference/'control_trace.npz') != report['files']['control_trace.npz']):
        raise ValueError('Reference capture, behavior or load evidence differs')
    with np.load(reference/'control_trace.npz', allow_pickle=False) as trace:
        if any(np.any(trace[key]) for key in ('reset', 'terminated', 'truncated')):
            raise ValueError('Reference has a reset or terminal transition')
        arrays = {name: trace[source][:, 0].copy() for name, source in (
            ('policy', 'policy_observation'), ('critic', 'critic_observation'), ('action', 'policy_action'))}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'demonstration.npz', **arrays)
    value = {'schema': SCHEMA, 'question': 'Does action imitation from the passing forward example help PPO learn forward walking?',
        'arms': list(ARMS), 'seed': 20260914, 'command': list(COMMAND),
        'model_sha256': report['model_sha256'], 'urdf_sha256': report['urdf_sha256'],
        'num_envs': 128, 'controls_per_update_per_replica': 24, 'updates': 1200,
        'transitions_per_arm': 3686400, 'evaluation_updates': [0, 300, 600, 1200],
        'limiter_rad_per_20ms': .04, 'bc': {'steps': 1000, 'batch_size': 256,
            'learning_rate': .001, 'train_rows': [0, 700], 'validation_rows': [700, 1000]},
        'demonstration_sha256': sha(output/'demonstration.npz'),
        'reference_inputs': {str(REFERENCE/name): sha(reference/name) for name in
            ('report.json', 'control_trace.npz', 'force_metrics.json')},
        'treatment': 'Fit actor MLP action predictions before PPO in the example arm. Keep the critic and action variance fixed during this fit. Discard the imitation optimizer.',
        'shared': 'Use the same network seed, admitted physics, fixed forward command, task reward, PPO settings and transition budget. Initialize both input normalizers from the first 700 example rows. Restore PPO random state after imitation. Normalization resumes online during PPO.',
        'primary': {'case_id': 'learning:translate_0.05_0deg', 'controls': 1000,
            'evaluation_seed': 20260914, 'deterministic_policy': True,
            'comparison': 'Compare final forward pass and planar tracking error at update 1200. Record the first passing scheduled checkpoint. Retain the unchanged native motor and contact checks.',
            'benefit_rule': 'A passing example-arm final policy with a failing scratch-arm final policy supports benefit in this seed. If both pass, an earlier first passing scheduled checkpoint supports a learning-speed benefit. Other outcomes do not establish this pilot benefit.'},
        'diagnostics': ['learning:quiet_20s', 'learning:forward_0.05_to_stop'],
        'loads': 'Record ground-contact force and applied/requested motor torque at 400 Hz. Compare loads with achieved speed; do not infer energy efficiency from a policy that fails to move.',
        'selection': 'Evaluate the scheduled checkpoints for both arms, including update zero. If the example arm passes at update zero, report a copied policy and its retention through PPO. Do not claim that PPO discovered that gait. Do not select only the best checkpoint or stop an arm after inspecting results.',
        'scope': 'One paired seed and one fixed forward command. Validation rows are later cycles of the same recording, not an independent locomotion test. This tests imitation initialization, not AMP or a paper reproduction. Stage 2 remains incomplete.',
        'extra_work': 'Report 1000 imitation updates and the existing demonstration acquisition separately from the equal PPO transition budgets.',
        'completion': 'Both full training jobs, scheduled policy evaluations, force summaries, videos and a paired result are needed for a learning conclusion. Setup and smoke checks do not establish benefit.',
        'stage2_complete': False}
    save(output/'PROTOCOL.json', value)
    load_protocol(output/'PROTOCOL.json', sha(output/'PROTOCOL.json'), arm='scratch',
                  seed=value['seed'], updates=value['updates'])
    return value


def prepare_arm(protocol_path, output, remote_root, *, arm, mode='train', smoke=False,
                checkpoint=None, checkpoint_sha=None, checkpoint_declaration_sha=None, inputs=None):
    protocol_path, output, remote_root = Path(protocol_path), Path(output), Path(remote_root)
    expected = sha(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    updates = 2 if smoke else protocol['updates']
    load_protocol(protocol_path, expected, arm=arm, seed=protocol['seed'], updates=updates, smoke=smoke)
    binding = prepare(output, remote_root, mode=mode, updates=updates, seed=protocol['seed'], experiment=True, inputs=inputs,
        checkpoint=checkpoint, checkpoint_sha=checkpoint_sha, checkpoint_declaration_sha=checkpoint_declaration_sha)
    inputs = output/'experiment'; inputs.mkdir()
    for name in ('PROTOCOL.json', 'demonstration.npz'):
        shutil.copy2(protocol_path.parent/name, inputs/name)
        binding['input_files'][str(remote_root/'experiment'/name)] = sha(inputs/name)
    binding['extra_mounts'].append([str(remote_root/'experiment'), '/realized_prior'])
    binding['command_args'] += ['--experiment-protocol', '/realized_prior/PROTOCOL.json',
        '--experiment-protocol-sha256', expected, '--experiment-arm', arm]
    if smoke:
        binding['command_args'].append('--experiment-smoke')
    save(output/'binding.json', binding)
    pack = json.loads((output/'PACK.json').read_text())
    pack.update(binding_sha256=sha(output/'binding.json'), experiment_protocol_sha256=expected,
        arm=arm, smoke=smoke, behavior_cloning=arm == 'example',
        files={p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
               if p.is_file() and p.name != 'PACK.json'})
    save(output/'PACK.json', pack)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    protocol = commands.add_parser('protocol')
    protocol.add_argument('--output', type=Path, required=True)
    pack = commands.add_parser('pack')
    pack.add_argument('--protocol', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--remote-root', required=True)
    pack.add_argument('--inputs', type=Path, required=True)
    pack.add_argument('--arm', choices=ARMS, required=True)
    pack.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    pack.add_argument('--smoke', action='store_true')
    pack.add_argument('--checkpoint', type=Path)
    pack.add_argument('--checkpoint-sha256')
    pack.add_argument('--checkpoint-declaration-sha256')
    args = parser.parse_args()
    if args.command == 'protocol':
        result = create_protocol(args.output)
    else:
        result = prepare_arm(args.protocol, args.output, args.remote_root, arm=args.arm,
            mode=args.mode, smoke=args.smoke, checkpoint=args.checkpoint,
            checkpoint_sha=args.checkpoint_sha256, checkpoint_declaration_sha=args.checkpoint_declaration_sha256,
            inputs=args.inputs)
    print(json.dumps({'output': str(args.output), 'schema': result['schema']}))


if __name__ == '__main__':
    main()
