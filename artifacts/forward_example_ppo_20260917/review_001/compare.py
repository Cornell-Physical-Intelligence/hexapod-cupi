"""Apply the frozen pilot benefit rule to all scheduled policy evaluations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    inputs = {}
    def read(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())
    protocol_path = STUDY/'protocol_001/PROTOCOL.json'
    protocol = read(protocol_path)
    assert protocol['arms'] == ['scratch', 'example']
    assert protocol['evaluation_updates'] == [0, 300, 600, 1200]
    training, rows, initializations, identities = {}, {}, {}, {}
    for arm in protocol['arms']:
        train = STUDY/f'{arm}_train_003/run/standing'
        state = read(train/'state.json')
        assert state['status'] == 'completed' and not state['errors']
        assert state['updates'] == protocol['updates'] == 1200
        assert state['transitions'] == protocol['transitions_per_arm'] == 3686400
        assert state['identity']['experiment']['protocol_sha256'] == sha(protocol_path)
        assert state['identity']['experiment']['arm'] == arm
        assert state['identity']['experiment']['smoke'] is False
        assert state['identity']['experiment']['command'] == [.05, 0., 0.]
        assert state['identity']['seed'] == protocol['seed']
        assert state['identity']['config']['num_envs'] == protocol['num_envs']
        identities[arm] = state['identity']
        initialization = read(train/'initialization.json')
        assert initialization['bc_updates'] == (1000 if arm == 'example' else 0)
        assert initialization['ppo_optimizer_empty'] and initialization['ppo_rng_preserved']
        initializations[arm] = initialization
        training[arm] = {'updates': state['updates'], 'transitions': state['transitions'],
            'bc_updates': initialization['bc_updates'], 'final_checkpoint_sha256': state['checkpoint_sha256'],
            'source_freeze_sha256': state['runtime_binding']['runtime_tree_sha256']}
        rows[arm] = []
        for update in protocol['evaluation_updates']:
            audit = read(STUDY/f'review_001/{arm}_update{update:06d}.json')
            assert audit['arm'] == arm and audit['updates'] == update
            assert audit['protocol_sha256'] == sha(protocol_path) and audit['cleanup_verified']
            assert len(audit['cases']) == 3
            forward = next(case for case in audit['cases'] if case['case_id'] == protocol['primary']['case_id'])
            original = forward['original_result']
            assert forward['controls'] == forward['requested_controls'] == 1000
            assert forward['acquisition_complete'] and forward['failure'] is None
            assert forward['physics_steps'] == 8000 and forward['force_summary_reproduced']
            assert original['lineage']['checkpoint_sha256'] == audit['checkpoint_sha256']
            if update == 1200:
                assert audit['checkpoint_sha256'] == state['checkpoint_sha256']
            default_pack = STUDY/f'{arm}_evaluate_update{update:06d}_001'
            eval_pack = ROOT/audit.get('evaluation_pack', str(default_pack.relative_to(ROOT)))
            assert eval_pack.parent == STUDY
            eval_path = eval_pack/'run/standing/evaluation_00'
            assert sha(eval_path/'report.json') == forward['report_sha256']
            assert sha(eval_path/'rollout.mp4') == forward['video_sha256']
            report = read(eval_path/'report.json')
            assert report['cases'][0]['case_id'] == protocol['primary']['case_id']
            assert report['cases'][0]['command'] == protocol['command']
            assert report['seed'] == protocol['seed']
            loads = read(eval_path/'force_metrics.json')
            window = loads['cases'][0]['windows']['commanded_locomotion_after_settle']
            assert window['samples'] == 7200
            rows[arm].append({'updates': update, 'ppo_transitions': update*128*24,
                'checkpoint_sha256': audit['checkpoint_sha256'],
                'evaluation_pack': str(eval_pack.relative_to(ROOT)),
                'forward_screen_pass': original['pass'], 'failed_bounds': original['failed_bounds'],
                'mean_forward_speed_mps': original['metrics']['mean_velocity_mps'][0],
                'planar_error_mps': original['metrics']['planar_error_mps'],
                'mean_vertical_support_n': window['total_vertical_support_n']['mean'],
                'support_p95_n': window['total_vertical_support_n']['p95'],
                'mean_abs_applied_motor_torque_nm': window['motor_torque']['applied']['mean_abs_across_joints_nm'],
                'worst_joint_rms_torque_nm': window['motor_torque']['applied']['worst_joint_rms_nm'],
                'diagnostics': [{'case_id': case['case_id'], 'pass': case['original_result']['pass'],
                    'recorded_controls': case['controls'], 'requested_controls': case['requested_controls'],
                    'acquisition_complete': case['acquisition_complete'], 'failure': case['failure'],
                    'failed_bounds': case['original_result']['failed_bounds']} for case in audit['cases']
                    if case['case_id'] != protocol['primary']['case_id']],
                'forward_video': str((eval_path/'rollout.mp4').relative_to(ROOT)),
                'forward_video_sha256': forward['video_sha256']})
    for key in ('initial_actor_sha256', 'shared_actor_non_mlp_sha256', 'shared_critic_sha256', 'before'):
        assert initializations['scratch'][key] == initializations['example'][key], key
    assert initializations['scratch']['final_actor_sha256'] != initializations['example']['final_actor_sha256']
    assert training['scratch']['source_freeze_sha256'] == training['example']['source_freeze_sha256']
    for key in ('source_files', 'model_sha256', 'usd_sha256', 'physics_source_files',
                'physics_config', 'upstream_source_files', 'ppo_config', 'seed'):
        assert identities['scratch'][key] == identities['example'][key], key
    retry = read(STUDY/'EVALUATION_RETRY_001.json')
    retry_verification = read(STUDY/'EVALUATION_RETRY_VERIFICATION_001.json')
    assert retry_verification['retry_receipt_sha256'] == sha(STUDY/'EVALUATION_RETRY_001.json')
    assert retry['arm'] == 'example' and retry['updates'] == 600
    repeated = next(row for row in rows['example'] if row['updates'] == 600)
    assert repeated['evaluation_pack'] == retry['retry_pack']
    assert repeated['checkpoint_sha256'] == retry['checkpoint_sha256']
    interrupted = read(ROOT/retry['interrupted_pack']/'run/jobs/standing.json')
    assert interrupted['status'] == 'failed'
    assert interrupted['error'] == "RuntimeError('Unrelated CUDA process appeared; yielding this owned job')"
    first = {arm: next((row['updates'] for row in rows[arm] if row['forward_screen_pass']), None)
             for arm in protocol['arms']}
    scratch_pass, example_pass = (rows[arm][-1]['forward_screen_pass'] for arm in protocol['arms'])
    final_benefit = example_pass and not scratch_pass
    earlier_pass = bool(scratch_pass and example_pass and first['example'] < first['scratch'])
    copied = rows['example'][0]['forward_screen_pass']
    benefit = bool(final_benefit or earlier_pass)
    conclusion = ('The declared benefit rule passes for this paired seed.' if benefit
                  else 'The declared benefit rule does not establish benefit for this paired seed.')
    if copied:
        conclusion += ' The example arm passes before PPO, which establishes imitation.'
        conclusion += (' Its final policy retains a passing forward result.' if example_pass
                       else ' Its final policy loses the passing forward result.')
    else:
        conclusion += ' The copied actor fails the forward screen before PPO.'
    result = {'schema': 'canonical_forward_example_comparison_v1',
        'utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(protocol_path),
        'training': training, 'scheduled_forward_results': rows,
        'operational_retries': [retry],
        'extra_work': protocol['extra_work'],
        'first_passing_scheduled_update': first,
        'declared_benefit_rule': protocol['primary']['benefit_rule'],
        'benefit_established_for_this_pair': benefit,
        'example_passes_before_ppo': copied, 'conclusion': conclusion,
        'load_scope': 'Compare loads with achieved speed. These values do not establish energy efficiency at matched motion.',
        'scope': protocol['scope'], 'inputs': inputs,
        'analysis_source_sha256': sha(Path(__file__)), 'stage2_complete': False}
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'conclusion': conclusion, 'first_passing_scheduled_update': first}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
