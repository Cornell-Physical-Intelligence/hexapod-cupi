"""Stdlib all-step diagnostic readout; stdout only and never an admission."""
import argparse
import hashlib
import json
from pathlib import Path
from numeric_evidence import numeric
from saved_aggregation import aggregate, norm, require, LEGS

DT = .0025
SETTLE_STEPS = 1600


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def detect(samples):
    """The saved strict >1 N detector, expanded to settling and bracketed events."""
    counts = {leg: {'all_low_samples': 0, 'post_settle_low_samples': 0,
                    'isolated_zero_losses': 0, 'isolated_zero_128_signature': 0} for leg in LEGS}
    events = []
    for i, row in enumerate(samples):
        require(row['sequence'] == i, 'Noncontiguous detector sequence')
        for k, leg in enumerate(LEGS):
            force = row['toe_force_norm_n'][k]
            if force <= 1.:
                counts[leg]['all_low_samples'] += 1
                counts[leg]['post_settle_low_samples'] += int(i >= SETTLE_STEPS)
                isolated = (0 < i < len(samples)-1
                            and samples[i-1]['toe_force_norm_n'][k] > 1.
                            and samples[i+1]['toe_force_norm_n'][k] > 1.)
                if force == 0. and isolated:
                    counts[leg]['isolated_zero_losses'] += 1
                    signature = row['toe_patch_signatures'][leg]
                    matched = signature['exact128_inactive_zero_tuples']
                    counts[leg]['isolated_zero_128_signature'] += int(matched)
                    events.append({'sequence': i, 'time_s': (i+1)*DT,
                                   'leg': leg, 'body': leg+'_tibia',
                                   'post_settle': i >= SETTLE_STEPS,
                                   'before_force_n': samples[i-1]['toe_force_norm_n'][k],
                                   'force_n': force,
                                   'after_force_n': samples[i+1]['toe_force_norm_n'][k],
                                   **signature})
    return {'steps': len(samples), 'per_toe': counts, 'events': events,
            'window': {'all_steps': len(samples), 'settle_steps': SETTLE_STEPS,
                       'dt_s': DT, 'support_is_strictly_above_n': 1.},
            'standing_admission': False, 'batch_admission': False, 'training_allowed': False}


def analyze(phase):
    phase = Path(phase)
    read = lambda name: json.loads((phase/name).read_text())
    state = read('state.json')
    require(state['schema'] == 'canonical_single_placement_diagnostic_v1', 'Not diagnostic evidence')
    require(state['status'] == 'completed' and state['explicit_steps_completed'] == 8000,
            'Complete 8000-step acquisition required for this readout')
    require(state['identity']['num_envs'] == 1 and state['identity']['training_allowed'] is False,
            'Single non-admitting diagnostic identity required')
    session = read('session.json')
    require(session['root_paths'] == ['/Robot'], 'Different topology')
    checked = {}
    def check(name):
        actual = sha(phase/name)
        require(state['outputs'].get(name) == actual, 'Changed or unsealed input: '+name)
        checked[name] = actual
        return phase/name
    check('session.json'); check('contact_view.json'); check('standing_report.json')
    samples = []
    for name in session['substep_files']:
        path = check(name)
        seq = numeric(path, 'sequence', (800,), 'int')
        flags = numeric(path, 'distal_contact', (800,1,6), 'bool')
        forces = numeric(path, 'distal_force_world_n', (800,1,6,3))
        for j, s in enumerate(seq):
            require(s == len(samples), 'Sequence changed')
            values = [norm(forces[(j*6+k)*3:(j*6+k+1)*3]) for k in range(6)]
            require([bool(flags[j*6+k]) for k in range(6)] == [v>1. for v in values],
                    'Saved support detector disagrees with raw force')
            samples.append({'sequence': s, 'toe_force_norm_n': values,
                            'toe_force_vectors': [list(forces[(j*6+k)*3:(j*6+k+1)*3]) for k in range(6)]})
    require(len(samples) == 8000, 'Wrong step count')
    capacity = read('contact_view.json')['capacity']
    peak_used = 0
    i = -1
    with check('contacts.jsonl').open() as stream:
        for i, line in enumerate(stream):
            require(i < 8000 and len(line) <= 32 << 20, 'Unexpected contact stream length')
            row = json.loads(line)
            require(row['sequence'] == i, 'Contact sequence differs')
            patches = row['patches']
            toe, _, _ = aggregate(patches, 1)
            require(toe[0] == samples[i]['toe_force_vectors'], 'Saved exact force aggregation differs')
            require(len(patches) <= capacity and all(p['buffer_index']<capacity for p in patches),
                    'Exported contact buffer exceeded')
            peak_used = max(peak_used, len(patches))
            signs = {}
            for leg in LEGS:
                selected = [p for p in patches if p['env'] == 0 and p['body'] == leg+'_tibia']
                exact = (len(selected) == 128 and all(p['inactive_zero_normal'] is True
                         and p['normal_force_n'] == 0. and p['normal_world'] == [0.,0.,0.]
                         and p['separation_m'] == 0. for p in selected))
                signs[leg] = {'patch_count': len(selected),
                              'inactive_count': sum(p['inactive_zero_normal'] for p in selected),
                              'exact128_inactive_zero_tuples': exact}
            samples[i]['toe_patch_signatures'] = signs
    require(i == 7999, 'Incomplete contact stream')
    result = detect(samples)
    report = read('standing_report.json')
    require(sum(any(v<=1. for v in row['toe_force_norm_n']) for row in samples[SETTLE_STEPS:]) ==
            report['replicas'][0]['physical']['post_settle_missing_six_toe_substeps'],
            'Detector loses reported support failures')
    result.update(schema='canonical_placement_event_readout_v1', identity=state['identity'],
                  original_gate_report=report, consumed_inputs=checked,
                  contact_slots={'observed_capacity':capacity,'maximum_exported_used':peak_used,
                                 'meaning_of_128':'Observed record signature only; native starts/counts and internal per-pair capacity are not inferred'})
    for name, expected in checked.items():
        require(sha(phase/name) == expected, 'Input changed during analysis')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.phase), indent=2, allow_nan=False))
