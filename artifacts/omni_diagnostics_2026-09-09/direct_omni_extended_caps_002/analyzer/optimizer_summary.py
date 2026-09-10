"""Finite, partial-evidence optimizer summaries; standard library, no training imports.

This records diagnostics, never admits a run or turns a loss trend into a quiet
metric. Native host acceptance remains independently required.
"""
import math
import statistics

SCHEMA = 'direct315_actor_gradients_v1'
LOSS_KEYS = ('value', 'surrogate', 'entropy', 'caps_temporal', 'caps_spatial',
             'caps_weighted', 'caps_valid_pair_fraction',
             'caps_quiet_temporal_mean', 'caps_moving_temporal_mean')
NORM_KEYS = ('ppo_actor', 'quiet_temporal', 'moving_temporal', 'spatial')


def stats(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    if not values:
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'median': None}
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'mean': statistics.fmean(values), 'median': statistics.median(values)}


def summarize(receipt):
    result = {'schema': SCHEMA, 'errors': [], 'updates': [], 'sparse_gradients': [],
              'scope': 'Optimizer diagnostics only. Aggregate temporal losses are not quiet target-step p95; sparse gradients do not describe every minibatch or the Adam parameter update.',
              'pair_count_scope': 'Optimizer minibatch presentations, including repeated PPO epochs; not unique simulation transitions.',
              'quiet_mean_scope': 'Per-update conditional mean over valid exact-zero pairs; a zero with zero quiet-pair count is not observed quiet quality.'}
    errors = result['errors']
    if not isinstance(receipt, dict):
        receipt = {}; errors.append('Training receipt is missing or not an object')
    if receipt.get('optimizer_diagnostics_schema') != SCHEMA:
        errors.append('Optimizer diagnostics schema is missing or differs')
    result['producer_reported_complete'] = receipt.get('complete') is True
    result['producer_reported_updates_completed'] = receipt.get('updates_completed')
    selection = receipt.get('selection') or {}
    selection = selection if isinstance(selection, dict) else {}
    result['selection'] = selection
    rows = receipt.get('optimizer_updates')
    if not isinstance(rows, list):
        rows = []; errors.append('Optimizer update list is missing or malformed')
    if type(receipt.get('updates_completed')) is not int or receipt['updates_completed'] != len(rows):
        errors.append('Producer update count does not match retained update inventory')
    if result['producer_reported_complete'] and selection.get('updates') != len(rows):
        errors.append('Completed receipt does not match selected update budget')

    def number(value, label, *, nonnegative=False):
        if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
            errors.append(label + ': missing/invalid finite number')
            return None
        return value

    batch_size = selection.get('replicas')
    batch_size = batch_size * 6 if type(batch_size) is int and batch_size > 0 else None
    all_kl, all_before, all_after, end_rates = [], [], [], []
    total_counts = {'valid_pairs': 0, 'quiet_pairs': 0, 'moving_pairs': 0}
    count_rows = 0
    for update_index, update in enumerate(rows, 1):
        label = 'update ' + str(update_index)
        if not isinstance(update, dict):
            errors.append(label + ': malformed row'); continue
        if type(update.get('completed_update')) is not int or update.get('completed_update') != update_index:
            errors.append(label + ': sequence label differs')
        losses = update.get('losses')
        if not isinstance(losses, dict):
            losses = {}; errors.append(label + ': missing losses')
        record = {'completed_update': update_index,
                  'losses': {k: number(losses.get(k), label + '.' + k) for k in LOSS_KEYS},
                  'optimizer_wall_seconds': number(update.get('optimizer_wall_seconds'), label + '.optimizer_wall_seconds', nonnegative=True),
                  'learning_rate': number(update.get('learning_rate'), label + '.learning_rate', nonnegative=True)}
        end_rates.append(record['learning_rate'])
        minibatches = update.get('minibatches')
        if not isinstance(minibatches, list):
            minibatches = []; errors.append(label + ': missing minibatches')
        if len(minibatches) != 20:
            errors.append(label + ': expected 20 retained minibatches')
        kls, rates_before, rates_after = [], [], []
        counts = {'valid_pairs': 0, 'quiet_pairs': 0, 'moving_pairs': 0}
        valid_count_rows = 0
        for mb_index, mb in enumerate(minibatches, 1):
            mb_label = label + ' minibatch ' + str(mb_index)
            if not isinstance(mb, dict):
                errors.append(mb_label + ': malformed row'); continue
            if mb.get('update') != update_index or mb.get('minibatch') != mb_index:
                errors.append(mb_label + ': sequence labels differ')
            kl = number(mb.get('kl_mean'), mb_label + '.kl_mean')
            before = number(mb.get('learning_rate_before'), mb_label + '.learning_rate_before', nonnegative=True)
            after = number(mb.get('learning_rate_after'), mb_label + '.learning_rate_after', nonnegative=True)
            kls.append(kl); rates_before.append(before); rates_after.append(after)
            for name, rate in (('before', before), ('after', after)):
                if rate is not None and not 1e-5 <= rate <= 1e-2:
                    errors.append(mb_label + ': LR ' + name + ' outside existing native bounds')
            pair = mb.get('pair_counts')
            pair_valid = False
            if (not isinstance(pair, dict) or set(pair) != set(counts)
                    or any(type(v) is not int or v < 0 or (batch_size is not None and v > batch_size) for v in pair.values())
                    or pair['quiet_pairs'] + pair['moving_pairs'] != pair['valid_pairs']):
                errors.append(mb_label + ': malformed pair counts')
            else:
                pair_valid = True
                valid_count_rows += 1
                for key in counts: counts[key] += pair[key]
            gradient = mb.get('gradient')
            expected_sparse = update_index in ((1, 10, 25, 50, 100, 250, 500) if selection.get('allocation') == 'extended' else (1, 10, 25, 50)) and mb_index in (1, 20)
            if expected_sparse and not isinstance(gradient, dict):
                errors.append(mb_label + ': missing scheduled sparse gradient')
            elif gradient is not None:
                if not expected_sparse:
                    errors.append(mb_label + ': unexpected sparse gradient row')
                if not isinstance(gradient, dict):
                    errors.append(mb_label + ': malformed gradient'); continue
                norms = gradient.get('norms')
                if not isinstance(norms, dict): norms = {}
                if set(norms) != set(NORM_KEYS): errors.append(mb_label + ': gradient component inventory differs')
                clean = {'update': update_index, 'minibatch': mb_index,
                         'norms': {k: number(norms.get(k), mb_label + '.' + k, nonnegative=True) for k in NORM_KEYS},
                         'pair_counts': pair if pair_valid else None}
                for key in ('component_sum_norm', 'combined_actor_before_clip', 'combined_actor_after_clip'):
                    clean[key] = number(gradient.get(key), mb_label + '.' + key, nonnegative=True)
                cosine = gradient.get('ppo_quiet_cosine')
                if cosine is not None:
                    cosine = number(cosine, mb_label + '.ppo_quiet_cosine')
                    if cosine is not None and not -1.00001 <= cosine <= 1.00001:
                        errors.append(mb_label + ': cosine outside numeric range'); cosine = None
                clean['ppo_quiet_cosine'] = cosine
                clean['cosine_null_reason'] = 'Undefined or unavailable; zero component norms yield no cosine.' if cosine is None else None
                result['sparse_gradients'].append(clean)
        if rates_after and record['learning_rate'] != rates_after[-1]:
            errors.append(label + ': final minibatch and update-end learning rates differ')
        record.update(minibatches_retained=len(minibatches), pair_count_rows_valid=valid_count_rows,
                      pair_presentations=counts, quiet_valid_pair_fraction=counts['quiet_pairs']/counts['valid_pairs'] if counts['valid_pairs'] else None,
                      kl_mean=stats(kls), learning_rate_before=stats(rates_before), learning_rate_after=stats(rates_after),
                      minibatch_learning_rates=[{'minibatch': i + 1, 'before': b, 'after': a, 'kl_mean': k} for i, (b, a, k) in enumerate(zip(rates_before, rates_after, kls))])
        result['updates'].append(record)
        all_kl.extend(kls); all_before.extend(rates_before); all_after.extend(rates_after)
        count_rows += valid_count_rows
        for key in total_counts: total_counts[key] += counts[key]
    result.update(updates_retained=len(rows), minibatches_retained=sum(u['minibatches_retained'] for u in result['updates']),
                  pair_count_rows_valid=count_rows, pair_presentations=total_counts,
                  quiet_valid_pair_fraction=total_counts['quiet_pairs']/total_counts['valid_pairs'] if total_counts['valid_pairs'] else None,
                  kl_mean=stats(all_kl), learning_rate_before=stats(all_before), learning_rate_after=stats(all_after),
                  update_end_learning_rates=end_rates,
                  learning_rate_at_existing_floor_count=sum(v == 1e-5 for v in all_after),
                  sparse_gradient_rows_retained=len(result['sparse_gradients']))
    # Disjoint initial/final windows, so a two-update smoke compares 1 vs 1.
    width = min(10, len(result['updates']) // 2)
    result['loss_windows'] = {'updates_per_window': width,
        'initial_update_ids': [u['completed_update'] for u in result['updates'][:width]],
        'final_update_ids': [u['completed_update'] for u in result['updates'][-width:]] if width else [],
        'initial': {key: stats([u['losses'][key] for u in result['updates'][:width]]) for key in LOSS_KEYS},
        'final': {key: stats([u['losses'][key] for u in result['updates'][-width:]]) if width else stats([]) for key in LOSS_KEYS}}
    result['diagnostic_inventory_complete'] = not errors
    result['admission'] = False
    return result


def human(summary):
    def fmt(value): return f'{value:.6g}' if type(value) in (int, float) and math.isfinite(value) else 'unavailable'
    if not isinstance(summary, dict): return ['Optimizer diagnostics are unavailable.']
    lines = ['Optimizer diagnostics (measurement, not a qualification):', '',
        f"Retained {summary.get('updates_retained', 0)} update rows / {summary.get('minibatches_retained', 0)} minibatches; {summary.get('sparse_gradient_rows_retained', 0)} sparse gradient rows.",
        f"Minibatch LR range {fmt((summary.get('learning_rate_after') or {}).get('min'))}–{fmt((summary.get('learning_rate_after') or {}).get('max'))}; {summary.get('learning_rate_at_existing_floor_count', 0)} retained rows at the existing 1e-5 floor. Mean recorded KL {fmt((summary.get('kl_mean') or {}).get('mean'))}.", '',
        '| Loss | Initial-window mean | Final-window mean |', '|---|---:|---:|']
    windows = summary.get('loss_windows') or {}
    for key in ('caps_temporal', 'caps_quiet_temporal_mean', 'caps_moving_temporal_mean', 'caps_spatial', 'caps_weighted'):
        lines.append(f"| {key} | {fmt((windows.get('initial', {}).get(key) or {}).get('mean'))} | {fmt((windows.get('final', {}).get(key) or {}).get('mean'))} |")
    lines += ['', f"Windows contain {windows.get('updates_per_window', 0)} disjoint updates each; exact IDs and all minibatch LR/KL values are in report.json.",
              'Conditional quiet/moving means use their own pair counts. These optimizer presentations repeat PPO epochs; they are not unique environment steps.', '',
              '| Update / minibatch | PPO actor norm | Quiet norm | Moving norm | Spatial norm | PPO–quiet cosine | Combined before → after clip |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary.get('sparse_gradients', []):
        norms = row.get('norms') or {}
        values = ' | '.join(fmt(norms.get(k)) for k in NORM_KEYS)
        lines.append(f"| {row.get('update')} / {row.get('minibatch')} | {values} | {fmt(row.get('ppo_quiet_cosine'))} | {fmt(row.get('combined_actor_before_clip'))} → {fmt(row.get('combined_actor_after_clip'))} |")
    lines += ['', summary.get('scope', 'No optimizer-based admission.')]
    if summary.get('errors'): lines += ['', 'Optimizer evidence gaps:'] + ['- ' + str(e) for e in summary['errors']]
    return lines
