#!/usr/bin/env python3
"""Summarize completed omni evaluations without changing their acceptance limits."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
campaign=ROOT/'artifacts/omni_flat_2026-09-09/omni_flat_002'
rows=[]
for stage in sorted(campaign.glob('stage_*')):
    path=stage/'evaluation.json'
    if not path.exists():continue
    e=json.loads(path.read_text())
    if not e.get('complete'):continue
    counts={k:0 for k in ('terminations','truncations','nonfoot','saturation','planar','yaw','tilt','vertical')}
    for r in e['static']:
        checks={'terminations':r['terminations']==0,'truncations':r['truncations']==0,
                'nonfoot':r['nonfoot_fraction']<=.001,'saturation':r['torque_saturation_fraction']<=.005,
                'planar':r['planar_error_mps']<=max(.025,.25*math.hypot(*r['command'][:2])),
                'yaw':r['yaw_error_rad_s']<=max(.06,.25*abs(r['command'][2])),
                'tilt':r['tilt_rms_deg']<=5,'vertical':r['vertical_velocity_rms_mps']<=.04}
        if all(checks.values()) != r['pass']:raise RuntimeError('Acceptance definition drift')
        for k,value in checks.items():counts[k]+=not value
    metrics=('planar_error_mps','yaw_error_rad_s','torque_saturation_fraction','nonfoot_fraction',
             'tilt_rms_deg','vertical_velocity_rms_mps','positive_mechanical_power_w')
    rows.append({'stage':stage.name,'checkpoint_sha256':e['checkpoint_sha256'],
                 'evaluation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'static_total':len(e['static']),'static_passed':sum(r['pass'] for r in e['static']),
                 'transition_passed':sum(r['pass'] for r in e['transitions']),
                 'failed_rows_by_criterion':counts,
                 'median_over_scenarios':{k:float(np.median([r[k] for r in e['static']])) for k in metrics}})
report={'status':'completed_campaign_not_qualified','stages':rows,
        'definitions':['Saturation is requested/computed torque above1.6Nm, not applied torque above1.6Nm',
                       'Failure counts overlap; one row can fail several criteria',
                       'Current evaluation omits first2s globally but includes later resets; diagnose post-reset contamination separately',
                       'Demo uses ideal simulator pose feedback and cannot replace the open-loop qualification'],
        'next_experiment':{'purpose':'identify motor/velocity oscillation before another long run',
            'unchanged':['checkpoint','asset','1.6Nm applied cap','command scenarios'],
            'record':['runtime joint_names','joint target/position/velocity','computed/applied torque',
                      'termination cause and time-since-reset','body velocity vs finite-difference pose',
                      'per-joint saturation duration and tracking error'],
            'comparisons':['existing controller','one change: target slew0.06 to0.03rad per20ms',
                           'separate numerical probe: external-forces-every-iteration solver option if available'],
            'follow_up':'If oscillation is physical, short PPO continuation with stronger motor-demand/smoothness costs; keep gates fixed',
            'not_started':True}}
(campaign/'progress_review.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Completed omni campaign review','','The run finished; the flat motion milestone remains unqualified.','',
       '| Stage | Median planar error (m/s) | Median yaw error (rad/s) | Median requested-torque saturation | Static gates |',
       '|---|---:|---:|---:|---:|']
for r in rows:
    m=r['median_over_scenarios'];lines.append(f"| {r['stage']} | {m['planar_error_mps']:.4f} | {m['yaw_error_rad_s']:.4f} | {m['torque_saturation_fraction']:.1%} | {r['static_passed']}/{r['static_total']} |")
lines+=['','The final checkpoint improves position tracking but still has excessive requested motor demand and instantaneous velocity/yaw error. Applied motor torque remains capped at1.6Nm. Safety terminations are not automatically falls.','',
        'Next: a short instrumented comparison of the same checkpoint, recording each named joint, processed target, position, velocity, computed/applied torque and reset age. Separate steady walking from reset transients. Compare one target-slew change at a time; separately test any supported solver option before attributing all noise to learned behavior. Then choose a short training repair. Do not launch a long terrain campaign based solely on the improved path video.','',
        'The corrected69-second path demo completed all five trials without terminations. Its ideal simulator localization is available to the outer follower, not the proprioceptive PPO actor.','']
(campaign/'PROGRESS_REVIEW.md').write_text('\n'.join(lines))
print(json.dumps(report,indent=2))
