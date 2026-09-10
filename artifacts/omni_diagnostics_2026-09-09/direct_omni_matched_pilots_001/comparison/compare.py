"""Compare source-matched 50-update branches from frozen analyzer002 evidence."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,statistics
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--curriculum',type=Path,default=ROOT/'tmp/direct_omni_train_pilot_curriculum_analysis_001/report.json');parser.add_argument('--caps',type=Path,default=ROOT/'tmp/direct_omni_train_pilot_caps_analysis_001/report.json');parser.add_argument('--analyzer',type=Path,default=ROOT/'tmp/direct_omni_train_analysis_002');parser.add_argument('--output',type=Path,default=HERE);args=parser.parse_args();ANALYZER=args.analyzer;HERE=args.output;HERE.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('frozen_comparison_helpers',ANALYZER/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={'curriculum':args.curriculum,'caps':args.caps};reports={k:json.loads(p.read_text()) for k,p in paths.items()}
a,b=reports['curriculum'],reports['caps'];inputs={str(p):sha(p) for p in paths.values()}
for branch,d in reports.items():
 assert not d['errors'] and d['campaign']['status']=='completed' and d['training']['complete'] and d['training']['updates_completed']==50
 assert d['training']['selection']['branch']==branch and d['training']['selection']['replicas']==1024
 assert d['campaign']['identity']['source_manifest_sha256']=='64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e'
 assert d['training']['strict_reload']['passed'] and d['final_stop']['total_replicas']==48
 assert d['initial_constant']['checkpoint_sha256']==m.ORIGINAL
constant=m.compare_constant(a['final_constant'],b['final_constant']);stop=m.compare_stop(a['final_stop'],b['final_stop'])
summary={}
for branch,d in reports.items():
 t=d['training'];rows=d['final_stop']['cases'];s={'checkpoint_sha256':t['checkpoint_sha256'],'updates':50,'transitions':t['transitions'],'wrapper_seconds':t['training_wrapper_wall_seconds'],'transitions_per_second':t['wrapper_transitions_per_second'],'training_terminations':sum(x['terminations'] for x in t['per_row']),'training_timeouts':sum(x['truncations'] for x in t['per_row']),'training_nonfoot_env_steps':sum(x['nonfoot_env_steps'] for x in t['per_row']),'final_quiet_passed':d['final_stop']['quiet_passed_replicas'],'final_quiet_metrics':{}}
 for key,bound in m.QUIET_GATES.items():
  vals=[r['quiet'][key] for r in rows];s['final_quiet_metrics'][key]={'bound':bound,'minimum':min(vals),'median':statistics.median(vals),'maximum':max(vals),'failed_replicas':sum(v>bound for v in vals)}
 s['constant_cases_with_saturation_over_0_005']=[r['name'] for r in d['final_constant']['cases'] if r['metrics']['torque_saturation_fraction']>.005]
 summary[branch]=s
report={'schema':'matched_direct315_curriculum_CAPS50_review_v1','Stage2_complete':False,'automatic_continuation':False,'inputs_sha256':inputs,'frozen_analyzer_sha256':sha(ANALYZER/'FREEZE_SHA256.json'),'initial_constant_evidence_exact_equal':a['initial_constant']==b['initial_constant'],'initial_stop_evidence_exact_equal':a['initial_stop']==b['initial_stop'],'summary':summary,'constant_caps_minus_curriculum':constant,'all48_stop_caps_minus_curriculum':stop,'preview_candidate':'caps','preview_scope':'Latest 50-update unqualified progress only; fewer extreme final stop excursions, not a smoothness/acceptance success.','next_decision':'Neither branch passed quiet. Preserve both; inspect source audit and per-step raw target/observation response before selecting a changed smoothness intervention. No loss-only or median-only promotion and no automatic unchanged continuation.'}
lines=['# Matched 50-update direct PPO pilots','','Both pilots completed 50 updates and strict reload. Both failed 48/48 final quiet trials. Stage 2 remains incomplete.','', '| Quantity | Curriculum | CAPS |','|---|---:|---:|']
for label,key in [('Actual transitions','transitions'),('Learn plus final verification (s)','wrapper_seconds'),('Training terminations','training_terminations'),('Training nonfoot environment-steps','training_nonfoot_env_steps'),('Final quiet passing replicas','final_quiet_passed')]:lines.append(f"| {label} | {summary['curriculum'][key]} | {summary['caps'][key]} |")
lines+=['','| Final quiet metric, 48 replicas | Existing bound | Curriculum median / worst | CAPS median / worst |','|---|---:|---:|---:|']
for key in ['max_planar_excursion_m','max_joint_velocity_rms_rad_s','max_target_step_abs_p95_rad_per_20ms','max_requested_torque_saturation_fraction']:
 x,y=summary['curriculum']['final_quiet_metrics'][key],summary['caps']['final_quiet_metrics'][key];lines.append(f"| {key} | {x['bound']:.6g} | {x['median']:.6g} / {x['maximum']:.6g} | {y['median']:.6g} / {y['maximum']:.6g} |")
lines+=['','These summaries do not hide per-replica failures: all 48 paired stop rows and all 12 directional changes are in report.json. Initial constant and initial stop evidence compare exactly between branches. Applied torque remains capped; requested saturation still fails. Raw SDK joint rates and interval-angle evidence remain separate.','', 'CAPS is the proposed preview checkpoint because its worst measured final stop excursion is 75.1 mm versus 1.887 m for curriculum. Neither is qualified: even CAPS exceeds 10 mm drift, 0.03 rad/s raw joint RMS, .002 rad target-step and .5% requested-saturation bounds. A preview must identify 50 updates, 1× playback and Stage 2 incomplete.','',report['next_decision'],'','Latest checkpoints:','']
for branch,s in summary.items():lines.append(f"- {branch}: `{s['checkpoint_sha256']}` (50 updates).")
assert all(sha(Path(k))==v for k,v in inputs.items())
(HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'initial_constant_exact_equal':report['initial_constant_evidence_exact_equal'],'initial_stop_exact_equal':report['initial_stop_evidence_exact_equal'],'preview_candidate':'caps','Stage2_complete':False},indent=2))
