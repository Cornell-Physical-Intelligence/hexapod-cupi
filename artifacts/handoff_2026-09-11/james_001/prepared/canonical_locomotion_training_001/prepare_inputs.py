from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent; repo=R.parents[1]
if (R/'FREEZE_SHA256.json').exists()or(R/'fable/FABLE_LAUNCH.json').exists():raise SystemExit('Preserve frozen inputs/actual consultation; create a new version')
files={
'adapter.py':'tmp/canonical_ppo_integration_003/canonical_direct_ppo/adapter.py',
'frames.py':'tmp/canonical_ppo_integration_003/canonical_direct_ppo/frames.py',
'commands.py':'tmp/canonical_walk_objective_001/commands.py',
'objective.py':'tmp/canonical_walk_objective_001/objective.py',
'causal_command.py':'tmp/canonical_walk_objective_001/causal_command.py'}
inputs={}
for name,path in files.items():
 b=(repo/path).read_bytes();(R/'oracles'/name).write_bytes(b);inputs[path]={'sha256':hashlib.sha256(b).hexdigest(),'selected_copy':'oracles/'+name}
for path in ['tmp/canonical_ppo_integration_003/FREEZE_SHA256.json','tmp/updated_native_standing_005/FREEZE_SHA256.json','tmp/canonical_walk_objective_001/FREEZE_SHA256.json','tmp/canonical_ppo_throughput_design_001/BENCHMARK.json','tmp/canonical_ppo_throughput_design_001/EVIDENCE.json']:
 b=(repo/path).read_bytes(); inputs[path]={'sha256':hashlib.sha256(b).hexdigest(),'copy':False}
(R/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n')
shutil.copyfile(repo/'tmp/canonical_walk_objective_001/fable/run_fable.py',R/'fable/run_fable.py')
prompt='''Act as an independent implementation reviewer. Respond in <=700 words of final findings only; no tools, no edits, no authority to launch. Exact model Claude Fable5.1, MAX. User requires central docs/PROJECT_SITE.md records for later repo adoption; this is isolated CPU proposal only.
Canonical detailed direct-drive 7.466088235kg,19 bodies18 joints. Native source005 uses32 position/0 velocity solver,50Hz holds/400Hz PD,1.6Nm and <=0.5% requested saturation,0.040rad/20ms actual target limiter. No geometry/physics changes. Fresh1 standing passed; matching32 pending. Frozen PPO003 is32x24x2 zero-command no-auto-reset infrastructure,405 actor/408 critic,5x81 history, rawdq and actual heldtarget-q, previous clipped action. MLP512/256/128 ELU, empiricalnormalizers, action targetscale.10rad/std.02; no gaitclock/filter/fakevelocity. Six-toe standing bridge rejects all lifts and nonzero commands. It cannot run walking unchanged.
Existing reviewed objective001 proposes level0 requests.02-.04m/s/.08-.16rad/s, allbearings/yaw/arcs/stops/reversals;32x24x50 noautoreset is unapproved proposal. Tracks all8 body-root samples, tilt/effort/actual target slew/acceleration/slip; no q-neutral posture reward. Commands fixed12s per-row programs, explicit program restart with NO physical/history reset. Reward must consume c[t], posthold history includes c[t+1] and heldtarget[t]-q[t+1]. Initial actor history repeats firstvalidframe (declared initialization, not five measured previous commands).
New bounded implementation CPU seams: (1) transactional command/history sequencing with no next actor on incomplete hold; (2) reward packet receives exact8 native states and typed contact evidence, maps world->nativebody->(-y,x,z), root COM velocity corrected to rootorigin. Finite material-point slip uses same current patch material point transformed into previous link pose; no patch-position drift. Native classifier already accepts rawnormals within1e-3 and exact f=0,n=0,sep=0 inactive rows. Reward normalizes ONLY previously accepted active normals. Current source005 exposes8numeric rows but NOT rawpatches via accessor (only fullJSONstream). New read-only complete8patch accessor needed; preserve everyzero/failed row.
No moving min support count or gait schedule adopted. Suggest explicit separate proposed moving rule pending real singlefootlift/return and review; never sixsupport whenever c==0 (feet can stillreturn). Empty valid contacts can'tbe admitted as locomotion. A Boolean force-off->on event alone is not qualifiedflight. Keep standingprelude/formalquiet scorer unchanged. Safeinitial noautoreset rollout should retain failureprefix+abort, not quietly drop transition or bootstrapfall.
Throughput: actualprior32 neutral1000controls8000steps585s session,600s hosttimeout distinct;contactsJSON4.620GB andprefixcopieswholeactivefile. Local losslesscodec parity exists but notadopted; no measurednative speedup. Formalacceptance raw/evidence versus futuretrainingtelemetry require explicit recordingcontract not silentdownsampling. AfterCPUchanges avoid rerunningnativejustforpackaging. Evaluate exact checkpoint withfulloriginalformalquiet10s andallbearingheldouts, separate from2updateintegration.
Please identify up to4 concrete implementation bugs/redflags, smallest useful next native job aftersmoke, and explicitly resolve movingcontactpredicate vs trainingtelemetry design. No invented thresholds or nativeadmissions. Are our proposed boundedCPU seams enough to make nextpilot honest; what missingmeasurements block it? Avoid genericRLsurvey. Below exact selected code for causal/reward/command semantics and inputhashes.\n'''
for name in ['commands.py','causal_command.py','objective.py']:
 prompt+='\nFILE '+files[name]+'\n'+(R/'oracles'/name).read_text()
prompt+='\nINPUT_HASHES\n'+json.dumps(inputs)
(R/'fable/prompt.md').write_text(prompt)
