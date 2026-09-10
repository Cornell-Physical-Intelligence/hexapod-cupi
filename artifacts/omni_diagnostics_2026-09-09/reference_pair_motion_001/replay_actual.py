"""Recompute actual paired result in a temporary copy; preserve every frozen byte."""
from pathlib import Path
import tempfile,shutil,subprocess,sys,json
H=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as td:
 root=Path(td);work=root/'actual_review';shutil.copytree(H/'actual_review',work)
 # Raw inputs stay immutable; only derived files in this temporary review copy change.
 (root/'run').symlink_to((H/'run').resolve(),target_is_directory=True)
 shutil.copy2(H/'portable_replay/omni_diagnostics.py',work/'oracle/omni_diagnostics.py')
 result=subprocess.run([sys.executable,'-B',str(work/'analyze.py')],capture_output=True,text=True)
 if result.returncode:raise RuntimeError(result.stdout+result.stderr)
 computed=json.loads((work/'report.json').read_text());original=json.loads((H/'actual_review/report.json').read_text())
 assert computed['independent_gate']['passed'] is False and computed['standing_quiet_replay']['passed'] is True
 assert computed['independent_gate']['full_state_numerical_replay']['passed'] is True
 assert computed['independent_gate']['full_state_numerical_replay']['reference_controls_replayed']==2200
 assert computed['independent_gate']['full_state_numerical_replay']['poststep_checks_independently_replayed']==2200
 assert abs(computed['independent_gate']['independent_forward_motion']['displacement_integral_difference_m']-original['independent_gate']['independent_forward_motion']['displacement_integral_difference_m'])<1e-8
 assert computed['400Hz']['samples']==19201 and computed['400Hz']['passed'] is True
 print(json.dumps({'replayed_controller_and_poststep_checks':2200,'substep_samples':19201,'standing32quiet':True,'paired_motion_admitted':False,
  'original5mm_rejection_preserved':True,'frozen_raw_or_review_modified':False},indent=2))
