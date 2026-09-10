"""A declared numerical candidate, not an identified motor or native admission."""
import json
from pathlib import Path
import numpy as np

def candidate(M,dt=.0025):
 M=np.asarray(M,dtype=float)
 if M.shape!=(18,18)or not np.isfinite(M).all()or not np.isfinite(dt)or dt<=0:
  raise ValueError('Invalid rigid-link model')
 np.linalg.cholesky(M)
 effective=1/np.diag(np.linalg.inv(M))
 # 0.581644 Nm ideal static load / 0.05 rad declared error budget = 11.633.
 # Round upward to 12; do not represent this design target as a hardware tolerance.
 kp=np.full(18,12.);kd=2*.8*np.sqrt(kp*effective)
 A=np.linalg.solve(M,np.diag(kp));B=np.linalg.solve(M,np.diag(kd))
 S=np.block([[np.eye(18)-dt*dt*A,dt*(np.eye(18)-dt*B)],[-dt*A,np.eye(18)-dt*B]])
 return {'stiffness_nm_per_rad':kp.tolist(),'damping_nm_s_per_rad':kd.tolist(),
  'dt_s':dt,'declared_static_error_design_budget_rad':.05,'declared_damping_ratio_heuristic':.8,
  'linear_fixed_root_semiimplicit_euler_spectral_radius':float(max(abs(np.linalg.eigvals(S)))),
  'scope':'Zero-pose rigid-link fixed-root linear numerical candidate only. No motor armature, native contact/solver, delay, saturation or floating-root stability proof.'}

if __name__=='__main__':
 base=Path(__file__).resolve().parent;r=json.loads((base/'inertia_report.json').read_text())
 output=base/'servo_candidate.json'
 if output.exists():raise SystemExit('Preserve existing candidate receipt')
 output.write_text(json.dumps({'joint_names':r['joint_names'],**candidate(r['mass_matrix_kg_m2'])},indent=2)+'\n')
