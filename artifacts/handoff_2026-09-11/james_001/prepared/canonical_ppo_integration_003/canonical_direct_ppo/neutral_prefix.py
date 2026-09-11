"""Immutable neutral acquisition snapshot without closing or changing the native session."""
from pathlib import Path
import json,os,shutil
import numpy as np


def snapshot(session,destination,score,save):
    if session.n!=32 or session.count!=8000 or session.captured_count!=8000 or session.control!=1000 or session.reset_count!=1 or session.failure is not None:
        raise ValueError('Fresh complete32 neutral prefix required before policy')
    if len(session.controls)!=1000:raise ValueError('Incomplete neutral control rows')
    from .source005_solver_diagnostics import declaration,validate_legacy
    from .solver_evidence import validate_solver
    validate_solver(json.loads((Path(session.output)/'solver_readback.json').read_text()),session.n,final=False)
    validate_legacy(json.loads((Path(session.output)/'legacy_friction_readback.json').read_text()),session.n,session.names)
    session.flush() # flushes the active contact stream, but never closes or resets it
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    source=Path(session.output)
    for name in ['initial_reset.json','native_readback.json','contacts.jsonl','solver_readback.json','legacy_friction_readback.json','contact_view.json']:
        # The active contact stream will grow. Its prefix must be an independent copy.
        shutil.copyfile(source/name,dest/name)
    for name in session.files:
        if Path(name).name!=name or not name.startswith('substeps_')or not name.endswith('.npz'):raise ValueError('Unexpected raw chunk name')
        # Flushed native chunks are immutable; subsequent steps create new chunks.
        try:os.link(source/name,dest/name)
        except OSError:shutil.copyfile(source/name,dest/name)
    rows=[{k:np.asarray(v).copy()for k,v in r.items()}for r in session.controls]
    np.savez_compressed(dest/'control_trace.npz',**{k:np.stack([r[k]for r in rows])for k in rows[0]})
    save(dest/'session.json',{'steps':session.count,'controls':session.control,'reset_count':session.reset_count,
         'failure':session.failure,'substep_files':list(session.files),'body_names':list(session.body_names),
         'joint_names':list(session.names),'root_paths':list(session.roots),'captured_steps':session.captured_count,
         'all_rows_recorded':session.captured_count==session.count,'solver_diagnostics':declaration(session.n,session.captured_count),'scope':'Snapshot before actor construction; original session remains live.'})
    report=score(dest);save(dest/'standing_report.json',report)
    from .binding_contract import sha
    inventory={str(p.relative_to(dest)):sha(p)for p in sorted(dest.rglob('*'))if p.is_file()}
    save(dest/'SHA256.json',inventory)
    if report.get('all_pass')is not True or report.get('num_envs')!=32 or len(report.get('replicas',[]))!=32 or any(r.get('pass')is not True for r in report['replicas']):
        raise ValueError('Fresh neutral prefix rejected; actor was not constructed')
    return {'standing_report_sha256':sha(dest/'standing_report.json'),'inventory_sha256':sha(dest/'SHA256.json'),
            'controls':1000,'substeps':8000,'standing_pass':True,'session_closed':False,'additional_reset':False}
