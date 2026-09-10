"""Source-bound, isolated lazy prediction successor; no old module mutation.

Keep the exact parent step except skipping side-effect-free endpoint prediction
when no active row launches. The198-rounding recurrence remains unchanged for
all launch-containing batches. This has a new identity and no policy admission.
"""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,textwrap
PARENT_FREEZE='5f46b6f99c172bb037004e758f4f3715d41da42725abd10bfd313ee2b84fe286'
PARENT_SOURCE='784d25a7598b21acb2633f27355ba5074023af78d790a36bed89d99094466637'
OLD='''        predicted_p,predicted_R=self._predict(target)
        endpoint=(predicted_R@self.s['neutral'][self.ids,nextleg,:,None]).squeeze(-1)+predicted_p
        endpoint=torch.cat((endpoint[:,:2],self.s['anchors'][self.ids,nextleg,2,None]),-1)
        start=self.s['anchors'][self.ids,nextleg];duration=torch.full_like(self.s['time'],2.)
        coeff=self._coeff(start,endpoint,duration);hcoeff=self._coeff(start,endpoint,duration*.8)
        launch=self.active&liftoff
'''
NEW='''        launch=self.active&liftoff
        if bool(launch.any()):
            predicted_p,predicted_R=self._predict(target)
            endpoint=(predicted_R@self.s['neutral'][self.ids,nextleg,:,None]).squeeze(-1)+predicted_p
            endpoint=torch.cat((endpoint[:,:2],self.s['anchors'][self.ids,nextleg,2,None]),-1)
            start=self.s['anchors'][self.ids,nextleg];duration=torch.full_like(self.s['time'],2.)
            coeff=self._coeff(start,endpoint,duration);hcoeff=self._coeff(start,endpoint,duration*.8)
        else:
            # Every subsequent trajectory update is masked by launch. Reuse the
            # existing shape/dtype-correct values; none can alter a row's state.
            endpoint=self.s['sw_end'];duration=self.s['sw_duration']
            coeff=self.s['sw_coeff'];hcoeff=self.s['sw_hcoeff']
'''

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def build_classes(parent_root):
    root=Path(parent_root).resolve();manifest=root/'FREEZE_SHA256.json'
    if sha(manifest)!=PARENT_FREEZE or sha(root/'batch_wave.py')!=PARENT_SOURCE:raise ValueError('Exact frozen wave005 parent required')
    wanted=json.loads(manifest.read_text())
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('No symbolic parent substitution')
    found={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()};found.pop('FREEZE_SHA256.json')
    if found!=wanted:raise ValueError('Parent dependency changed')
    loaded=sys.modules.get('tensor_kernel')
    if loaded is not None and Path(loaded.__file__).resolve()!=root/'parent/kernel/tensor_kernel.py':raise ValueError('Wrong geometry already imported')
    spec=importlib.util.spec_from_file_location('_frozen_lazy_wave005_parent',root/'batch_wave.py')
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    source=(root/'batch_wave.py').read_text();tree=ast.parse(source)
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='BatchWave005')
    node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='step')
    lines=source.splitlines(True);original=''.join(lines[node.lineno-1:node.end_lineno])
    if original.count(OLD)!=1:raise ValueError('Frozen step anchor differs')
    changed=original.replace(OLD,NEW)
    namespace=dict(parent.__dict__);exec(compile(textwrap.dedent(changed),str(Path(__file__).resolve())+'::bound_lazy_step','exec'),namespace)
    class LazyPredictionWave005(parent.BatchWave005):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.identity.update(implementation='batched_wave005_lazy_prediction_prototype_v1',
                lazy_successor_sha256=sha(Path(__file__).resolve()),parent_batch_sha256=PARENT_SOURCE,
                parent_freeze_sha256=PARENT_FREEZE,policy_training_allowed=False)
        step=namespace['step']
    return parent.BatchWave005,LazyPredictionWave005,parent,original,changed
