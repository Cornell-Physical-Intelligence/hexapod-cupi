"""Versioned native-entry composition; exact frozen initialization and servo loop retained."""
from pathlib import Path
from contextlib import contextmanager
import ast,hashlib,sys

STANDING_FREEZE='f81f61d257cb92b65ad80f3b7d27c45497e66254092e244a32da02555b11e2a4'
ORIGINAL_TAIL='''  session.close();session=None
  from standing_score import score
  report=score(out);save(out/'standing_report.json',report)
  state['standing_pass']=report['all_pass'];state['status']='completed';state['native_error_events']=errors;state['wall_s']=time.monotonic()-started
  print(f'CANONICAL_STANDING_ACQUISITION_COMPLETED standing_pass={report["all_pass"]} training_allowed=false',flush=True)'''
LEARNER_TAIL='''  learner_callback(session,model,out,state,errors,a)
  state['explicit_steps_completed']=session.count
  session.close();session=None
  state['status']='completed';state['native_error_events']=errors;state['wall_s']=time.monotonic()-started
  print('CANONICAL_PPO_SMOKE_COMPLETED updates=2 quality_admitted=false',flush=True)'''
HELPERS=('standing_contract','native_support','inspect_core','standing_math','standing_session','standing_score','quiet_metrics')


def adapted_source(root):
    from .binding_contract import verify_directory,sha
    root=Path(root).resolve();manifest=verify_directory(root,STANDING_FREEZE)
    original=(root/'run_standing.py').read_text()
    if sha(root/'run_standing.py')!=manifest['run_standing.py']or original.count(ORIGINAL_TAIL)!=1:raise ValueError('Exact frozen native-entry seam changed')
    adapted=original.replace(ORIGINAL_TAIL,LEARNER_TAIL,1)
    if adapted.replace(LEARNER_TAIL,ORIGINAL_TAIL,1)!=original:raise ValueError('Unexpected native source adaptation')
    # Prefix, loop and exception/finally statements remain byte identical. No rewritten physics step.
    ast.parse(adapted)
    return adapted,{'schema':'canonical_native_ppo_entry_composition_v1','standing_source_sha256':STANDING_FREEZE,
        'original_entry_sha256':manifest['run_standing.py'],'adapted_entry_sha256':hashlib.sha256(adapted.encode()).hexdigest(),
        'changed_seam':'post1000 scoring tail only; fresh-prefix admission,48 policy controls,then close',
        'neutral_physics_statements_unchanged':True,'exception_finally_statements_unchanged':True}


@contextmanager
def exact_helper_imports(root):
    root=Path(root).resolve();old_path=list(sys.path)
    for name in HELPERS:
        present=sys.modules.get(name)
        if present is not None and Path(getattr(present,'__file__','')).resolve()!=root/(name+'.py'):
            raise ValueError('Foreign cached native helper:'+name)
    sys.path.insert(0,str(root))
    try:yield
    finally:sys.path[:]=old_path


def load(root,verify_inputs,finish,callback,schema):
    root=Path(root).resolve();source,receipt=adapted_source(root)
    with exact_helper_imports(root):
        namespace={'__name__':'_canonical_ppo_native_entry','__file__':str(root/'run_standing.py')}
        # Compile the full verified module, avoiding isolated-function/compiler-context mismatch.
        exec(compile(source,str(root/'run_standing.py'),'exec'),namespace)
    namespace.update(verify_inputs=verify_inputs,finish=finish,learner_callback=callback,SCHEMA=schema)
    return namespace['main'],receipt
