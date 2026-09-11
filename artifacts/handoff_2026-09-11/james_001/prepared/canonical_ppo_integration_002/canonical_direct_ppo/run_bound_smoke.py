"""Composition seam called by a separately bound native standing initializer."""
from pathlib import Path
import inspect
from .binding_contract import verify_admissions,bind_consumer_lineage,read,sha


def run(session,model,standing_source,standing_one,standing32,bindings_file,output,*,device,native_errors=None):
    # This barrier precedes importing Torch/PPO or constructing a new actor.
    bindings=read(bindings_file)
    lineage=bind_consumer_lineage(verify_admissions(standing_source,standing_one,standing32,bindings))
    runtime_file=Path(inspect.getsourcefile(type(session))).resolve()
    expected=Path(standing_source).resolve()/'standing_session.py'
    if runtime_file!=expected or sha(runtime_file)!=read(Path(standing_source)/'FREEZE_SHA256.json')['standing_session.py']:raise ValueError('Session does not use exact admitted native runtime')
    if session.failure is not None:raise ValueError('Cannot learn after a native session failure')
    from .native_bridge import CanonicalNativeBridge
    from .runner import learn_smoke
    bridge=CanonicalNativeBridge(session,model,device=device,native_errors=native_errors)
    return learn_smoke(bridge,lineage,output,device=device)
