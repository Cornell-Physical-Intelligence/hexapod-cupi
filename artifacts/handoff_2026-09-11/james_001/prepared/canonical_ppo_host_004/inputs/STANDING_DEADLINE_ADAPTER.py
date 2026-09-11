"""Exact source009 run_owned adaptation for canonical standing only.

Only two 600-second expressions and the corresponding misleading timeout text
change. Cleanup, locks, ownership, stop handling and the 90-second AppReady
bound keep their original AST. No frozen supervisor file is written.
"""
import ast
import hashlib
from pathlib import Path
import types

SCHEMA = 'canonical_standing_supervisor_deadline_v1'
SUPERVISOR_SHA256 = '9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
PHASES = ('standing',)
HELPER_NAME = '_canonical_standing_phase_deadline_seconds'
SEAMS = (
    ('deadline_seconds=600', 'deadline_seconds=_canonical_standing_phase_deadline_seconds(args, phase)'),
    ('deadline = time.monotonic() + 600', 'deadline = time.monotonic() + _canonical_standing_phase_deadline_seconds(args, phase)'),
    ('raise TimeoutError("Standing phase exceeded ten-minute bound")',
     'raise TimeoutError(f"{phase} phase exceeded {_canonical_standing_phase_deadline_seconds(args, phase)}-second bound")'),
)


def phase_deadline_seconds(args, phase):
    if type(getattr(args, 'num_envs', None)) is not int or args.num_envs not in (1, 32) or phase not in PHASES:
        raise ValueError('Deadline adapter is restricted to explicit standing1 or32')
    return 1200


def inspect_supervisor(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SUPERVISOR_SHA256:
        raise ValueError('Deadline adapter requires the exact frozen source009 supervisor')
    tree = ast.parse(data.decode('utf-8'), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_owned']
    if len(nodes) != 1:
        raise ValueError('Expected one exact run_owned function')
    original = ast.get_source_segment(data.decode('utf-8'), nodes[0])
    adapted = original
    for before, after in SEAMS:
        if adapted.count(before) != 1:
            raise ValueError('Required deadline source seam is absent or ambiguous: ' + before)
        adapted = adapted.replace(before, after, 1)
    # Reversing all three exact substitutions must restore every original byte
    # of the function; this includes every cleanup/ownership/lock statement.
    reverse = adapted
    for before, after in reversed(SEAMS):
        if reverse.count(after) != 1:
            raise ValueError('Ambiguous adapted source seam')
        reverse = reverse.replace(after, before, 1)
    if reverse != original:
        raise ValueError('Unexpected source delta outside three deadline seams')
    original_ast = ast.parse(original).body[0]
    adapted_ast = ast.parse(adapted).body[0]
    # Exact finally AST equality is independently enforced, beyond reversal.
    original_try = next(n for n in original_ast.body if isinstance(n, ast.Try))
    adapted_try = next(n for n in adapted_ast.body if isinstance(n, ast.Try))
    if ast.dump(ast.Module(body=original_try.finalbody, type_ignores=[])) != ast.dump(ast.Module(body=adapted_try.finalbody, type_ignores=[])):
        raise ValueError('Cleanup AST changed')
    return {
        'schema': SCHEMA,
        'supervisor_sha256': SUPERVISOR_SHA256,
        'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_run_owned_sha256': hashlib.sha256(original.encode()).hexdigest(),
        'adapted_run_owned_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'source_substitutions': 3,
        'phase_deadline_seconds': {phase: 1200 for phase in PHASES},
        'app_ready_deadline_seconds': 90,
        'cleanup_AST_unchanged': True,
        'original_source_unchanged': True,
        'original_function_source': original,
        'adapted_function_source': adapted,
        '_original_module_source': data.decode('utf-8'),
    }


def install(module, path):
    path = Path(path).resolve()
    proof = inspect_supervisor(path)
    if Path(module.__file__).resolve() != path:
        raise ValueError('Supervisor module origin differs from verified source')
    function = module.run_owned
    if not isinstance(function, types.FunctionType) or function.__globals__ is not vars(module):
        raise ValueError('Supervisor run_owned is not in its own module globals')
    # Compile the full verified module without executing it, matching Python's
    # normal import compilation context. Isolated AST compilation is not
    # CodeType-identical on Python 3.12. No imports or statements execute here.
    code = compile(proof['_original_module_source'], str(path), 'exec', dont_inherit=True)
    expected = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'run_owned')
    if function.__code__ != expected or function.__defaults__ or function.__kwdefaults__ or function.__closure__:
        raise ValueError('Loaded run_owned differs from exact verified source')
    if HELPER_NAME in vars(module) or 'CANONICAL_DEADLINE_ADAPTER' in vars(module):
        raise ValueError('Deadline adapter already installed or global name collision')
    adapted_code = compile(proof['adapted_function_source'], str(path) + '::' + SCHEMA, 'exec')
    body = next(c for c in adapted_code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'run_owned')
    replacement = types.FunctionType(body, vars(module), function.__name__)
    replacement.__module__ = function.__module__
    replacement.__doc__ = function.__doc__
    metadata = {k: v for k, v in proof.items() if k not in ('original_function_source', 'adapted_function_source', '_original_module_source')}
    # Install only after all source/code checks succeed.
    setattr(module, HELPER_NAME, phase_deadline_seconds)
    module.CANONICAL_DEADLINE_ADAPTER = metadata
    module.run_owned = replacement
    return metadata
