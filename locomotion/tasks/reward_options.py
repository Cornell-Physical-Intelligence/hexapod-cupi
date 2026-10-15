"""Check reward option keys, value types and word choices with the standard library.

``locomotion.runs.prepare`` runs on hosts without Torch, and each reward task module imports it. This
module reads the selected task's reward configuration from its source instead: the annotated fields
of the dataclass and of its bases, and each ``self.<field> not in (...)`` choice in ``validate``.
A value converts with its field's type, as the task's ``reward_config`` does. Numeric bounds and
the other checks of ``validate`` stay with the trainer, which applies ``reward_config`` before launch.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
TASKS = {"v4": ("task_v4", "RewardV4Config"), "v4_yaw_gate": ("task_v4_yaw_gate", "RewardV4YawGateConfig")}
TYPES = {"float": float, "int": int, "str": str}


def _class(module, name):
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def schema(task):
    """``({field: type}, {field: allowed words})`` of the selected task's reward configuration."""
    if task not in TASKS:
        raise ValueError(f"Unknown reward task {task!r}")
    types, choices, pending = {}, {}, [TASKS[task]]
    while pending:
        module, name = pending.pop()
        node = _class(module, name)
        for base in node.bases:
            # A base is a class of the same module or ``module.Class`` of a sibling module.
            pending.append((base.value.id, base.attr) if isinstance(base, ast.Attribute) else (module, base.id))
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                annotation = ast.unparse(item.annotation)
                if annotation not in TYPES:
                    raise ValueError(f"{name}.{item.target.id} has an unsupported type {annotation}")
                types.setdefault(item.target.id, TYPES[annotation])
            if isinstance(item, ast.FunctionDef) and item.name == "validate":
                for compare in ast.walk(item):
                    if (isinstance(compare, ast.Compare) and isinstance(compare.left, ast.Attribute)
                            and isinstance(compare.left.value, ast.Name) and compare.left.value.id == "self"
                            and len(compare.ops) == 1 and isinstance(compare.ops[0], ast.NotIn)
                            and isinstance(compare.comparators[0], ast.Tuple)
                            and all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                                    for e in compare.comparators[0].elts)):
                        choices[compare.left.attr] = {e.value for e in compare.comparators[0].elts}
    return types, choices


def check(task, spec):
    """Raise ``ValueError`` unless each ``key=value`` names a field once and converts to its type and choices."""
    types, choices = schema(task)
    seen = set()
    for item in filter(None, spec.split(",")):
        key, separator, value = item.partition("=")
        if not separator or key not in types or key in seen:
            raise ValueError(f"Reward options are unique fields of the {task} reward configuration, got {item!r}")
        seen.add(key)
        converted = types[key](value)
        if key in choices and converted not in choices[key]:
            raise ValueError(f"{key} is one of {sorted(choices[key])}, got {value!r}")
