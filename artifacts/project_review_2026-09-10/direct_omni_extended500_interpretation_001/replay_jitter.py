"""Portable location adapter for the byte-preserved independent jitter analysis.

Only its checkout-root assignment is rebound. Signal analysis is unchanged.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = Path(__file__).resolve().parent / 'jitter_review/analyze.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '2715d8195ed51da457992d5be5b73c3fd8d18548cad3ebc4cc1931f6d45ccaa7'
tree = ast.parse(source.read_text(), filename=str(source))
bindings = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'ROOT']
assert len(bindings) == 1
assert ast.dump(bindings[0].value) == ast.dump(ast.parse('Path(__file__).resolve().parents[2]', mode='eval').body)
bindings[0].value = ast.parse('Path(' + repr(str(a.repo_root.resolve())) + ')', mode='eval').body
ast.fix_missing_locations(tree)
sys.argv = [str(source), '--output', str(a.output.resolve())]
exec(compile(tree, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})
