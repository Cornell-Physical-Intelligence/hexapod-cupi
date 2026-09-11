#!/usr/bin/env python3
"""List owned tools and check the maintained-source boundary without scanning evidence."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = 'configs/source_inventory.json'
SCOPES = ('tools', 'experiments/c_length_study/tools', 'experiments/terrain/tools')


def read_inventory(root=ROOT):
    return json.loads((root / INVENTORY).read_text())


def check(root=ROOT):
    inventory = read_inventory(root)
    rows = inventory['tools']
    paths = [row['path'] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate source ownership')
    actual = {str(p.relative_to(root)) for directory in SCOPES
              for p in (root / directory).rglob('*.py') if '__pycache__' not in p.parts}
    if actual != set(paths):
        raise ValueError(f'Source inventory differs: unowned={sorted(actual-set(paths))}; missing={sorted(set(paths)-actual)}')
    for row in rows:
        path = root / row['path']
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source must be a regular in-repository file')
        if row['classification'] not in {'current', 'historical', 'prototype'} or not row['owner'] or not row['purpose']:
            raise ValueError('Every source needs classification, ownership and purpose')
    # Production package imports may never consume frozen evidence copies.
    for path in (root / 'packages').rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            if any(name == 'artifacts' or name.startswith('artifacts.') for name in names):
                raise ValueError(f'Production package imports evidence: {path.relative_to(root)}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'check'])
    parser.add_argument('--classification', choices=['current', 'historical', 'prototype'])
    args = parser.parse_args()
    rows = check()
    if args.command == 'check':
        print(f'Ownership and paths verified for {len(rows)} tools; production imports exclude artifact modules')
    else:
        for kind in ['current', 'prototype', 'historical']:
            if args.classification and args.classification != kind:
                continue
            print(kind.upper())
            for row in rows:
                if row['classification'] == kind:
                    print(f"  {row['path']} — {row['owner']}: {row['purpose']}")


if __name__ == '__main__':
    main()
