#!/usr/bin/env python3
"""Generate a source map from a Git revision without importing project code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote


PACKAGES = {'core', 'backends', 'tasks', 'learning', 'evaluation', 'runs', 'priors', 'surrogate'}


def area(path):
    parts = Path(path).parts
    return parts[1] if len(parts) > 2 and parts[1] in PACKAGES else 'entry points and experiments'


def generate(repo, ref="HEAD"):
    repo = Path(repo)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    revision = git("rev-parse", ref + "^{commit}").decode().strip()
    read = lambda path: (repo/path).read_bytes()
    ownership = {r["path"]: r for r in json.loads(read("configs/source_inventory.json"))["tools"]}
    paths = sorted(p.relative_to(repo).as_posix() for p in (repo/'locomotion').rglob('*.py')
                   if '__pycache__' not in p.parts and 'fixtures' not in p.parts)
    url = "https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/" + revision + "/"
    rows = []
    for path in paths:
        raw = read(path)
        text = raw.decode()
        tree = ast.parse(text, filename=path)
        meta = ownership.get(path, {})
        symbols = []

        def visit(node, parents=()):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = (*parents, child.name)
                    is_class = isinstance(child, ast.ClassDef)
                    symbols.append({
                        "name": ".".join(names), "kind": "class" if is_class else "function",
                        "line": child.lineno, "end_line": child.end_lineno,
                        "signature": ", ".join(ast.unparse(v) for v in child.bases) if is_class else ast.unparse(child.args),
                        "doc": (ast.get_docstring(child) or "").split("\n\n")[0].replace("\n", " "),
                        "url": url + quote(path, safe="/") + "#L" + str(child.lineno),
                    })
                    visit(child, names)
                else:
                    visit(child, parents)

        visit(tree)
        imports = [{"line": n.lineno, "text": ast.unparse(n)} for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        dynamic = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                          and n.value.startswith("locomotion.") and len(n.value) < 160})
        rows.append({"path": path, "area": area(path), "test": "/tests/" in path, "compatibility": (ast.get_docstring(tree) or "").startswith("Compatibility"),
                     "lines": len(text.splitlines()), "sha256": hashlib.sha256(raw).hexdigest(),
                     "owner": meta.get("owner", "Test module"), "classification": meta.get("classification", "test"),
                     "purpose": meta.get("purpose", (ast.get_docstring(tree) or "").split("\n\n")[0]),
                     "symbols": symbols, "imports": sorted(imports, key=lambda r: r["line"]),
                     "dynamic_references": dynamic, "url": url + quote(path, safe="/")})
    missing = [r["path"] for r in rows if not r["test"] and r["path"] not in ownership]
    if missing:
        raise ValueError("Production modules need inventory entries: " + repr(missing))
    totals = {}
    for label, test in (("production", False), ("tests", True)):
        selected = [r for r in rows if r["test"] == test]
        totals[label] = {"files": len(selected), "physical_lines": sum(r["lines"] for r in selected),
                         "classes": sum(s["kind"] == "class" for r in selected for s in r["symbols"]),
                         "functions_and_methods": sum(s["kind"] == "function" for r in selected for s in r["symbols"])}
    return {"schema": "hexapod_code_map_v1", "revision": revision, "working_tree_changed": bool(git("status", "--porcelain", "--", "locomotion", "configs/source_inventory.json")), "totals": totals,
            "scope": "Static definitions and imports, including nested definitions. Import strings and factory expressions can remain unresolved. This is not a runtime call graph or a test-coverage report.", "modules": rows}


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hexapod · Locomotion code</title>
<link rel="stylesheet" href="../poster.css">
<style>
.filters {display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:24px 0}
.filters input[type=search] {flex:1;min-width:220px}
.filters input[type=search],.filters select {font:inherit;color:inherit;background:transparent;border:1px solid var(--line);padding:10px 12px}
.filters input::placeholder {color:var(--ink);opacity:1}
.filters label {display:flex;align-items:center;gap:8px}
.module {padding:14px 0;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
.module summary {cursor:pointer;font-weight:700;line-height:1.55}
.module > p {margin:12px 0}
.definitions {margin:12px 0;padding-left:24px;columns:2;column-gap:32px}
.definitions li {padding:5px 0;break-inside:avoid;line-height:1.55}
@media(max-width:760px) {.definitions {columns:1}}
</style>
</head>
<body>
<a class="skip" href="#code">Skip to the code</a>
<header class="masthead"><a class="brand" href="../"><span class="brand-mark" aria-hidden="true">H/6</span><span>Hexapod · Cornell Physical Intelligence</span></a><nav aria-label="Pages"><a href="../">Progress</a><a href="#code" aria-current="page">Code</a></nav><a class="repo-link" href="https://github.com/Cornell-Physical-Intelligence/hexapod-cupi">GitHub ↗</a></header>
<main id="code">
<section class="intro"><h1>Locomotion code map</h1><p>Choose a file to see its classes and functions, or search for a name.</p></section>
<div class="filters" role="search" aria-label="Code map filters"><input id="query" type="search" aria-label="Search code map" placeholder="Search code"><select id="area" aria-label="Package"><option value="">All packages</option></select><label><input type="checkbox" id="tests"> Tests</label><label><input type="checkbox" id="compatibility"> Compatibility imports</label></div>
<p id="count" role="status"></p>
<div id="results"></div>
<footer class="section"><p id="revision"></p><p><a href="code-map.json">Download the source index</a></p></footer>
</main>
<script>const DATA=__DATA__;
const PACKAGES=[
['core','Core','Motor arithmetic, transforms and features.'],
['backends','Isaac backend','Scene construction and environment stepping.'],
['tasks','Tasks','Episode handling, reward versions and prescribed control.'],
['learning','Learning','Learner configuration and adapters.'],
['evaluation','Evaluation','Capture, scoring and diagnostics.'],
['runs','Runs','Validation, packaging and source identity.'],
['priors','Priors','Motion data and AMP datasets.'],
['surrogate','Surrogate','MuJoCo training and evaluation.'],
['entry points and experiments','Entry points and experiments','Command entry points and compatibility imports.']
];
const el=id=>document.getElementById(id), esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link=(url,text)=>`<a href="${esc(url)}">${esc(text)}</a>`;
el('revision').innerHTML=link(`https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/tree/${DATA.revision}`,`Source ${DATA.revision.slice(0,8)}`)+(DATA.working_tree_changed?' · Local preview includes uncommitted changes.':'');
for(const [value,name] of PACKAGES){const option=document.createElement('option');option.value=value;option.textContent=name;el('area').append(option)}
function render(){const query=el('query').value.trim().toLowerCase(), chosen=el('area').value, tests=el('tests').checked, compatibility=el('compatibility').checked;
const matches=(m)=>[m.path,m.purpose,...m.symbols.map(s=>s.name+' '+s.doc)].join(' ').toLowerCase().includes(query);
const rows=DATA.modules.filter(m=>(tests||!m.test)&&(compatibility||!m.compatibility)&&(!chosen||m.area===chosen)&&(!query||matches(m)));
el('count').textContent=query?`${rows.length} files match “${el('query').value.trim()}”`:'';
el('results').innerHTML=PACKAGES.map(([area,title,purpose])=>{const modules=rows.filter(m=>m.area===area);if(!modules.length)return '';
return `<section class="section"><h2>${esc(title)}</h2><p>${esc(purpose)}</p>${modules.map(m=>{const symbols=m.symbols.filter(s=>!query||m.path.toLowerCase().includes(query)||s.name.toLowerCase().includes(query)||s.doc.toLowerCase().includes(query));
return `<details class="module"${query?' open':''}><summary>${esc(m.path.replace('locomotion/',''))}</summary><p>${link(m.url,'Open source file')}</p>${symbols.length?`<ul class="definitions">${symbols.map(s=>`<li>${link(s.url,s.name)}</li>`).join('')}</ul>`:''}</details>`}).join('')}</section>`}).join('')||'<p class="section">No matching modules.</p>'}
for(const id of ['query','area','tests','compatibility'])el(id).addEventListener('input',render);render();
</script>
</body>
</html>'''


def build(repo, output, ref='HEAD'):
    data = generate(repo, ref)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'code-map.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    (output/'index.html').write_text(PAGE.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = build(args.repo, args.output)
    print(json.dumps({'revision': data['revision'], 'totals': data['totals']}))


if __name__ == '__main__':
    main()
