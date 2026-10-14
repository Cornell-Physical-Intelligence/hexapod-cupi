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
        rows.append({"path": path, "area": area(path), "test": "/tests/" in path,
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


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hexapod locomotion code map</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#18232e;font:16px/1.55 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:38px 24px}a{color:#145e97}h1{font-size:32px;margin:4px 0}small,.muted{color:#536779}code{font:13px ui-monospace,monospace;overflow-wrap:anywhere}nav{position:sticky;top:0;background:#f3f5f7;padding:14px 0;display:flex;gap:12px;flex-wrap:wrap;align-items:center;z-index:1}input[type=search]{flex:1;min-width:240px}input,select{padding:10px;border:1px solid #a6b6c7;border-radius:6px;font:inherit}article{background:white;margin:14px 0;border:1px solid #d4dde5;border-radius:10px;padding:20px}h2{font-size:18px;margin:0 0 6px}.badge{font-size:12px;border-radius:4px;background:#e4edf4;padding:3px 6px}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;vertical-align:top;border-bottom:1px solid #e0e6ec}td:first-child{width:38%}.imports{white-space:pre-wrap;font:13px/1.65 ui-monospace,monospace;overflow-wrap:anywhere}details{margin:12px 0}footer{margin:30px 0}#empty{padding:25px;background:#fff}a:focus,input:focus,select:focus,summary:focus{outline:3px solid #db8c24;outline-offset:3px}@media(max-width:700px){td,th{display:block;width:100%!important}th{display:none}}@media print{nav{position:static}article{break-inside:avoid}}
</style><main><a href="../">Project progress</a><h1>Locomotion code map</h1><p id="meta"></p><p class="muted">Search a class, function, filename or purpose. Source links point to the build commit. You can include tests and inspect imports without loading Isaac, Torch or MuJoCo.</p><nav aria-label="Code map filters"><input id="query" type="search" aria-label="Search code map" placeholder="Search, for example ProximityGuard"><select id="area" aria-label="Area"><option value="">All areas</option></select><label><input type="checkbox" id="tests"> Include tests</label><label><input type="checkbox" id="functions"> Include functions and methods</label></nav><p id="count" aria-live="polite"></p><div id="results"></div><footer id="scope" class="muted"></footer></main>
<script>const DATA=__DATA__;
const el=id=>document.getElementById(id), esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link=(url,text)=>`<a href="${esc(url)}">${esc(text)}</a>`;
const t=DATA.totals.production;el('meta').textContent=`${DATA.revision.slice(0,8)}${DATA.working_tree_changed?' (local changes)':''} · ${t.files} production modules · ${t.classes} class definitions · ${t.functions_and_methods} functions and methods · ${t.physical_lines.toLocaleString()} physical lines`;
el('scope').textContent=DATA.scope;
for(const name of [...new Set(DATA.modules.map(m=>m.area))].sort()){const option=document.createElement('option');option.value=name;option.textContent=name;el('area').append(option)}
function render(){const query=el('query').value.trim().toLowerCase(), chosen=el('area').value, tests=el('tests').checked, functions=el('functions').checked;
const rows=DATA.modules.filter(m=>(tests||!m.test)&&(!chosen||m.area===chosen)&&(!query||[m.path,m.purpose,m.owner,...m.symbols.map(s=>s.name+' '+s.doc)].join(' ').toLowerCase().includes(query)));
el('count').textContent=`${rows.length} modules shown`;
el('results').innerHTML=rows.map(m=>{const symbols=m.symbols.filter(s=>(functions||s.kind==='class'||query)&&(!query||m.path.toLowerCase().includes(query)||s.name.toLowerCase().includes(query)||s.doc.toLowerCase().includes(query)||m.purpose.toLowerCase().includes(query)));
return `<article><span class="badge">${esc(m.area)}${m.test?' · test':''}</span><h2>${link(m.url,m.path)}</h2><p>${esc(m.purpose)}</p><p class="muted">${esc(m.owner)} · ${esc(m.classification)} · ${m.lines} physical lines</p>${symbols.length?`<table><thead><tr><th>Definition</th><th>Source docstring</th></tr></thead><tbody>${symbols.map(s=>`<tr><td><code>${link(s.url,s.name)}</code><br><small>${s.kind} · line ${s.line}</small></td><td>${esc(s.doc||'No docstring.')}<details><summary>Signature or bases</summary><code>${esc(s.signature)}</code></details></td></tr>`).join('')}</tbody></table>`:'<p class="muted">No matching class definitions. Enable functions to inspect the functional API.</p>'}<details><summary>Imports and declared dynamic references</summary><div class="imports">${esc(m.imports.map(i=>i.line+': '+i.text).join('\n'))}</div><div class="imports">${esc(m.dynamic_references.join('\n'))}</div></details><details><summary>Source identity</summary><code>${esc(m.sha256)}</code></details></article>`}).join('')||'<p id="empty">No matching modules.</p>'}
for(const id of ['query','area','tests','functions'])el(id).addEventListener('input',render);render();
</script></html>'''


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
