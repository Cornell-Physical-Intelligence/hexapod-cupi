#!/usr/bin/env python3
"""Generate a source map from a Git revision without importing project code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote


PACKAGES = {'gait_clock', 'behavior_cloning', 'priors', 'surrogate', 'tools'}


def area(path):
    parts = Path(path).parts
    if '/tests/' in path:
        return 'tests'
    return parts[1] if len(parts) > 2 and parts[1] in PACKAGES else 'package'


def module_name(path):
    name = path[:-3].replace('/', '.')
    return name[:-len('.__init__')] if name.endswith('.__init__') else name


def dependencies(tree, path, known):
    """Repository files that this file imports, each resolved by its longest known module prefix."""
    here = module_name(path)
    package = here if path.endswith('/__init__.py') else here.rpartition('.')[0]
    found = set()

    def resolve(name):
        parts = name.split('.')
        for end in range(len(parts), 0, -1):
            target = known.get('.'.join(parts[:end]))
            if target:
                if target != path:
                    found.add(target)
                return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolve(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split('.')
                base = base[:len(base) - node.level + 1]
                prefix = '.'.join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ''
            for alias in node.names:
                resolve(prefix + '.' + alias.name if prefix else alias.name)
    return sorted(found)


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
    known = {module_name(path): path for path in paths}
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
                     "dynamic_references": dynamic, "depends_on": dependencies(tree, path, known),
                     "url": url + quote(path, safe="/")})
    missing = [r["path"] for r in rows if not r["test"] and r["path"] not in ownership]
    if missing:
        raise ValueError("Production modules need inventory entries: " + repr(missing))
    totals = {}
    for label, test in (("production", False), ("tests", True)):
        selected = [r for r in rows if r["test"] == test]
        totals[label] = {"files": len(selected), "physical_lines": sum(r["lines"] for r in selected),
                         "classes": sum(s["kind"] == "class" for r in selected for s in r["symbols"]),
                         "functions_and_methods": sum(s["kind"] == "function" for r in selected for s in r["symbols"])}
    return {"schema": "hexapod_code_map_v2", "revision": revision, "working_tree_changed": bool(git("status", "--porcelain", "--", "locomotion", "configs/source_inventory.json")), "totals": totals,
            "scope": "Static definitions and imports, including nested definitions. Import strings and factory expressions can remain unresolved. This is not a runtime call graph or a test-coverage report.", "modules": rows}


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hexapod · Locomotion code</title>
<link rel="stylesheet" href="../poster.css">
<style>
.filters {display:flex;gap:24px;flex-wrap:wrap;align-items:center;margin:4px 0 12px}
.filters input[type=search] {flex:1;min-width:220px;font:inherit;color:inherit;background:transparent;border:0;border-bottom:2px solid var(--ink);padding:8px 2px}
.filters input::placeholder {color:var(--ink);opacity:1}
.filters label {display:flex;align-items:center;gap:8px}
.map {display:block;width:100%;height:auto}
.map [data-package],.map [data-module] {cursor:pointer}
.map [data-package]:focus-visible,.map [data-module]:focus-visible {outline:3px solid var(--red)}
.map text {fill:var(--ink);pointer-events:none;paint-order:stroke;stroke:white;stroke-width:5px;stroke-linejoin:round}
.map .file text,.map .sym + text,.map text.on-dark {stroke:none}
.map .pkg rect {fill:var(--ink)}
.map .pkg.selected rect {fill:var(--red)}
.map .pkg.hit rect {stroke:var(--red);stroke-width:8}
.map .pkg.selected.hit rect {stroke:var(--ink)}
.map .use {fill:none;stroke:var(--grey)}
.map .use.active {stroke:var(--red)}
.map .arrow path {fill:var(--grey)}
.map .arrow.active path {fill:var(--red)}
.map .file rect {fill:white;stroke:var(--ink);stroke-width:1.5}
.map .file.test rect {stroke-dasharray:6 4}
.map .file.selected rect {fill:var(--ink)}
.map .file.selected text {fill:white}
.map .file.hit rect {stroke:var(--red);stroke-width:4}
.map .frame {fill:white;stroke:var(--ink);stroke-width:1.5}
.map .sym.class {fill:var(--ink)}
.map .sym.function {fill:white;stroke:var(--ink);stroke-width:1.5}
.map .sym.hit {fill:var(--red);stroke:var(--red)}
.map text.on-dark {fill:white}
#results p,#file p {margin:8px 0;overflow-wrap:anywhere}
</style>
</head>
<body>
<a class="skip" href="#code">Skip to the code</a>
<header class="masthead"><a class="brand" href="../"><span class="brand-mark" aria-hidden="true">H/6</span><span>Hexapod · Cornell Physical Intelligence</span></a><nav aria-label="Pages"><a href="../">Progress</a><a href="#code" aria-current="page">Code</a></nav><a class="repo-link" href="https://github.com/Cornell-Physical-Intelligence/hexapod-cupi">GitHub ↗</a></header>
<main id="code">
<section class="intro"><h1>Locomotion code map</h1><p>Each black square is a package, sized by its lines of code. An arrow points to a package it imports, so the packages others build on sit to the right. Select a square to see its files.</p></section>
<div class="filters" role="search" aria-label="Code map filters"><input id="query" type="search" aria-label="Search code map" placeholder="Search a file, class or function"><label><input type="checkbox" id="tests"> Tests</label></div>
<p id="count" role="status"></p>
<div id="results"></div>
<svg id="packages" class="map" role="group" aria-label="Packages and the packages they import"></svg>
<section class="section" aria-labelledby="files-heading"><h2 id="files-heading">Files</h2><p id="files-caption"></p><svg id="files" class="map" role="group" aria-label="Files in the selected package"></svg></section>
<section class="section" aria-labelledby="file-heading"><h2 id="file-heading">File</h2><p id="file-caption">The bar runs from the first line of the file to the last. Black blocks are classes and outlined blocks are functions; select a block to open its source.</p><svg id="strip" class="map" role="group" aria-label="Classes and functions in the selected file"></svg><div id="file"></div></section>
<footer class="section"><p id="revision"></p><p><a href="code-map.json">Download the source index</a></p></footer>
</main>
<script>const DATA=__DATA__;
const PACKAGES=[
['gait_clock','Gait clock','Omnidirectional PPO task and explicit motor control.'],
['behavior_cloning','Behavior cloning','Scripted BC and stock PPO fine-tuning.'],
['priors','Motion priors','Demonstrations and AMP learning.'],
['surrogate','CPU surrogate','MuJoCo training and trace replay.'],
['tools','Tools','Native jobs, capture and motion scoring.'],
['package','Locomotion','Package initialization.'],
['tests','Tests','Unit and parity tests.']
];
const TITLE=Object.fromEntries(PACKAGES.map(p=>[p[0],p[1]])), PURPOSE=Object.fromEntries(PACKAGES.map(p=>[p[0],p[2]]));
const el=id=>document.getElementById(id), esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link=(url,text)=>`<a href="${esc(url)}" target="_blank" rel="noopener">${esc(text)}</a>`;
const pick=(path,text)=>`<a href="#file-heading" data-file="${esc(path)}">${esc(text)}</a>`;
const num=n=>n.toLocaleString('en-US'), base=path=>path.split('/').pop(), clip=(text,width,size)=>{const room=Math.floor(width/(size*0.56));return text.length>room?text.slice(0,Math.max(room-1,1))+'…':text;};
const byPath=Object.fromEntries(DATA.modules.map(m=>[m.path,m]));
el('revision').innerHTML=link(`https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/tree/${DATA.revision}`,`Source ${DATA.revision.slice(0,8)}`)+(DATA.working_tree_changed?' · Local preview includes uncommitted changes.':'');
let selectedPackage=null, selectedFile=null;
const visible=m=>el('tests').checked||!m.test;
const query=()=>el('query').value.trim().toLowerCase();
const matches=(m,q)=>[m.path,m.purpose,...m.symbols.map(s=>s.name+' '+s.doc)].join(' ').toLowerCase().includes(q);
const hits=()=>{const q=query();return q?DATA.modules.filter(m=>visible(m)&&matches(m,q)).sort((a,b)=>a.path.localeCompare(b.path)):[];};
const largest=list=>list.slice().sort((a,b)=>b.lines-a.lines)[0];
const through=path=>byPath[path]&&visible(byPath[path])?[byPath[path]]:[];
function graph(){
  const shown=DATA.modules.filter(visible), lines={}, files={}, edges={};
  for(const m of shown){lines[m.area]=(lines[m.area]||0)+m.lines;files[m.area]=(files[m.area]||0)+1;
    for(const target of new Set(m.depends_on.flatMap(p=>through(p)))){if(target.area===m.area)continue;const key=m.area+'>'+target.area;edges[key]=(edges[key]||0)+1;}}
  const ids=PACKAGES.map(p=>p[0]).filter(id=>lines[id]);
  return {ids,lines,files,edges};
}
function drawPackages(found){
  const g=graph(), hitAreas=new Set(found.map(m=>m.area)), ids=g.ids.filter(id=>g.lines[id]>=10);
  const most=Math.max(...ids.map(id=>g.lines[id])), side=id=>Math.round(44+80*Math.sqrt(g.lines[id]/most));
  // Three columns ranked by packages imported minus packages importing: the foundation others build on sits on the right.
  const links=Object.keys(g.edges).map(k=>k.split('>')), outs=id=>links.filter(([a])=>a===id).length, ins=id=>links.filter(([,b])=>b===id).length;
  const ranked=ids.slice().sort((a,b)=>(outs(b)-ins(b))-(outs(a)-ins(a))||g.lines[b]-g.lines[a]), per=Math.ceil(ranked.length/3);
  const columns=[0,1,2].map(c=>ranked.slice(c*per,(c+1)*per).sort((a,b)=>g.lines[b]-g.lines[a]));
  const W=1000, centers=[150,500,850], cell=196, pos={};
  columns.forEach((column,c)=>column.forEach((id,r)=>{const s=side(id);pos[id]={x:centers[c]-s/2,y:16+r*cell+(124-s)/2,s,c};}));
  const H=24+Math.max(...columns.map(c=>c.length))*cell;
  const markers=['','active'].map(k=>`<marker id="arrow${k}" class="arrow ${k}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8"/></marker>`).join('');
  const arrows=Object.entries(g.edges).filter(([key])=>key.split('>').every(id=>pos[id])).map(([key,n])=>{
    const [a,b]=key.split('>'), p=pos[a], q=pos[b], active=a===selectedPackage||b===selectedPackage, py=p.y+p.s/2, qy=q.y+q.s/2;
    let d;
    if(p.c<q.c){const x1=p.x+p.s, x2=q.x-8, mid=(x1+x2)/2;d=`M${x1} ${py} C${mid} ${py},${mid} ${qy},${x2} ${qy}`;}
    else if(p.c>q.c){const x1=p.x, x2=q.x+q.s+8, mid=(x1+x2)/2;d=`M${x1} ${py} C${mid} ${py},${mid} ${qy},${x2} ${qy}`;}
    else{const x1=p.x+p.s, x2=q.x+q.s+8, out=Math.max(x1,x2)+70;d=`M${x1} ${py} C${out} ${py},${out} ${qy},${x2} ${qy}`;}
    return `<path class="use${active?' active':''}" data-edge="${esc(a+'>'+b)}" d="${d}" stroke-width="${(1.2+0.9*Math.log2(n)).toFixed(1)}" marker-end="url(#arrow${active?'active':''})"><title>${esc(TITLE[a])} imports ${esc(TITLE[b])} in ${n} places</title></path>`;}).join('');
  const squares=ids.map(id=>{const p=pos[id], cx=p.x+p.s/2, on=id===selectedPackage;
    return `<g class="pkg${on?' selected':''}${hitAreas.has(id)?' hit':''}" data-package="${esc(id)}" role="button" tabindex="0" aria-pressed="${on}" aria-label="${esc(TITLE[id])}: ${g.files[id]} files, ${num(g.lines[id])} lines"><rect x="${p.x}" y="${p.y}" width="${p.s}" height="${p.s}"/><text x="${cx}" y="${p.y+p.s+22}" text-anchor="middle" font-size="18" font-weight="700">${esc(TITLE[id])}</text><text x="${cx}" y="${p.y+p.s+42}" text-anchor="middle" font-size="15">${g.files[id]} files · ${num(g.lines[id])} lines</text></g>`;}).join('');
  el('packages').setAttribute?.('viewBox',`0 0 ${W} ${H}`);
  el('packages').innerHTML=`<defs>${markers}</defs>${arrows}${squares}`;
}
function squarify(items,box){
  const total=items.reduce((a,b)=>a+b.value,0), scale=box.w*box.h/total, out=[];
  const worst=(row,short)=>{const areas=row.map(r=>r.value*scale), s=areas.reduce((a,b)=>a+b,0);return Math.max(short*short*Math.max(...areas)/(s*s),s*s/(short*short*Math.min(...areas)));};
  let rest=items.slice();
  while(rest.length){const short=Math.min(box.w,box.h);let row=[rest[0]], best=worst(row,short);
    for(let i=1;i<rest.length;i++){const next=[...row,rest[i]], score=worst(next,short);if(score>best)break;row=next;best=score;}
    const s=row.reduce((a,r)=>a+r.value*scale,0);
    if(box.w>=box.h){const w=s/box.h;let y=box.y;for(const r of row){const h=r.value*scale/w;out.push({...r,x:box.x,y,w,h});y+=h;}box.x+=w;box.w-=w;}
    else{const h=s/box.w;let x=box.x;for(const r of row){const w=r.value*scale/h;out.push({...r,x,y:box.y,w,h});x+=w;}box.y+=h;box.h-=h;}
    rest=rest.slice(row.length);}
  return out;
}
function drawFiles(found){
  const files=DATA.modules.filter(m=>visible(m)&&m.area===selectedPackage).sort((a,b)=>b.lines-a.lines), hitPaths=new Set(found.map(m=>m.path));
  el('files-heading').textContent=TITLE[selectedPackage]||'Files';
  el('files-caption').textContent=`${PURPOSE[selectedPackage]||''} Each rectangle is one file, sized by its lines. Select a file to see its classes and functions.`;
  const W=1000, H=Math.max(260,Math.min(520,files.length*26));
  const tiles=squarify(files.map(m=>({value:Math.max(m.lines,1),m})),{x:0,y:0,w:W,h:H}).map(t=>{const m=t.m, on=m.path===selectedFile;
    const cls=`file${m.test?' test':''}${on?' selected':''}${hitPaths.has(m.path)?' hit':''}`;
    const label=t.w>64&&t.h>24?`<text x="${t.x+8}" y="${t.y+20}" font-size="14" font-weight="700">${esc(clip(base(m.path),t.w-14,14))}</text>`:'';
    const size=t.w>64&&t.h>46?`<text x="${t.x+8}" y="${t.y+38}" font-size="13">${num(m.lines)} lines</text>`:'';
    return `<g class="${cls}" data-module="${esc(m.path)}" role="button" tabindex="0" aria-pressed="${on}" aria-label="${esc(m.path)}, ${m.lines} lines"><rect x="${t.x+2}" y="${t.y+2}" width="${Math.max(t.w-4,1)}" height="${Math.max(t.h-4,1)}"/>${label}${size}<title>${esc(m.path)} · ${num(m.lines)} lines</title></g>`;}).join('');
  el('files').setAttribute?.('viewBox',`0 0 ${W} ${H}`);
  el('files').innerHTML=tiles;
}
function drawFile(found){
  const m=byPath[selectedFile];
  if(!m){el('file-heading').textContent='File';el('strip').innerHTML='';el('file').innerHTML='';return;}
  const q=query(), W=1000, n=Math.max(m.lines,1), top=m.symbols.filter(s=>!s.name.includes('.'));
  el('file-heading').textContent=m.path.replace('locomotion/','');
  const blocks=top.map(s=>{const x=10+980*(s.line-1)/n, w=Math.max(3,980*(s.end_line-s.line+1)/n), hit=q&&(s.name+' '+s.doc).toLowerCase().includes(q);
    const text=w>70?`<text x="${x+6}" y="37" font-size="14" class="${s.kind==='class'||hit?'on-dark':''}">${esc(clip(s.name,w-10,14))}</text>`:'';
    return `<a href="${esc(s.url)}" target="_blank" rel="noopener" aria-label="${esc(s.kind)} ${esc(s.name)}, lines ${s.line} to ${s.end_line}"><rect class="sym ${esc(s.kind)}${hit?' hit':''}" x="${x}" y="14" width="${w}" height="36"><title>${esc(s.name)} · lines ${s.line} to ${s.end_line}</title></rect>${text}</a>`;}).join('');
  el('strip').setAttribute?.('viewBox',`0 0 ${W} 74`);
  el('strip').innerHTML=`<rect class="frame" x="10" y="14" width="980" height="36"/>${blocks}<text x="10" y="70" font-size="14">Line 1</text><text x="990" y="70" font-size="14" text-anchor="end">Line ${num(m.lines)}</text>`;
  const files=paths=>paths.length?paths.map(p=>pick(p,p.replace('locomotion/',''))).join(', '):'None';
  el('file').innerHTML=`<p>${esc(m.purpose||'')} ${link(m.url,'Open the file ↗')}</p><p><strong>Imports:</strong> ${files([...new Set(m.depends_on.flatMap(p=>through(p)).map(t=>t.path))])}</p><p><strong>Imported by:</strong> ${files(DATA.modules.filter(o=>visible(o)&&o.depends_on.flatMap(p=>through(p)).some(t=>t.path===m.path)).map(o=>o.path))}</p>${m.symbols.length?`<p><strong>Definitions:</strong> ${m.symbols.map(s=>link(s.url,s.name)).join(', ')}</p>`:''}`;
}
function render(){
  const found=hits(), q=query();
  const shown=DATA.modules.filter(visible);
  if(!shown.some(m=>m.area===selectedPackage)){const sums={};for(const m of shown)sums[m.area]=(sums[m.area]||0)+m.lines;selectedPackage=Object.keys(sums).sort((a,b)=>sums[b]-sums[a])[0];selectedFile=null;}
  if(!selectedFile||!byPath[selectedFile]||!visible(byPath[selectedFile])||byPath[selectedFile].area!==selectedPackage)selectedFile=largest(shown.filter(m=>m.area===selectedPackage)).path;
  el('count').textContent=q?`${found.length} ${found.length===1?'file matches':'files match'} “${el('query').value.trim()}”`:'';
  el('results').innerHTML=q?(found.length?found.map(m=>{const named=m.symbols.filter(s=>(s.name+' '+s.doc).toLowerCase().includes(q));
    return `<p>${pick(m.path,m.path.replace('locomotion/',''))}${named.length?': '+named.map(s=>link(s.url,s.name)).join(', '):''}</p>`;}).join(''):'<p>No matching files.</p>'):'';
  drawPackages(found);drawFiles(found);drawFile(found);
}
function choose(target){
  const node=target?.closest?.('[data-package],[data-module],[data-file]');if(!node)return false;
  const d=node.dataset;
  if(d.package){selectedPackage=d.package;selectedFile=null;}
  else{const path=d.module||d.file;selectedFile=path;selectedPackage=byPath[path].area;}
  render();return true;
}
for(const id of ['packages','files','results','file']){
  el(id).addEventListener('click',e=>{if(e.target?.closest?.('[data-file]'))e.preventDefault?.();choose(e.target);});
  el(id).addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&choose(e.target))e.preventDefault?.();});
}
el('query').addEventListener('input',()=>{const found=hits();if(found.length){selectedFile=found[0].path;selectedPackage=found[0].area;}render();});
el('tests').addEventListener('input',render);
render();
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
