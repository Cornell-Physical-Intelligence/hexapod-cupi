"""Validate the mandatory research-poster update contract and build GitHub Pages."""
from __future__ import annotations
import argparse, datetime as dt, fnmatch, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def read(path):
    return json.loads(path.read_text())

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def historical_source(value):
    inventory_path = ROOT / 'configs/source_inventory.json'
    if not inventory_path.exists():
        return None
    inventory = read(inventory_path)
    require(re.fullmatch('[0-9a-f]{40}', inventory['baseline']), 'Invalid historical source baseline')
    for item in inventory['tools']:
        if value == item['previous_path'] and value != item['path']:
            require((ROOT / item['path']).is_file(), 'Moved source replacement is missing')
            return 'https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/' + inventory['baseline'] + '/' + quote(value, safe='/')
    return None


def reference(value):
    require(isinstance(value, str) and value, 'A nonempty evidence reference is required')
    if value.startswith('https://'):
        u = urlsplit(value)
        require(u.hostname and not u.username and not u.password, 'Invalid public source URL')
        return None
    require(not urlsplit(value).scheme and not value.startswith('/'), 'Use repository-relative references')
    p = ROOT / value
    require(p.resolve().is_relative_to(ROOT) and not any(x == '..' for x in Path(value).parts), 'Reference escapes repository')
    if not p.exists() and historical_source(value):
        return None
    require(p.exists() and not p.is_symlink(), 'Missing or symbolic evidence: ' + value)
    return p

def checkpoint_media(project, receipt):
    path = reference(project['checkpoint']['path'])
    require(path is not None and path.is_file(), 'Checkpoint bytes must be an explicit repository file')
    require(sha(path) == receipt['final_checkpoint_sha256'], 'Selected checkpoint bytes differ from the training receipt')
    matching = project['checkpoint']['video_matches']
    require(type(matching) is bool, 'Recording-match flag must be boolean')
    if matching:
        video = next((m for m in project['media'] if m['id'] == project['primary_video_id']), None)
        require(video is not None and video['type'] == 'video' and video['controller'] == receipt['final_checkpoint_sha256'],
                'Matching video must identify the selected checkpoint')

def registry():
    p = read(SITE / 'project.json')
    require(p['schema_version'] == 2, 'Unsupported progress schema')
    for key in ['status_source', 'research_source']:
        reference(p[key])
    for item in p['nodes'] + p['milestones']:
        reference(item['source'])
    ids = [n['id'] for n in p['nodes']]
    require(len(ids) == len(set(ids)), 'Repeated system node ID')
    require(all(n['state'] in ['implemented', 'in_progress', 'prepared', 'planned'] for n in p['nodes']), 'Unspecified system state')
    require(all(a in ids and b in ids for a, b in p['edges']), 'Unknown graph endpoint')
    for item in p['media']:
        path = reference(item['path'])
        require(path is not None and path.is_file(), 'Media must be an explicit repository file')
        require(item['type'] in ['video', 'image'], 'Unknown media type')
        require(sha(path) == item['sha256'], 'Media bytes changed: ' + item['id'])
        require(item['caption'] and item['controller'], 'Media must identify the actual content')
        reference(item['evidence'])
    require(len({m['id'] for m in p['media']}) == len(p['media']), 'Repeated media ID')
    for item in p['papers']:
        reference(item['url'])
    receipt = read(reference(p['checkpoint']['receipt']))
    require(receipt.get('complete') is True, 'Selected latest checkpoint needs a completed training receipt')
    require(re.fullmatch('[0-9a-f]{64}', receipt['final_checkpoint_sha256']), 'Invalid selected checkpoint SHA')
    checkpoint_media(p, receipt)
    reference(p['checkpoint']['evidence'])
    stop = read(reference(p['stop_result']['path']))
    reference(p['stop_result']['evidence'])
    require(stop.get('complete') is True and stop['checkpoint_sha256'] == receipt['final_checkpoint_sha256'], 'Checkpoint and stop evidence differ')
    validate_progress(p)
    return p, receipt, stop

PROGRESS_STATES = {'needs_definition', 'ready', 'active', 'blocked', 'accepted'}
ATTEMPT_RESULTS = {'passed', 'failed', 'interrupted', 'inconclusive'}
STATE_LABELS = {'needs_definition': 'Needs definition', 'ready': 'Ready',
                'active': 'Active', 'blocked': 'Blocked', 'accepted': 'Accepted'}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def field(value, keys):
    require(isinstance(keys, list) and keys and all(nonempty(k) for k in keys), 'Invalid metric field path')
    for key in keys:
        require(isinstance(value, dict) and key in value, 'Missing metric field: ' + key)
        value = value[key]
    return value


def validate_progress(project):
    progress = project['progress']
    when = dt.datetime.fromisoformat(progress['as_of'].replace('Z', '+00:00'))
    require(when.utcoffset() == dt.timedelta(0), 'Progress uses a UTC evidence timestamp')
    require(re.fullmatch('[0-9a-f]{40}', progress['source_commit']), 'Progress needs an exact source commit')
    architecture = reference(progress['architecture'])
    require(architecture is not None and architecture.is_file(), 'Architecture must be local')
    requirements = set(re.findall(r'R-[0-9]{2}', architecture.read_text()))
    reference(progress['execution_source'])
    reference(progress['prior_snapshot'])
    require(nonempty(progress['summary']) and nonempty(progress['definition_policy']), 'Progress explanation is required')
    markers = project['milestones']
    ids = [m['id'] for m in markers]
    require(ids and len(ids) == len(set(ids)), 'Repeated or empty roadmap IDs')
    require(all(re.fullmatch('[a-z][a-z0-9_-]*', i) for i in ids), 'Invalid roadmap ID')
    states = {m['id']: m['status'] for m in markers}
    for marker in markers:
        require(marker['status'] in PROGRESS_STATES, 'Unknown capability state')
        require(all(nonempty(marker[k]) for k in ('title', 'description', 'gate', 'backend', 'owner', 'question')), 'Incomplete roadmap meaning')
        deps = marker['depends_on']
        require(isinstance(deps, list) and len(deps) == len(set(deps)) and all(d in ids for d in deps), 'Unknown or repeated dependency')
        require(marker['requirements'] and all(r in requirements for r in marker['requirements']), 'Unknown architecture requirement')
        reference(marker['source'])
        if marker['status'] == 'blocked':
            require(nonempty(marker['blocker']), 'Blocked capability needs a reason')
        step = marker['next_step']
        if step is not None:
            require(isinstance(step, dict) and all(nonempty(step.get(k)) for k in
                    ('outcome', 'owner', 'reviewer', 'issue', 'acceptance', 'baseline')), 'Incomplete next-step definition')
            reference(step['issue'])
        if marker['status'] in {'ready', 'active'}:
            require(step is not None, 'Ready work needs a defined next step')
            require(all(states[d] == 'accepted' for d in deps), 'Ready work has unmet dependencies')
        if marker['status'] == 'accepted':
            acceptance = marker['acceptance']
            require(isinstance(acceptance, dict) and all(nonempty(acceptance.get(k)) for k in
                    ('by', 'scope', 'source')), 'Accepted capability needs human acceptance and scoped evidence')
            reference(acceptance['source'])
            require(all(states[d] == 'accepted' for d in deps), 'Accepted capability has unmet dependencies')
    by_id = {m['id']: m for m in markers}
    visited, visiting = set(), set()
    def visit(identifier):
        require(identifier not in visiting, 'Roadmap dependency cycle')
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in by_id[identifier]['depends_on']:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)
    for identifier in ids:
        visit(identifier)
    facts = progress['facts']
    require(isinstance(facts, list) and facts, 'Progress needs evidence')
    fact_ids = [f['id'] for f in facts]
    require(len(fact_ids) == len(set(fact_ids)), 'Repeated evidence ID')
    for fact in facts:
        require(fact['result'] in ATTEMPT_RESULTS, 'Unknown attempt result')
        require(all(nonempty(fact[k]) for k in ('title', 'backend', 'text')), 'Incomplete evidence scope')
        reference(fact['source'])
        if 'metric' in fact:
            metric = fact['metric']
            source = reference(metric['path'])
            require(source is not None and source.is_file(), 'Metric source must be a local file')
            raw = read(source)
            for key, pointer in [('value', 'field'), ('total', 'total_field')]:
                value = metric[key]
                require(type(value) in (int, float) and value == value and abs(value) != float('inf'), 'Metric must be finite')
                actual = field(raw, metric[pointer])
                require(type(actual) is type(value) and actual == value, 'Progress metric differs from source evidence')
            require(nonempty(metric['units']) and nonempty(metric['window']), 'Metric needs units and window')


def status_text(project):
    """Deterministic text projection; STATUS is never an input to this renderer."""
    progress = project['progress']
    lines = ['# Hexapod progress', '',
             '<!-- Generated from site/project.json by tools/project_site.py status. Do not edit. -->', '',
             f"Evidence snapshot: {progress['as_of']} · source `{progress['source_commit']}`.", '',
             progress['summary'], '',
             '[Architecture](ARCHITECTURE.md) · [Visual roadmap](https://cornell-physical-intelligence.github.io/hexapod-cupi/#roadmap)', '',
             '## Roadmap', '']
    for marker in project['milestones']:
        lines.extend([f"### {marker['title']} — {STATE_LABELS[marker['status']]}", '',
                      marker['description'], '', f"Scope: {marker['backend']}. Owner: {marker['owner']}.", '',
                      f"Required proof: {marker['gate']}", '', f"Current limitation: {marker['blocker']}", '',
                      f"[Evidence]({marker['source']}) · Architecture: {', '.join(marker['requirements'])}.", ''])
        if marker['acceptance']:
            lines.extend([f"Accepted scope: {marker['acceptance']['scope']}", ''])
        if marker['next_step']:
            step = marker['next_step']
            lines.extend([f"Next defined step: [{step['outcome']}]({step['issue']}).", ''])
        else:
            lines.extend([f"Next step: **not defined**. {marker['question']}", ''])
    lines.extend(['## Recorded attempts', ''])
    for fact in progress['facts']:
        lines.extend([f"- **{fact['title']} ({fact['result']})** — {fact['backend']}. {fact['text']} [Evidence]({fact['source']})"])
    lines.extend(['', progress['definition_policy'], '',
                  f"Current compute rules and reservation records: [operations]({progress['execution_source']}). This is not live GPU telemetry.", '',
                  f"[Full execution snapshot before consolidation]({progress['prior_snapshot']}). Historical results retain their original evidence and gates.", ''])
    return '\n'.join(lines)


def write_status():
    project, _, _ = registry()
    (ROOT / project['status_source']).write_text(status_text(project))
    print('Generated', project['status_source'], 'from site/project.json')


def records():
    out = []
    for path in sorted((SITE / 'updates').glob('*.json')):
        row = read(path)
        required = {'id', 'date', 'title', 'summary', 'areas', 'changes', 'evidence', 'next', 'no_project_impact'}
        require(required <= row.keys(), 'Incomplete update: ' + path.name)
        require(row['id'] == path.stem, 'Update ID must match its filename')
        when = dt.datetime.fromisoformat(row['date'].replace('Z', '+00:00'))
        require(when.utcoffset() == dt.timedelta(0), 'Use UTC update timestamps')
        require(type(row['no_project_impact']) is bool, 'No-impact flag must be boolean')
        require(row['areas'] and row['changes'] and row['summary'].strip() and row['next'].strip(), 'Empty change explanation')
        if row['no_project_impact']:
            require(len(row.get('reason', '').strip()) >= 20, 'Explain the specific no-impact reason')
        for pattern in row['changes']:
            require(isinstance(pattern, str) and pattern and not pattern.startswith(('/', '*')) and '..' not in Path(pattern).parts, 'Use bounded changed-path patterns')
        for value in row['evidence']:
            reference(value)
        out.append(row)
    require(out, 'A research update record is required')
    return out

def changed_paths(base):
    require(re.fullmatch('[0-9a-fA-F]{7,40}', base), 'Base must be an explicit Git SHA')
    git('rev-parse', '--verify', base + '^{commit}')
    changed = set(filter(None, git('diff', '--name-only', base).splitlines()))
    changed.update(filter(None, git('ls-files', '--others', '--exclude-standard').splitlines()))
    return sorted(changed)

def check(base=None):
    project, receipt, stop = registry()
    updates = records()
    if project:
        require((ROOT / project['status_source']).read_text() == status_text(project), 'Generated STATUS differs; run tools/project_site.py status')
    if base:
        changed = changed_paths(base)
        for name in changed:
            if name.startswith('site/updates/'):
                existed = subprocess.run(['git', 'cat-file', '-e', base + ':' + name], cwd=ROOT, capture_output=True).returncode == 0
                require(not existed, 'Research updates are append-only: ' + name)
        new_records = []
        for row in updates:
            name = 'site/updates/' + row['id'] + '.json'
            if name not in changed:
                continue
            exists = subprocess.run(['git', 'cat-file', '-e', base + ':' + name], cwd=ROOT, capture_output=True).returncode == 0
            require(not exists, 'Research updates are append-only; add a new record: ' + name)
            new_records.append(row)
        substantive = [f for f in changed if not f.startswith('site/updates/')]
        patterns = [p for row in new_records for p in row['changes']]
        uncovered = [f for f in substantive if not any(fnmatch.fnmatchcase(f, p) for p in patterns)]
        require(not uncovered, 'Repository changes lack a new central update record:\n' + '\n'.join(uncovered))
    return project, receipt, stop, updates

def build():
    project, receipt, stop, updates = check()
    revision = git('rev-parse', 'HEAD')
    origin = 'https://github.com/' + project['repository']
    def source_url(value):
        if value.startswith('https://'):
            return value
        if not (ROOT / value).exists() and historical_source(value):
            return historical_source(value)
        kind = 'tree' if (ROOT / value).is_dir() else 'blob'
        return origin + '/' + kind + '/' + revision + '/' + quote(value, safe='/')
    destination = SITE / 'dist'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    for name in ['index.html', 'poster.css', 'poster.js']:
        shutil.copy2(SITE / name, destination / name)
    (destination / '.nojekyll').touch()
    for item in project['media']:
        target = Path('media') / (item['id'] + Path(item['path']).suffix.lower())
        (destination / target).parent.mkdir(exist_ok=True)
        if sys.platform == 'darwin':
            subprocess.run(['cp', '-c', str(ROOT / item['path']), str(destination / target)], check=True)
        else:
            shutil.copy2(ROOT / item['path'], destination / target)
        item['url'] = target.as_posix()
        item['evidence_url'] = source_url(item['evidence'])
    for item in project['nodes'] + project['milestones']:
        item['source_url'] = source_url(item['source'])
    for row in updates:
        row['evidence_links'] = [{'label': Path(v).name, 'url': historical_source(v) or source_url(v)} for v in row['evidence']]
    for fact in project['progress']['facts']:
        fact['source_url'] = source_url(fact['source'])
    first = project['progress']['summary']
    replicas = [r for scenario in stop['scenarios'] for r in scenario['replicas']]
    project.update(version={'revision': revision, 'commit_date': git('show', '-s', '--format=%cI', 'HEAD'), 'built_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'commit_url': origin + '/commit/' + revision},
                   architecture_url=source_url(project['progress']['architecture']), execution_snapshot=first, status_url=source_url(project['status_source']), research_url=source_url(project['research_source']),
                   checkpoint_data={'sha256': receipt['final_checkpoint_sha256'], 'updates': receipt['updates_completed'], 'reload_passed': receipt['reload']['passed'], 'evidence_url': source_url(project['checkpoint']['evidence'])},
                   stop_data={'passing': sum(r['quiet']['pass'] is True for r in replicas), 'replicas': len(replicas), 'evidence_url': source_url(project['stop_result']['evidence'])},
                   updates=sorted(updates, key=lambda r: r['date'], reverse=True)[:20])
    (destination / 'project-data.json').write_text(json.dumps(project, indent=2) + '\n')
    (destination / 'version.json').write_text(json.dumps(project['version']) + '\n')
    print('Built live research poster:', destination)
    print('Revision:', revision, '| selected media:', len(project['media']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'build', 'status'])
    parser.add_argument('--base', help='Explicit Git base SHA for contributor update coverage')
    args = parser.parse_args()
    if args.command == 'status':
        require(args.base is None, 'Use check --base after generating status')
        write_status()
    elif args.command == 'build':
        require(args.base is None, 'Use check --base before build')
        build()
    else:
        check(args.base)
        print('Research poster schema, evidence, media and change records verified')
