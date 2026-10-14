"""Validate the mandatory research-poster update contract and build GitHub Pages."""
from __future__ import annotations
import argparse, datetime as dt, fnmatch, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'
SITE_URL = 'https://cornell-physical-intelligence.github.io/hexapod-cupi/'
sys.path.insert(0, str(ROOT))
from tools import archive, code_map

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
    archived = archive.url(value, ROOT)
    if archived:
        return archived
    inventory_path = ROOT / 'configs/source_inventory.json'
    if not inventory_path.exists():
        return None
    inventory = read(inventory_path)
    require(re.fullmatch('[0-9a-f]{40}', inventory['baseline']), 'Invalid historical source baseline')
    retired = inventory.get('retired_sources', {})
    if value in retired.get('files', {}) and not (ROOT / value).exists():
        require(re.fullmatch('[0-9a-f]{40}', retired['source_commit']), 'Invalid retired source commit')
        content = subprocess.check_output(['git', 'show', retired['source_commit']+':'+value], cwd=ROOT)
        require(hashlib.sha256(content).hexdigest() == retired['files'][value], 'Retired source bytes changed')
        return 'https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/' + retired['source_commit'] + '/' + quote(value, safe='/')
    for item in inventory['tools']:
        if value == item['previous_path'] and value != item['path']:
            require((ROOT / item['path']).is_file(), 'Moved source replacement is missing')
            return 'https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/' + inventory['baseline'] + '/' + quote(value, safe='/')
    for item in inventory.get('archived_documents', []):
        if value == item['previous_path'] and not (ROOT / value).exists():
            if item.get('storage') == 'git':
                require(re.fullmatch('[0-9a-f]{40}', item['source_commit']), 'Invalid document source commit')
                content = subprocess.check_output(['git', 'show', item['source_commit']+':'+value], cwd=ROOT)
                require(hashlib.sha256(content).hexdigest() == item['sha256'], 'Archived document bytes changed')
                return 'https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/' + item['source_commit'] + '/' + quote(value, safe='/')
            path = ROOT / item['path']
            require(path.resolve().is_relative_to(ROOT) and not path.is_symlink()
                    and path.is_file(), 'Archived document is missing or unsafe')
            require(re.fullmatch('[0-9a-f]{40}', item['source_commit']), 'Invalid document source commit')
            require(sha(path) == item['sha256'], 'Archived document bytes changed')
            return 'https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/' + item['source_commit'] + '/' + quote(value, safe='/')
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
    if not p.exists():
        retained = archive.retained(value, ROOT)
        if retained is not None:
            return retained
    if not p.exists() and historical_source(value):
        return None
    require(p.exists() and not p.is_symlink(), 'Missing or symbolic evidence: ' + value)
    current = ROOT
    for part in Path(value).parts:
        require(any(child.name == part for child in current.iterdir()),
                'Evidence capitalization differs: ' + value)
        current = current / part
    return p

def registry():
    p = read(SITE / 'project.json')
    require(p['schema_version'] == 3, 'Unsupported progress schema')
    # STATUS is generated output, never committed; only its file name is validated.
    require(re.fullmatch(r'[A-Za-z0-9_.-]+\.md', p['status_source']), 'Status output must be a root Markdown file name')
    ids = set()
    for item in p['media']:
        path = reference(item['path'])
        require(path is not None and path.is_file(), 'Media must be an explicit repository file')
        require(item['type'] in ['video', 'image'], 'Unknown media type')
        require(sha(path) == item['sha256'], 'Media bytes changed: ' + item['id'])
        require(nonempty(item.get('caption')) and nonempty(item.get('controller')), 'Media must identify the actual content')
        reference(item['evidence'])
        require(item['id'] not in ids, 'Repeated media ID')
        ids.add(item['id'])
    require(all(item.get('poster_id') is None or item['poster_id'] in ids for item in p['media']),
            'Unknown video poster')
    for item in p['nodes'] + p['accomplishments'] + [p['robot']]:
        reference(item['source'])
    validate_progress(p)
    return p

STEP_STATES = {'done', 'now', 'next', 'later'}
NODE_STATES = {'working', 'in_progress', 'not_started'}
STEP_LABELS = {'done': 'Done', 'now': 'Now', 'next': 'Next', 'later': 'Later'}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def field(value, keys):
    require(isinstance(keys, list) and keys and all(nonempty(k) for k in keys), 'Invalid metric field path')
    for key in keys:
        require(isinstance(value, dict) and key in value, 'Missing metric field: ' + key)
        value = value[key]
    return value


def count(value):
    """A total is a number in the source, or the size of the list or table it names."""
    return len(value) if isinstance(value, (dict, list)) else value


def validate_progress(project):
    latest = project['latest']
    dt.date.fromisoformat(latest['date'])
    require(all(nonempty(latest.get(k)) for k in ('title', 'text', 'next')), 'Latest progress needs a title, text and next step')
    metric = latest['metric']
    require(nonempty(metric.get('label')), 'Metric needs a label')
    for key in ('value', 'total'):
        value = metric[key]
        require(type(value) in (int, float) and value == value and abs(value) != float('inf'), 'Metric must be finite')
    require(isinstance(metric.get('sources'), list) and metric['sources'], 'Metric needs evidence')
    for source in metric['sources']:
        path = reference(source['path'])
        require(path is not None and path.is_file(), 'Metric source must be a local file')
        raw = read(path)
        for key, pointer, read_value in [('value', 'field', field), ('total', 'total_field', lambda r, k: count(field(r, k)))]:
            actual = read_value(raw, source[pointer])
            require(type(actual) is type(metric[key]) and actual == metric[key], 'Progress metric differs from source evidence')
    media = {m['id']: m for m in project['media']}
    require(latest['media_ids'] and all(i in media for i in latest['media_ids']), 'Latest progress needs known media')
    steps = project['roadmap']
    step_ids = [s['id'] for s in steps]
    require(step_ids and len(step_ids) == len(set(step_ids)), 'Repeated or empty roadmap IDs')
    require(all(re.fullmatch('[a-z][a-z0-9_-]*', i) for i in step_ids), 'Invalid roadmap ID')
    require(all(s['state'] in STEP_STATES and nonempty(s.get('title')) for s in steps), 'Unknown roadmap state')
    states = [s['state'] for s in steps]
    require(states.count('now') == 1, 'The roadmap needs exactly one current step')
    current = states.index('now')
    require(all(s == 'done' for s in states[:current]) and 'done' not in states[current:],
            'Completed steps must come before the current step')
    for step in steps:
        if step['state'] == 'done':
            require(nonempty(step.get('accepted_by')) and nonempty(step.get('source')),
                    'A done roadmap step needs a named human acceptance and evidence')
            reference(step['source'])
    accomplishments = project['accomplishments']
    require(accomplishments, 'Accomplishments are required')
    dates = [dt.date.fromisoformat(a['date']) for a in accomplishments]
    require(dates == sorted(dates, reverse=True), 'List accomplishments newest first')
    require(all(nonempty(a.get('text')) for a in accomplishments), 'Each accomplishment needs text')
    nodes = project['nodes']
    node_ids = [n['id'] for n in nodes]
    require(len(node_ids) == len(set(node_ids)), 'Repeated system node ID')
    require(all(n['state'] in NODE_STATES for n in nodes), 'Unspecified system state')
    require(all(nonempty(n.get('title')) and nonempty(n.get('text')) for n in nodes), 'Each system part needs a title and text')
    require(all(type(n['column']) is int and type(n['row']) is int for n in nodes), 'System parts need a grid position')
    require(len({(n['column'], n['row']) for n in nodes}) == len(nodes), 'Two system parts share a position')
    require(all(a in node_ids and b in node_ids for a, b in project['edges']), 'Unknown graph endpoint')
    robot = project['robot']
    require(robot['media_id'] in media and nonempty(robot.get('caption')), 'Robot section needs its media and caption')
    require(f"{robot['mass_kg']:.2f} kg" in robot['caption'] and f"{robot['motors']} motors" in robot['caption'],
            'Robot caption must state the recorded mass and motor count')
    require(all(nonempty(g.get('term')) and nonempty(g.get('meaning')) for g in project['glossary']), 'Each term needs a meaning')


def status_text(project, link=lambda value: value):
    """Deterministic text projection; STATUS is never an input to this renderer."""
    latest = project['latest']
    metric = latest['metric']
    lines = ['# Hexapod progress', '',
             '<!-- Generated from site/project.json by tools/project_site.py status. Do not edit. -->', '',
             project['mission'], '',
             f"## Latest: {latest['title']} ({latest['date']})", '',
             latest['text'], '',
             f"{metric['value']} of {metric['total']} {metric['label']}.", '',
             f"Next: {latest['next']}", '',
             '## Roadmap', '']
    lines.extend(f"- {step['title']}: {STEP_LABELS[step['state']]}" for step in project['roadmap'])
    lines.extend(['', '## Accomplishments', ''])
    lines.extend(f"- {a['date']}: {a['text']} [Evidence]({link(a['source'])})" for a in project['accomplishments'])
    lines.extend(['', f"[Progress site]({SITE_URL}) · [Architecture]({link('ARCHITECTURE.md')})", ''])
    return '\n'.join(lines)


def write_status(output=None):
    project = registry()
    path = ROOT / project['status_source'] if output is None else Path(output)
    path.write_text(status_text(project))
    print('Generated', path, 'from site/project.json (untracked)')


def records():
    out = []
    originals = {}
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
        require(isinstance(row['evidence'], list) and all(nonempty(v) for v in row['evidence']),
                'Update evidence must be a list of nonempty references')
        originals[row['id']] = (path, sha(path), when)
        out.append(row)
    require(out, 'A research update record is required')
    by_id = {row['id']: row for row in out}
    corrections = {}
    fields = {'old_record_id', 'old_record_sha256', 'old_reference',
              'replacement_reference', 'replacement_sha256'}
    for row in out:
        if 'evidence_reference_corrections' not in row:
            continue
        declarations = row['evidence_reference_corrections']
        require(isinstance(declarations, list) and declarations,
                'Evidence corrections must be a nonempty list')
        for correction in declarations:
            require(isinstance(correction, dict) and correction.keys() == fields
                    and all(nonempty(correction[k]) for k in fields),
                    'Incomplete evidence correction')
            target = correction['old_record_id']
            require(target in originals, 'Evidence correction target does not exist')
            _, old_sha, old_when = originals[target]
            require(old_when < originals[row['id']][2],
                    'Evidence correction must target an older update')
            require(re.fullmatch('[0-9a-f]{64}', correction['old_record_sha256'])
                    and old_sha == correction['old_record_sha256'],
                    'Evidence correction old record SHA differs')
            old_reference = correction['old_reference']
            require(by_id[target]['evidence'].count(old_reference) == 1,
                    'Evidence correction must identify one original reference')
            key = (target, old_reference)
            require(key not in corrections, 'Duplicate or conflicting evidence correction')
            replacement = correction['replacement_reference']
            require(replacement != old_reference and replacement in row['evidence'],
                    'Correction record must cite its distinct canonical replacement')
            path = reference(replacement)
            require(path is not None and path.is_file()
                    and re.fullmatch('[0-9a-f]{64}', correction['replacement_sha256'])
                    and sha(path) == correction['replacement_sha256'],
                    'Evidence correction replacement file or SHA differs')
            corrections[key] = dict(correction, correction_record_id=row['id'])
    # Only these freshly read presentation objects change. The pinned original
    # update files remain immutable, with their evidence and hash exposed below.
    for row in out:
        resolved = []
        applied = []
        for value in row['evidence']:
            correction = corrections.get((row['id'], value))
            replacement = correction['replacement_reference'] if correction else value
            reference(replacement)
            resolved.append(replacement)
            if correction:
                applied.append(correction)
        if applied:
            row['original_evidence'] = row['evidence']
            row['original_record'] = str(originals[row['id']][0].relative_to(ROOT))
            row['original_record_sha256'] = originals[row['id']][1]
            row['evidence_corrections_applied'] = applied
            row['evidence'] = resolved
    return out

def changed_paths(base):
    require(re.fullmatch('[0-9a-fA-F]{7,40}', base), 'Base must be an explicit Git SHA')
    git('rev-parse', '--verify', base + '^{commit}')
    def paths(*args):
        output = subprocess.check_output(['git', *args], cwd=ROOT, text=True)
        return filter(None, output.split('\0'))
    changed = set(paths('diff', '--no-renames', '--name-only', '-z', base))
    changed.update(paths('ls-files', '-z', '--others', '--exclude-standard'))
    return sorted(changed)

def check(base=None):
    project = registry()
    updates = records()
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
    return project, updates

def build():
    project, _ = check()
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
    (destination / project['status_source']).write_text(status_text(project, source_url))
    media = []
    for item in project['media']:
        target = Path('media') / (item['id'] + Path(item['path']).suffix.lower())
        (destination / target).parent.mkdir(exist_ok=True)
        if sys.platform == 'darwin':
            subprocess.run(['cp', '-c', str(ROOT / item['path']), str(destination / target)], check=True)
        else:
            shutil.copy2(ROOT / item['path'], destination / target)
        media.append({'id': item['id'], 'type': item['type'], 'url': target.as_posix(),
                      'caption': item['caption'], 'poster_id': item.get('poster_id')})
    for node in project['nodes']:
        node['source_url'] = source_url(node['source'])
    version = {'revision': revision, 'commit_date': git('show', '-s', '--format=%cI', 'HEAD'),
               'built_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'commit_url': origin + '/commit/' + revision}
    page = {key: project[key] for key in ('title', 'repository', 'mission', 'latest', 'roadmap',
                                           'accomplishments', 'nodes', 'edges', 'robot', 'glossary')}
    page.update(media=media, version=version)
    (destination / 'project-data.json').write_text(json.dumps(page, indent=2) + '\n')
    (destination / 'version.json').write_text(json.dumps(version) + '\n')
    code_map.build(ROOT, destination/'code')
    print('Built progress site:', destination)
    print('Revision:', revision, '| selected media:', len(media))

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
