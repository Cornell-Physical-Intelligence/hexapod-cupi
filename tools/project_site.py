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

def reference(value):
    require(isinstance(value, str) and value, 'A nonempty evidence reference is required')
    if value.startswith('https://'):
        u = urlsplit(value)
        require(u.hostname and not u.username and not u.password, 'Invalid public source URL')
        return None
    require(not urlsplit(value).scheme and not value.startswith('/'), 'Use repository-relative references')
    p = ROOT / value
    require(p.resolve().is_relative_to(ROOT) and not any(x == '..' for x in Path(value).parts), 'Reference escapes repository')
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
    require(p['schema_version'] == 1, 'Unsupported poster schema')
    for key in ['status_source', 'research_source']:
        reference(p[key])
    for item in p['nodes'] + p['milestones'] + p['findings']:
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
    return p, receipt, stop

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
    for item in project['nodes'] + project['milestones'] + project['findings']:
        item['source_url'] = source_url(item['source'])
    for row in updates:
        row['evidence_links'] = [{'label': Path(v).name, 'url': source_url(v)} for v in row['evidence']]
    status = (ROOT / project['status_source']).read_text()
    first = next(p for p in status.split('\n\n') if p.startswith('Updated '))
    replicas = [r for scenario in stop['scenarios'] for r in scenario['replicas']]
    project.update(version={'revision': revision, 'commit_date': git('show', '-s', '--format=%cI', 'HEAD'), 'built_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'commit_url': origin + '/commit/' + revision},
                   execution_snapshot=first, status_url=source_url(project['status_source']), research_url=source_url(project['research_source']),
                   checkpoint_data={'sha256': receipt['final_checkpoint_sha256'], 'updates': receipt['updates_completed'], 'reload_passed': receipt['reload']['passed'], 'evidence_url': source_url(project['checkpoint']['evidence'])},
                   stop_data={'passing': sum(r['quiet']['pass'] is True for r in replicas), 'replicas': len(replicas), 'evidence_url': source_url(project['stop_result']['evidence'])},
                   updates=sorted(updates, key=lambda r: r['date'], reverse=True)[:20])
    (destination / 'project-data.json').write_text(json.dumps(project, indent=2) + '\n')
    (destination / 'version.json').write_text(json.dumps(project['version']) + '\n')
    print('Built live research poster:', destination)
    print('Revision:', revision, '| selected media:', len(project['media']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'build'])
    parser.add_argument('--base', help='Explicit Git base SHA for contributor update coverage')
    args = parser.parse_args()
    if args.command == 'build':
        require(args.base is None, 'Use check --base before build')
        build()
    else:
        check(args.base)
        print('Research poster schema, evidence, media and change records verified')
