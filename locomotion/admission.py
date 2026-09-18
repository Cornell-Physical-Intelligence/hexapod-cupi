"""Bind a run to its model and recorded native standing checks."""
from pathlib import Path
import argparse
import json

from .env_config import MODEL_SHA256, USD_SHA256, sha


def require_admission(path, identity, cfg):
    """New training can consume only authentic matching one/batch standing passes."""
    if path is None:
        raise ValueError("Training/recording requires fresh model/source-bound standing admission")
    value = json.loads(Path(path).read_text())
    return validate_admission(value, identity, cfg)


def validate_admission(value, identity, cfg, mounts=None):
    def resolve(name):
        path = Path(name)
        for target, source in (mounts or {}).items():
            if path.is_relative_to(target):
                return Path(source)/path.relative_to(target)
        return path
    if value.get("schema") != "hexapod_locomotion_standing_admission_v1":
        raise ValueError("Wrong standing admission schema")
    if value.get("model_sha256") != MODEL_SHA256 or value.get("usd_sha256") != USD_SHA256:
        raise ValueError("Standing admission model differs")
    if value.get("physics_source_files") != identity["physics_source_files"] or value.get("stance_sha256") != identity["stance_sha256"]:
        raise ValueError("Standing source or neutral stance differs")
    if value.get("physics_config") != identity["physics_config"]:
        raise ValueError("Standing physics configuration differs")
    for key in ("geometry_sha256", "geometry_extrema_sha256"):
        if value.get(key) != identity[key]:
            raise ValueError("Standing geometry identity differs")
    if type(value.get("num_envs")) is not int or value["num_envs"] < 32:
        raise ValueError("At least 32 same-source standing replicas are required")
    for name, count in (("one", 1), ("batch", value.get("num_envs"))):
        row = value.get(name, {})
        report_path = resolve(row.get("report_path", ""))
        if not report_path.is_file() or sha(report_path) != row.get("report_sha256"):
            raise ValueError("Missing/changed actual standing report: " + name)
        report = json.loads(report_path.read_text())
        if (report.get("all_pass") is not True or report.get("num_envs") != count
                or report.get("controls") != 1000 or report.get("substeps") != 8000
                or len(report.get("replicas", [])) != count
                or [r.get("env") for r in report["replicas"]] != list(range(count))
                or not all(type(r.get("env")) is int and r.get("pass") is True
                    and r.get("failed_physical_bounds") == []
                    and r.get("quiet", {}).get("pass") is True
                    and r.get("quiet", {}).get("failed_bounds") == []
                    for r in report["replicas"])):
            raise ValueError("Standing quality rejected: " + name)
        state_path=resolve(row.get("state_path", ""))
        if not state_path.is_file() or sha(state_path)!=row.get("state_sha256"):
            raise ValueError("Missing/changed native standing acquisition: "+name)
        state=json.loads(state_path.read_text())
        if state.get("status")!='completed' or state.get("standing_gate_pass") is not True or state.get("errors"):
            raise ValueError("Native standing acquisition failed: "+name)
        observed=state.get("identity",{})
        for key in ('physics_source_files','physics_config','stance_sha256','model_sha256',
                    'usd_sha256','geometry_sha256','geometry_extrema_sha256'):
            if observed.get(key)!=identity[key]:
                raise ValueError("Native standing identity differs: "+name+':'+key)
        if observed.get('config',{}).get('num_envs')!=count:
            raise ValueError("Native standing replica count differs: "+name)
    if cfg.num_envs != 1 and value.get("num_envs") != cfg.num_envs:
        raise ValueError("Training replica layout lacks matching standing admission")
    return value


def create(one, batch, output, inputs):
    """Recompute both native captures before writing admission and launch inputs."""
    from .env import score_diagnostic
    from .env_config import EnvConfig
    one, batch, output, inputs = map(Path, (one, batch, output, inputs))
    roots = {'one': one.resolve(), 'batch': batch.resolve()}
    states = {name: json.loads((root/'state.json').read_text()) for name, root in roots.items()}
    identity = states['batch']['identity']
    count = identity['config']['num_envs']
    keys = ('model_sha256', 'usd_sha256', 'physics_source_files', 'physics_config',
            'stance_sha256', 'geometry_sha256', 'geometry_extrema_sha256')
    value = {'schema': 'hexapod_locomotion_standing_admission_v1', 'num_envs': count,
             **{key: identity[key] for key in keys}}
    declared = json.loads(inputs.read_text())
    mounts = {}
    for name, root in roots.items():
        report_path = root/'standing/standing_report.json'
        report = json.loads(report_path.read_text())
        if score_diagnostic(root/'standing') != report:
            raise ValueError('Standing report differs from raw recomputation: '+name)
        target = '/standing_'+name
        mounts[target] = root
        value[name] = {'report_path': target+'/standing/standing_report.json',
            'report_sha256': sha(report_path), 'state_path': target+'/state.json',
            'state_sha256': sha(root/'state.json')}
        declared['extra_mounts'].append([str(root), target])
        for path in (report_path, root/'state.json'):
            declared['input_files'][str(path)] = sha(path)
    validate_admission(value, identity, EnvConfig(num_envs=count), mounts)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'admission.json'
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    declared['extra_mounts'].append([str(output.resolve()), '/admission'])
    declared['input_files'][str(path.resolve())] = sha(path)
    (output/'inputs.json').write_text(json.dumps(declared, indent=2, allow_nan=False)+'\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('one', 'batch', 'output', 'inputs'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
