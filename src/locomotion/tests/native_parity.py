"""Replay c48_long for refactor tests; production checkpoint rules stay unchanged."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

CHECKPOINT_SHA = 'bda143d766cc0a2b35cd7f85d5c11871325287a7a3b08484fd9f6ac64dace6a9'
DECLARATION_SHA = '8959def5b9c758f192e8447126620d7a65d2e72a4ac9daf87bbc66cbb26158c3'
# Compare these provenance fields in the separate source receipt, not as weight ABI.
SOURCE_KEYS = {'schema', 'physics_source_files', 'adapter_sha256', 'entry_sha256'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proof():
    value = json.loads((Path(__file__).resolve().parents[2]/'PARITY.json').read_text())
    if (value['schema'] != 'hexapod_refactor_parity_v1'
            or value['baseline_commit'] != 'b2a6168c35156dfe0a8202201aa574b8d9e43fb1'
            or value['role'] not in ('baseline', 'candidate')):
        raise ValueError('Invalid test-only parity source declaration')
    return value


def verify_weights(checkpoint):
    if sha(checkpoint) != CHECKPOINT_SHA or sha(Path(checkpoint).with_suffix('.json')) != DECLARATION_SHA:
        raise ValueError('Parity requires the unchanged c48_long weights and declaration')


def require_comparison_physics(recorded, candidate):
    """Permit the stock ground/isolation experiment; keep other physics fixed."""
    expected = dict(recorded)
    if candidate.get('floor') == 'isaaclab_ground_plane':
        if recorded.get('floor') != '80m_two_triangle_mesh_y_equals_x_seam':
            raise ValueError('Unknown comparison baseline floor')
        expected.update(floor='isaaclab_ground_plane', filter_collisions=True)
    if candidate != expected:
        raise ValueError('Parity physics settings differ beyond ground and isolation')


def replay():
    from contracts import release
    import locomotion.scripts.cli as cli
    import locomotion.scripts.play as play
    args = sys.argv[1:]
    def option(name):
        if args.count(name) != 1:
            raise ValueError('One explicit parity argument required: ' + name)
        return args[args.index(name)+1]
    if (option('--eval-scope') != 'probes' or option('--num-envs') != '1' or option('--seed') != '20260917'):
        raise ValueError('Parity requires all 13 c48_long probes with the retained seed')
    checkpoint = Path(option('--checkpoint'))
    verify_weights(checkpoint)
    proof()
    from locomotion.evaluation.checkpoint import require_compatibility_configuration, require_learner_identity
    keys = tuple(key for key in release.COMPATIBILITY_KEYS if key not in SOURCE_KEYS | {'physics_config'})
    def compare_weight_abi(record, identity):
        require_comparison_physics(record['identity']['physics_config'], identity['physics_config'])
        if any(record['identity'][key] != identity[key] for key in keys):
            raise ValueError('Parity weight ABI differs')
        require_learner_identity(record, identity)
    # Baseline main uses the contract tuple; candidate main calls the extracted helper.
    # Keep model, servo settings, inputs, seed, upstream code and embedded metadata checks.
    with patch.object(release, 'COMPATIBILITY_KEYS', keys), patch.object(
            cli, 'require_checkpoint_identity', compare_weight_abi, create=True), patch.object(
            play, 'require_learner_configuration', require_compatibility_configuration):
        # The host supervisor verifies frozen sources and checkpoint inputs after Kit exits.
        return play.main(args)


def launch():
    import locomotion.spark.launch as supervisor
    proof()
    original = supervisor.command
    def command(binding, paths, name):
        if binding['module'] != 'locomotion.scripts.play' or binding['mode'] != 'evaluate':
            raise ValueError('The parity launcher accepts evaluation only')
        argv = original(binding, paths, name)
        index = argv.index('-m') + 1
        if argv[index] != 'locomotion.scripts.play':
            raise ValueError('Unexpected parity entry point')
        argv[index] = 'locomotion.tests.native_parity'
        return argv
    sys.argv.remove('--launch')
    with patch.object(supervisor, 'command', command):
        return supervisor.main()


if __name__ == '__main__':
    raise SystemExit(launch() if '--launch' in sys.argv else replay())
