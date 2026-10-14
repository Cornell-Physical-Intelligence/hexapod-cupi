"""Check checkpoint declarations without changing legacy acceptance rules."""
import json
from contracts.release import COMPATIBILITY_KEYS

def checkpoint_reward_version(record):
    """The reward a checkpoint was trained with; checkpoints from before the flag trained version 1."""
    version = record['identity'].get('reward_version', '1')
    if version not in ('1', '2', '3', '4'):
        raise ValueError('Unknown checkpoint reward version: '+str(version))
    return version


LEARNER_DEFAULTS = {'learner': 'ppo', 'networks': 'mlp', 'motion_prior': False}
LEARNER_KEYS = ('learner', 'networks', 'motion_prior', 'learner_source_files', 'amp_dataset')


def json_shape(value):
    """The value as a checkpoint record stores it: JSON turns tuples into lists."""
    return json.loads(json.dumps(value, allow_nan=False))


def require_learner_identity(record, identity):
    """Reject a checkpoint whose learner, networks or demonstration bank differ; older records mean stock PPO."""
    recorded = record['identity']
    for key in LEARNER_KEYS:
        if recorded.get(key, LEARNER_DEFAULTS.get(key)) != json_shape(identity.get(key, LEARNER_DEFAULTS.get(key))):
            raise ValueError('Checkpoint learner, networks or demonstration bank differs: '+key)


def require_learner_configuration(record, identity, config):
    """Reject a checkpoint trained with other RSL-RL sources or another learner configuration."""
    recorded = record['identity']
    if (recorded['upstream_source_files'] != identity['upstream_source_files']
            or recorded['ppo_config'] != json_shape(config)):
        raise ValueError('Checkpoint learner dependency or configuration differs')


def require_embedded_declaration(infos, record):
    """The declaration pickled inside the checkpoint must equal its JSON record."""
    if json_shape(infos) != {'identity': record['identity'], 'updates': record['updates']}:
        raise ValueError('Embedded checkpoint declaration differs')


def require_checkpoint_identity(record, identity):
    recorded = record['identity']
    if any(recorded[key] != identity[key] for key in COMPATIBILITY_KEYS):
        raise ValueError('Checkpoint model, physics, seed or implementation differs')
    if identity['schema'] == 'hexapod_locomotion_ppo_v2':
        if recorded.get('runtime_sources') != identity.get('runtime_sources'):
            raise ValueError('Checkpoint runtime dependencies differ')
    require_learner_identity(record, identity)
