"""Evaluate native locomotion and bind its evidence.

The package holds the motion acceptance scorer, the native policy evaluation runner,
standing capture and its scorer, the Kit RGB camera, training load accumulation,
contact-force and torque summaries, the fixed-mean noise probe, checkpoint identity
checks, native scene checks and standing admission, training task metrics and PPO
update diagnostics. Callers import the scorer names from the package.
"""

from locomotion.evaluation.scoring import (
    MOVING_RMS as MOVING_RMS,
)
from locomotion.evaluation.scoring import (
    QUIET_GATES as QUIET_GATES,
)
from locomotion.evaluation.scoring import (
    STAGE2C_TORQUE as STAGE2C_TORQUE,
)
from locomotion.evaluation.scoring import (
    STAND_RMS as STAND_RMS,
)
from locomotion.evaluation.scoring import (
    TAIL as TAIL,
)
from locomotion.evaluation.scoring import (
    evaluation_scenarios as evaluation_scenarios,
)
from locomotion.evaluation.scoring import (
    score_recording as score_recording,
)
from locomotion.evaluation.scoring import (
    score_transitions as score_transitions,
)
from locomotion.evaluation.scoring import (
    summarize_suite as summarize_suite,
)
from locomotion.evaluation.scoring import (
    trajectory_command as trajectory_command,
)
from locomotion.evaluation.scoring import (
    transition_sequence as transition_sequence,
)
