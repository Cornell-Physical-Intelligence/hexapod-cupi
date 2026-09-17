"""Pre-Kit-safe configuration for the accepted paper-walk RS05 actuator.

Only the SDK configuration class is imported here, as in ``rs05_v2.py``. Isaac
Lab resolves ``class_type`` to the runtime actuator after the simulation
backend starts, so importing this module loads no simulator and no USD.

The configuration keeps the implicit drive out of PhysX: the armature and the
joint friction are zero, and the runtime writes efforts alone. The gains
describe the accepted servo, and the runtime rejects a changed value.
"""

from __future__ import annotations

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass

from .rs05_paper_walk_model import (
    DAMPING_NM_S_PER_RAD_BY_NAME,
    STIFFNESS_NM_PER_RAD,
)

ACTUATOR_CLASS = "hexapod_env.actuators.rs05_paper_walk_runtime:RS05PaperWalkActuator"
#: Vendor peak torque. The applied ceiling comes from the speed curve and the
#: 1.6 N*m software clamp in the model module, which is tighter at every speed.
VENDOR_PEAK_NM = 5.5


@configclass
class RS05PaperWalkActuatorCfg(IdealPDActuatorCfg):
    class_type: str = ACTUATOR_CLASS
    active_joint_names: list[str] = []
    model_id: str = "rs05_paper_walk_v1"


def make_rs05_paper_walk_cfg(active_joint_names):
    """Bind the 18 named RS05 motors; never absorb another joint."""

    names = list(active_joint_names)
    table = dict(DAMPING_NM_S_PER_RAD_BY_NAME)
    if len(names) != 18 or set(names) != set(table):
        raise ValueError("The paper-walk actuator binds exactly the 18 named RS05 joints")
    return RS05PaperWalkActuatorCfg(
        joint_names_expr=names,
        active_joint_names=names,
        stiffness=STIFFNESS_NM_PER_RAD,
        damping=dict(table),
        armature=0.0,
        friction=0.0,
        effort_limit=VENDOR_PEAK_NM,
    )
