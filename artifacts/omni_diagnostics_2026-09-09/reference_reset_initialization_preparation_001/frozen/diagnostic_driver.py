"""Two independent zero-residual recovery arms; no actor/optimizer or native-step copy.

An external source-verified entrypoint owns environment construction, unchanged
fresh standing admission and final export. This module is CPU-prepared, not a
standalone launch authorization.
"""
import torch
from moving_session import MovingSession
from reset_pose_override import ResetPoseOverride

class QuietResetSession(MovingSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.diagnostic_reset_open=False
        self.cfg.update(scope='explicit_zero_residual_reset_initialization_diagnostic',PPO_permitted=False)

    def _schedule(self):return torch.zeros_like(self.commands)

    def _reset_rows(self,ended,terminated,truncated):
        if not ended.any():return
        if not self.diagnostic_reset_open:
            self._fatal('Unplanned terminal/reset in zero-residual diagnostic; preserve arm failure without retry')
        return super()._reset_rows(ended,terminated,truncated)

    def planned_reset(self,ids):
        if self.failure is not None or not self.active.all():raise RuntimeError('Prior recovery must finish before next declared reset')
        selected=torch.zeros_like(self.active);selected[ids]=True
        zeros=torch.zeros_like(selected)
        # Native outcome fields remain immutable. These existing TRAINING
        # ledger fields describe an explicitly artificial diagnostic reset;
        # no learner, value bootstrap or reward interpretation consumes them.
        row=self.rows[-1]
        if row['training_terminated'].any() or row['training_truncated'].any():raise RuntimeError('Cannot overwrite a real terminal transition')
        row['diagnostic_planned_reset_mask']=selected.cpu().numpy().copy()
        row['diagnostic_reset_is_not_training_timeout']=True
        row['training_truncated']=selected.cpu().numpy().copy()
        self.terminal_packet=self.get_observations()
        self.diagnostic_reset_open=True
        try:self._reset_rows(selected,zeros,selected)
        finally:self.diagnostic_reset_open=False


def run_arm(env,layout,points,stance,plan,arm,*,warp_to_torch,output):
    """At most1000 controls; caller starts a fresh32-row environment per arm.

    A failure is propagated after immutable evidence export. The campaign may
    run the OTHER predeclared arm once, but may not resume/retry this arm.
    """
    if int(env._sim_step_counter)!=0:raise RuntimeError('Independent arm requires a fresh no-control scene')
    override=ResetPoseOverride(env,plan,arm);session=None
    try:
        with override.installed():
            override.arm_next()
            env.reset(seed=0);env.episode_length_buf.zero_()
            session=QuietResetSession(env,layout,points,stance,warp_to_torch=warp_to_torch,output=output)
            with session:
                for epoch in plan['epochs'][1:]:
                    if session.control!=epoch['at_control']:raise RuntimeError('Diagnostic epoch/control mismatch')
                    ids=override.arm_next();session.planned_reset(ids)
                    for _ in range(200):session.step(session.zero)
                    if not session.active.all():raise RuntimeError('Declared200-control recovery did not complete')
                    if session.commands.any():raise RuntimeError('Zero-command intervention changed')
            if session.control!=1000 or override.next_epoch!=5:raise RuntimeError('Incomplete bounded reset campaign')
        return dict(arm=arm,status='completed_recovery_diagnostic',controls=session.control,reset_records=override.records,
                    physical_or_policy_admission=False,evaluation_gates_changed=False)
    finally:
        from pathlib import Path
        import json
        p=Path(output);p.mkdir(parents=True,exist_ok=True)
        record=dict(arm=arm,reset_records=override.records,completed_resets=override.next_epoch,
            session_failure=None if session is None else session.failure,
            original_reset_method_restored=env._reset_idx==override.original,
            no_PPO=True,no_hardware_startup_qualification=True)
        (p/'reset_initialization.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
