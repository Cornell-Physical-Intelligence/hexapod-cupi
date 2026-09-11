"""Proposed CPU command/observation transaction. No actor, reset, or native steps."""
import copy
from dataclasses import replace
import numpy as np
from oracles.commands import CommandBank
from oracles.causal_command import begin_control

class CommandHistory:
    def __init__(self, bank, history, initial_inputs):
        if history.num_envs != bank.n:
            raise ValueError('Command/history row mismatch')
        self.bank=CommandBank.from_state(bank.state())
        self.history=copy.deepcopy(history)
        self.history.reset_rows(np.arange(bank.n),replace(initial_inputs,command=self.bank.observe()))
        # Existing initialization convention: first valid frame repeated, not fabricated past sensor samples.
        self.index=0; self.pending=None

    def begin(self, held, q):
        if self.pending is not None: raise ValueError('Prior hold is incomplete')
        c=self.bank.observe()
        token=begin_control(c,self.history.history[:,-1],held,q,self.index)
        self.pending=copy.deepcopy(token)
        return copy.deepcopy(token)

    def complete(self, post_inputs, *, complete_substeps, reward_command):
        if self.pending is None: raise ValueError('No attempted hold')
        if type(complete_substeps) is not int or complete_substeps!=8:
            raise ValueError('Preserve partial prefix; do not advance command/history')
        if not np.array_equal(reward_command,self.pending['reward_requested_command']):
            raise ValueError('Reward must consume private c[t]')
        # All validation precedes commit. Command-program rollover is not a physical/history reset.
        bank=CommandBank.from_state(self.bank.state())
        boundary=bank.advance(); rows=np.flatnonzero(boundary)
        if rows.size: bank.reset_rows(rows)
        hist=copy.deepcopy(self.history)
        hist.push(replace(post_inputs,command=bank.observe()))
        self.bank=bank;self.history=hist;self.index+=1;self.pending=None
        return {'command_program_restarted_rows':rows.copy(),'physical_resets':0,'history_pushes':1}

    def actor_observation(self):
        if self.pending is not None: raise ValueError('No next actor until complete hold')
        return self.history.actor_observation()

    def state(self):
        # A pending transaction is deliberately checkpointed, not silently considered complete.
        return copy.deepcopy({'bank':self.bank.state(),'history':self.history,'index':self.index,'pending':self.pending})

    @classmethod
    def from_state(cls,state):
        # CPU-owned state object only; production serialization is a separate native lineage change.
        obj=cls.__new__(cls); s=copy.deepcopy(state)
        obj.bank=CommandBank.from_state(s['bank']); obj.history=s['history'];obj.index=s['index'];obj.pending=s['pending']
        return obj
