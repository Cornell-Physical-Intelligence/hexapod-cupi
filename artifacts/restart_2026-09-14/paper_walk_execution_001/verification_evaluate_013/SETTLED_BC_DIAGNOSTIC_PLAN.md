# Prospective diagnostic only — not dispatched

Question: does the known BC policy's failure to initiate forward walking depend on
starting directly from reset instead of the settled onset state represented by its
moving demonstrations? This does not test PPO006 or replace any qualification case.

Bind the exact BCfit004 checkpoint and its source/model/physics identities. Its
model is byte-identical to fit003 used for012, but retain both checkpoint identities.
Preserve012 as historical evidence and acquire a fresh cold-start control alongside
the settled-start treatment so changes in allocation or cache state are not silently
confounded with settling. Root owns review, scheduling and native execution.

- Control: one ordinary native reset, then20s of deterministicBC at command+0.05m/s.
- Treatment: one ordinary native reset,4s of actual neutral-target holding with zero
  command, then20s of the same deterministicBC at+0.05m/s. The neutral prefix must
  update the actual proprioceptive history and held-target state through normal
  stepping. No subsequent pose/joint write, history invention, phase input or reset.

The only treatment difference is the4s neutral prefix. Use the same floor, placement,
seed, robot,400/50Hz motor model and camera behavior. Record the full20s/24s traces,
including every400Hzcontact, target, torque, body/joint state and control observation.
Preserve any prefix failure. Report prefix and policy-onset boundaries explicitly;
score the20s policy portions with the unchanged static forward screen and report
physical bounds for the entire recordings. A treatment pass remains a diagnostic
result: it does not qualify the cold-start case or any of the96Stage2 cases.

Compare signed forward speed, projection on the command, toe-support pattern and
toe-reference motion, joint/target behavior, actuator limits, normalizer clipping
and estimator calibration. If both remain near-static, settling alone is insufficient.
If treatment improves, repeat the pair with a fresh run identity before attributing
the effect to onset state. Only then consider whether additional actual native
cold-start/transition demonstrations are justified. No new BC fit, source change,
reward change, statistics reset or native run is authorized by this plan itself.
