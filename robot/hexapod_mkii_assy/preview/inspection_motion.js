// Pure kinematic display motion. No learned policy, dynamics, or contact model.
export function animationPose({mode, phase, basePose, stancePose, legs, kinds, limits, leg, joint}) {
  const blend = (1 - Math.cos(phase)) / 2;
  const result = {...basePose};
  if (mode === 'pose') {
    for (const name of Object.keys(stancePose)) result[name] = (1 - blend) * stancePose[name];
  } else {
    const kind = mode === 'knee' ? 'tibia_pitch' : joint;
    if (!kinds.includes(kind)) throw new Error(`Unknown inspection joint ${kind}`);
    const selected = leg === 'all' ? legs : [leg];
    const {lower, upper} = limits[kind];
    for (const selectedLeg of selected) {
      if (!legs.includes(selectedLeg)) throw new Error(`Unknown inspection leg ${selectedLeg}`);
      result[`${selectedLeg}_${kind}`] = lower + (upper - lower) * blend;
    }
  }
  for (const name of Object.keys(result)) {
    const kind = name.split('_').slice(1).join('_'), {lower, upper} = limits[kind];
    result[name] = Math.min(upper, Math.max(lower, result[name]));
  }
  return result;
}

export function phaseForJoint(value, lower, upper) {
  return Math.acos(Math.min(1, Math.max(-1, 1 - 2 * (value - lower) / (upper - lower))));
}
