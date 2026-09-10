import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createMotionTour,sampleMotionTour} from '../../../robot/hexapod_mkii_updated_v1/preview/motion-tour.js';

const model=JSON.parse(fs.readFileSync(new URL('../../../robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json',import.meta.url),'utf8'));
const tour=createMotionTour(model.joints);
const expected=new Map(model.joints.map(j=>[j.name,j]));
assert.equal(tour.count,18);
assert.equal(new Set(tour.phases.map(p=>p.name)).size,18);
const endpoints=[];
for(const p of tour.phases.filter(p=>p.hold)) {
  const state=sampleMotionTour(tour,(p.start+p.end)/2),j=expected.get(state.name);
  const angle=state.label==='Lower endpoint'?j.lower:state.label==='Upper endpoint'?j.upper:(j.default_value||0);
  assert.equal(state.angle,angle);
  if(!state.label.startsWith('Return'))endpoints.push({joint:j.name,phase:state.label,angle_rad:angle});
}
assert.equal(endpoints.length,36);
assert.equal(endpoints.filter(e=>e.joint.includes('tibia')&&e.phase==='Upper endpoint'&&Math.abs(e.angle_rad-Math.PI)<1e-12).length,6);
// Exercise the browser's largest allowed time step at every supported speed.
// A slow/background frame must still display every endpoint hold.
for(const speed of [1,2,4]) {
  const visits=new Set();let previous=0;
  for(let t=0;t<tour.duration+.4;t+=.1*speed) {
    const s=sampleMotionTour(tour,t),j=expected.get(s.name);
    assert(s.angle>=j.lower-1e-12 && s.angle<=j.upper+1e-12);
    assert(s.index>=previous);previous=s.index;
    if(s.hold)visits.add(s.name+'/'+s.label);
  }
  assert.equal(visits.size,54,'A render cadence skipped an endpoint at '+speed+'×');
}
const final=sampleMotionTour(tour,tour.duration);
assert(final.complete);assert.equal(final.angle,0);
const initial=sampleMotionTour(tour,0);assert.equal(initial.angle,0);
assert.deepEqual(sampleMotionTour(tour,17.2),sampleMotionTour(tour,17.2),'Paused time must not change the pose');
console.log(JSON.stringify({passed:true,joints:18,endpoints:36,hold_visits:54,duration_seconds:tour.duration,tested_speeds:[1,2,4],endpoints_rad:endpoints},null,2));
