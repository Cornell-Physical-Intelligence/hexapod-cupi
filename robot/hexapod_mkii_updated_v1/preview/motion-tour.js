// Scalar joint travel deliberately stays unwrapped, including tibia +180°.
export function createMotionTour(joints) {
  const phases=[];
  let time=0;
  for (const [index,joint] of joints.entries()) {
    const neutral=joint.default_value||0;
    if (!(joint.lower<=neutral && neutral<=joint.upper)) throw new Error('Neutral is outside '+joint.name+' limits');
    let from=neutral;
    for (const [label,to] of [['Lower endpoint',joint.lower],['Upper endpoint',joint.upper],['Return to zero',neutral]]) {
      // Smooth interpolation has a 1.5× peak/mean ratio: ≤60°/s at 1×.
      const duration=Math.max(.8,1.5*Math.abs(to-from)/(Math.PI/3));
      phases.push({start:time,end:time+duration,index,name:joint.name,from,to,label,hold:false});
      time+=duration;
      phases.push({start:time,end:time+.75,index,name:joint.name,from:to,to,label,hold:true});
      time+=.75;from=to;
    }
  }
  return {phases,duration:time,count:joints.length};
}

export function sampleMotionTour(tour,elapsed) {
  const complete=elapsed>=tour.duration;
  const p=tour.phases.find(p=>elapsed<p.end)||tour.phases.at(-1);
  const t=Math.max(0,Math.min(1,(elapsed-p.start)/(p.end-p.start)));
  const blend=t*t*(3-2*t);
  return {...p,angle:p.from+(p.to-p.from)*blend,complete};
}
