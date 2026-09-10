import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { createMotionTour, sampleMotionTour } from './motion-tour.js';

const $ = id => document.getElementById(id);
const RAD = Math.PI / 180;
const COLORS = ['#889397','#d79740','#5088a3','#986fb0','#b37840','#429592','#727cad','#ca925f','#4c99ba','#a086b9','#b48135','#488f8b','#7d87a9','#c38c59','#5799ad','#9b75a4','#bc9952','#4d8f9d','#8791b8'];
const stage = $('stage');
document.querySelectorAll('button,select,input').forEach(el=>el.disabled=true);
const renderer = new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,2));
renderer.setClearColor(0xf7f8f8);
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38,1,.001,30);
camera.up.set(0,0,1);
const controls = new OrbitControls(camera,renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = .10;
scene.add(new THREE.HemisphereLight(0xffffff,0x929aa2,2.3));
for(const [position,power] of [[[.5,-.8,1.5],2.2],[[-.9,.3,.7],1.5]]){
  const light = new THREE.DirectionalLight(0xffffff,power);light.position.set(...position);scene.add(light);
}
const grid = new THREE.GridHelper(2,40,0xc3cdd1,0xe1e6e8);grid.rotation.x=Math.PI/2;grid.position.z=-.0005;scene.add(grid);
const robot = new THREE.Group();scene.add(robot);
const raycaster=new THREE.Raycaster(), pointer=new THREE.Vector2();
const temp=new THREE.Object3D(), quaternion=new THREE.Quaternion();
let overlapIDs=new Set(),modelDigest;
const MOTION_LIFT_M=.20,SWEEP_MAX_RATE_RAD_S=.5;
let model,selected=null,playing=false,startTime=0,phaseOffset=0,lastAngle=0;
let tour=null,tourPlaying=false,tourElapsed=0,tourIndex=-1,lastFrameTime=null,tourMovingLinks=new Set(),tourVisits=[];
const links=new Map(),joints=new Map(),parts=new Map(),batches=[],rows=new Map(),jointAxes=[];
let linkColors=new Map(),angles=new Map();

function fail(error){$('error').hidden=false;$('error').textContent=String(error.stack||error);$('summary').textContent='Model did not finish loading';playing=false;console.error(error);}
window.addEventListener('error',event=>fail(event.error||event.message));
window.addEventListener('unhandledrejection',event=>fail(event.reason));
function resize(){const w=stage.clientWidth,h=stage.clientHeight;renderer.setSize(w,h);camera.aspect=w/Math.max(h,1);camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe(stage);

function setView(name='iso'){
  const box=new THREE.Box3().setFromObject(robot),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
  const extent=Math.max(size.x,size.y,size.z,.25),distance=extent*1.75;
  const dir={iso:[1,-1.35,.9],front:[0,-1,.08],side:[1,0,.08],top:[0,-.0001,1]}[name];
  camera.position.copy(center).add(new THREE.Vector3(...dir).normalize().multiplyScalar(distance));
  controls.target.copy(center);controls.update();
}
function setRootHeight(height){const delta=height-robot.position.z;robot.position.z=height;camera.position.z+=delta;controls.target.z+=delta;}
function stop(){
  playing=false;$('sweep').textContent='▶ Sweep';$('sweep').setAttribute('aria-pressed','false');
  if(tour){tour=null;tourPlaying=false;tourMovingLinks.clear();updateAppearance();$('tour-status').textContent='Review stopped. Play all 18 joints to restart.';}
  $('tour').textContent='▶ Play all 18 joints';$('tour').setAttribute('aria-pressed','false');document.body.dataset.tourState='stopped';
}
function startTour(){
  setPose('stand');selectPart(null);$('isolate').checked=false;$('overlaps').checked=false;$('explode').value=0;$('link-filter').value='all';explode();filterParts();
  tour=createMotionTour([...joints.values()]);tourElapsed=0;tourIndex=-1;tourVisits=[];tourPlaying=true;
  setRootHeight(model.root_height_m+MOTION_LIFT_M);setView($('view').value);
  $('tour').textContent='❚❚ Pause review';$('tour').setAttribute('aria-pressed','true');document.body.dataset.tourState='playing';
}
function advanceTour(dt){
  tourElapsed=Math.min(tour.duration,tourElapsed+dt*Number($('tour-speed').value));
  const sample=sampleMotionTour(tour,tourElapsed);
  if(sample.index!==tourIndex){
    tourIndex=sample.index;$('joint').value=sample.name;tourMovingLinks.clear();
    const collect=name=>{tourMovingLinks.add(name);for(const j of joints.values())if(j.parent===name)collect(j.child);};
    collect(joints.get(sample.name).child);updateAppearance();
  }
  for(const [name,j] of joints)angles.set(name,j.default_value||0);
  angles.set(sample.name,sample.angle);applyAngles();updateJointUI();
  const key=sample.name+'/'+sample.label;
  if(sample.hold&&!tourVisits.some(v=>v.key===key))tourVisits.push({key,joint:sample.name,phase:sample.label,angle_rad:sample.angle});
  $('tour-status').textContent=(sample.index+1)+' / 18 · '+sample.name.replaceAll('_',' ')+' · '+sample.label+(sample.hold?' · hold':'');
  $('tour-progress').value=sample.index+(sample.complete?1:0);
  $('pose').textContent='Full range review · '+sample.name.replaceAll('_',' ')+' · '+(sample.angle/RAD).toFixed(2)+'°';
  document.body.dataset.tourJoint=sample.name;document.body.dataset.tourPhase=sample.label;document.body.dataset.tourEndpointVisits=tourVisits.length;
  if(sample.complete){
    tourPlaying=false;tourMovingLinks.clear();updateAppearance();$('tour').textContent='↻ Replay all 18';$('tour').setAttribute('aria-pressed','false');
    $('tour-status').textContent='Complete · 18 joints · all 36 endpoints visited · returned to zero';$('pose').textContent='Full range review complete · body lifted 200 mm';document.body.dataset.tourState='complete';
  }
}
function applyAngles(){
  for(const [name,j] of joints){
    const group=links.get(j.child);group.quaternion.fromArray(j.quaternion_xyzw);
    quaternion.setFromAxisAngle(new THREE.Vector3(...j.axis).normalize(),angles.get(name)||0);
    group.quaternion.multiply(quaternion);
  }
  robot.updateMatrixWorld(true);
}
function currentJoint(){return joints.get($('joint').value);}
function updateJointUI(){
  const j=currentJoint();if(!j)return;
  const value=(angles.get(j.name)||0)/RAD;
  $('lower').value=(j.lower/RAD).toFixed(3);$('upper').value=(j.upper/RAD).toFixed(3);
  $('angle').min=Math.min(j.lower/RAD,value);$('angle').max=Math.max(j.upper/RAD,value);
  $('angle').value=value;$('angle-label').textContent=value.toFixed(2)+'°';
  const outside=value<j.lower/RAD-.001||value>j.upper/RAD+.001;
  $('angle-label').style.color=outside?'#ad532c':'';
  const absolute=j.absolute_zero_azimuth_deg;
  $('joint-context').textContent=Number.isFinite(absolute)?'Body azimuth '+(absolute+value).toFixed(2)+'° · plate envelope '+j.absolute_lower_azimuth_deg.toFixed(2)+'° to '+j.absolute_upper_azimuth_deg.toFixed(2)+'°. Zero is the envelope midpoint; 0.5 mm plate clearance.':(model.joint_review_revision?'User-supplied travel limits in the original viewer zero. Femur and tibia zeros are unchanged.':'Initial inspection envelope; the reviewed travel-limit revision is being prepared.');
  jointAxes.forEach(a=>a.visible=$('axes').checked);
}
function setPose(kind){
  stop();for(const [name,j] of joints)angles.set(name,kind==='cad'?(model.cad_pose?.[name]??j.cad_value??0):(j.default_value||0));
  setRootHeight(kind==='cad'?(model.cad_root_height_m??model.root_height_m):model.root_height_m);
  $('pose').textContent=kind==='cad'?'Original exported pose · may exceed draft bounds':'Inspection stance';
  applyAngles();updateJointUI();
}
function describePart(p){
  $('detail').replaceChildren();
  const title=document.createElement('strong');title.textContent=p.name;$('detail').append(title);
  const dl=document.createElement('dl');
  for(const [key,value] of [['Source ID','#'+p.id],['Moving link',p.link],['CAD mass',(p.mass*1000).toFixed(4)+' g'],['Grouping',p.ownership_basis||'Geometry inference'],['Mesh',p.mesh],...(overlapIDs.has(p.id)?[['Review','Part of an exported screw/tibia overlap. Attachment grouping and clearance need review.']]:[])]){
    const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;dl.append(dt,dd);
  }
  $('detail').append(dl);$('selection-label').hidden=false;$('selection-label').textContent='#'+p.id+' · '+p.link+' · '+p.name;
}
function selectPart(id){
  const old=selected;selected=id===null?null:parts.get(Number(id));
  if(old)rows.get(old.id)?.classList.remove('selected');
  if(selected){rows.get(selected.id)?.classList.add('selected');describePart(selected);}
  else{$('selection-label').hidden=true;$('detail').textContent='Select a part to inspect its assigned link and CAD mass.';}
  updateAppearance();
}
function updateAppearance(){
  const ghost=$('ghost').checked,tinted=$('tint').checked,isolate=$('isolate').checked;
  const targetLink=selected?.link||($('link-filter').value!=='all'?$('link-filter').value:null);
  for(const batch of batches){
    batch.mesh.visible=!isolate||!targetLink||batch.link===targetLink;
    batch.mesh.material.transparent=ghost;batch.mesh.material.opacity=ghost?.23:1;
    batch.mesh.material.depthWrite=!ghost;
    for(let i=0;i<batch.parts.length;i++){
      const p=batch.parts[i];const c=new THREE.Color(tinted?linkColors.get(p.link):'#ffffff');
      if(!tinted)c.setRGB(...p.color.slice(0,3),THREE.SRGBColorSpace);
      if(selected&&selected.id===p.id)c.set('#efca35');
      else if(selected&&selected.link!==p.link&&!ghost)c.lerp(new THREE.Color('#e5e9eb'),.4);
      else if(tourMovingLinks.size&&!tourMovingLinks.has(p.link))c.lerp(new THREE.Color('#e5e9eb'),.65);
      batch.mesh.setColorAt(i,c);
    }
    if(batch.mesh.instanceColor)batch.mesh.instanceColor.needsUpdate=true;
  }
}
function explode(){
  const separation=Number($('explode').value)/1000;$('explode-label').textContent=$('explode').value+' mm';
  for(const batch of batches){
    const link=model.links.find(l=>l.name===batch.link),center=new THREE.Vector3(...link.com);
    for(let i=0;i<batch.parts.length;i++){
      const p=batch.parts[i];temp.position.fromArray(p.xyz);temp.quaternion.fromArray(p.quaternion_xyzw);temp.scale.set(1,1,1);
      if(separation){const direction=temp.position.clone().sub(center);if(direction.lengthSq()<1e-10)direction.set(0,0,1);temp.position.addScaledVector(direction.normalize(),separation);}
      if($('overlaps').checked&&!overlapIDs.has(p.id))temp.scale.set(0,0,0);
      temp.updateMatrix();batch.mesh.setMatrixAt(i,temp.matrix);
    }
    batch.mesh.instanceMatrix.needsUpdate=true;batch.mesh.computeBoundingSphere();
  }
}
function filterParts(){
  const query=$('search').value.trim().toLowerCase(),filter=$('link-filter').value;let count=0;
  for(const [id,p] of parts){const show=(!$('overlaps').checked||overlapIDs.has(id))&&(filter==='all'||p.link===filter)&&(!query||(p.name+' '+p.id+' '+p.link).toLowerCase().includes(query));rows.get(id).hidden=!show;if(show)count++;}
  $('part-count').textContent=count+' / '+parts.size;updateAppearance();
}
async function loadGeometry(files){
  const loader=new STLLoader(),geometries=new Map();let loaded=0;
  const queue=[...files];
  async function worker(){while(queue.length){const name=queue.shift(),g=await loader.loadAsync('../meshes/'+encodeURIComponent(name));g.computeVertexNormals();geometries.set(name,g);loaded++;$('summary').textContent='Loading original meshes · '+loaded+' / '+files.length;}}
  await Promise.all(Array.from({length:6},worker));return geometries;
}
async function initialize(){
  const response=await fetch('../model_rs05_mass_corrected.json');if(!response.ok)throw new Error('Model data HTTP '+response.status);const payload=await response.text();model=JSON.parse(payload);modelDigest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(payload)))).map(v=>v.toString(16).padStart(2,'0')).join('');
  if(model.parts.length!==1753||model.links.length!==19||model.joints.length!==18)throw new Error('Unexpected model inventory; inspect the build report.');
  const review=await (await fetch('../overlap_review_parts.json')).json();
  overlapIDs=new Set(Object.values(review.groups).flatMap(g=>[g.tibia_part_id,...g.screw_part_ids]));
  const geometries=await loadGeometry([...new Set(model.parts.map(p=>p.mesh))]);
  model.links.forEach((l,i)=>{const g=new THREE.Group();g.name=l.name;links.set(l.name,g);linkColors.set(l.name,COLORS[i%COLORS.length]);});
  const children=new Set(model.joints.map(j=>j.child));for(const l of model.links)if(!children.has(l.name))robot.add(links.get(l.name));
  for(const j of model.joints){
    joints.set(j.name,j);angles.set(j.name,j.default_value||0);const g=links.get(j.child);g.position.fromArray(j.xyz);g.quaternion.fromArray(j.quaternion_xyzw);links.get(j.parent).add(g);
    const axis=new THREE.ArrowHelper(new THREE.Vector3(...j.axis).normalize(),new THREE.Vector3(),.045,0x243c48,.009,.004);g.add(axis);jointAxes.push(axis);
    const opt=document.createElement('option');opt.value=j.name;opt.textContent=j.name.replaceAll('_',' ');$('joint').append(opt);
  }
  robot.position.z=model.root_height_m||0;
  const byBatch=new Map();
  for(const p of model.parts){
    p.color=p.color||[.6,.65,.7,1];parts.set(p.id,p);const key=p.link+'|'+p.mesh;
    if(!byBatch.has(key))byBatch.set(key,{link:p.link,geometry:geometries.get(p.mesh),parts:[]});byBatch.get(key).parts.push(p);
  }
  for(const batch of byBatch.values()){
    batch.mesh=new THREE.InstancedMesh(batch.geometry,new THREE.MeshStandardMaterial({color:0xffffff,roughness:.62,metalness:.18}),batch.parts.length);batch.mesh.name=batch.link;batch.mesh.userData.batch=batch;
    links.get(batch.link).add(batch.mesh);batches.push(batch);
  }
  const fragment=document.createDocumentFragment();
  for(const l of model.links){
    const group=model.parts.filter(p=>p.link===l.name);const opt=document.createElement('option');opt.value=l.name;opt.textContent=l.name+' · '+group.length+' parts · '+(l.mass*1000).toFixed(1)+' g';$('link-filter').append(opt);
    for(const p of group){
      const row=document.createElement('button');row.className='part'+(p.ownership_basis?.includes('chassis')?' inferred':'');row.setAttribute('role','listitem');row.dataset.partId=p.id;
      const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=linkColors.get(p.link);
      const text=document.createElement('span');text.className='text';const name=document.createElement('span');name.className='name';name.textContent=p.name;const meta=document.createElement('span');meta.className='meta';meta.textContent='#'+p.id+' · '+p.link+(p.ownership_basis?.includes('chassis')?' · review assignment':'');text.append(name,meta);
      const mass=document.createElement('span');mass.className='part-mass';mass.textContent=(p.mass*1000).toFixed(2)+' g';row.append(swatch,text,mass);row.onclick=()=>selectPart(p.id);rows.set(p.id,row);fragment.append(row);
    }
  }
  $('part-list').append(fragment);explode();applyAngles();updateAppearance();filterParts();updateJointUI();setView();
  $('summary').textContent=model.parts.length+' parts · '+model.links.length+' bodies · '+model.joints.length+' joints · '+model.links.reduce((s,l)=>s+l.mass,0).toFixed(6)+' kg with motor overrides';
  document.querySelectorAll('button,select,input').forEach(el=>el.disabled=false);
  document.body.dataset.modelReady='true';document.body.dataset.partCount=parts.size;document.body.dataset.batchCount=batches.length;
  if(new URLSearchParams(location.search).get('tour')==='1')startTour();
}
$('tour').onclick=()=>{
  if(!tour||tourElapsed>=tour.duration){startTour();return;}
  tourPlaying=!tourPlaying;$('tour').textContent=tourPlaying?'❚❚ Pause review':'▶ Resume review';$('tour').setAttribute('aria-pressed',String(tourPlaying));document.body.dataset.tourState=tourPlaying?'playing':'paused';
};
$('tour-stop').onclick=()=>{setPose('stand');$('tour-progress').value=0;};
$('stand').onclick=()=>setPose('stand');$('cad').onclick=()=>setPose('cad');$('fit').onclick=()=>setView($('view').value);$('view').onchange=()=>setView($('view').value);
$('joint').onchange=()=>{stop();updateJointUI();};
$('angle').oninput=()=>{stop();setRootHeight(model.root_height_m+MOTION_LIFT_M);angles.set(currentJoint().name,Number($('angle').value)*RAD);$('pose').textContent='Manual joint inspection · body lifted 200 mm';applyAngles();updateJointUI();};
for(const field of ['lower','upper'])$(field).onchange=()=>{
  stop();const lower=Number($('lower').value),upper=Number($('upper').value);if(!Number.isFinite(lower)||!Number.isFinite(upper)||lower>=upper||Math.max(Math.abs(lower),Math.abs(upper))>180){updateJointUI();return;}
  const j=currentJoint();j.lower=lower*RAD;j.upper=upper*RAD;updateJointUI();
};
$('sweep').onclick=()=>{
  if(playing){stop();return;}stop();const j=currentJoint();if(!j)return;playing=true;setRootHeight(model.root_height_m+MOTION_LIFT_M);startTime=performance.now();lastAngle=angles.get(j.name)||0;phaseOffset=Math.asin(THREE.MathUtils.clamp((lastAngle-(j.lower+j.upper)/2)/((j.upper-j.lower)/2),-1,1));$('sweep').textContent='❚❚ Pause';$('sweep').setAttribute('aria-pressed','true');$('pose').textContent='Kinematic sweep · body lifted 200 mm · '+j.name;
};
for(const id of ['ghost','tint','isolate'])$(id).onchange=updateAppearance;
$('axes').onchange=()=>jointAxes.forEach(a=>a.visible=$('axes').checked);
$('overlaps').onchange=()=>{explode();filterParts();$('pose').textContent=$('overlaps').checked?'CAD overlaps · inspect screw attachment and clearance':'Inspection view';};
$('explode').oninput=explode;$('search').oninput=filterParts;$('link-filter').onchange=filterParts;$('clear').onclick=()=>selectPart(null);
$('save-review').onclick=()=>{
  if(!model)return;const data={kind:'inspection_preferences_only',model_sha256:modelDigest,joint_review_revision:model.joint_review_revision??'initial_inspection',source_urdf_sha256:model.source_urdf_sha256,selected_part:selected?.id??null,angles_rad:Object.fromEntries(angles),draft_limits_rad:Object.fromEntries([...joints].map(([n,j])=>[n,{lower:j.lower,upper:j.upper}])),motion_review:{state:document.body.dataset.tourState,visits:tourVisits},note:'User preview edits are not hardware limits or qualified simulation settings.'};
  const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='mkii-joint-inspection-settings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
let pointerDown;
renderer.domElement.addEventListener('pointerdown',e=>{pointerDown=[e.clientX,e.clientY];});
renderer.domElement.addEventListener('pointerup',e=>{
  if(!pointerDown||Math.hypot(e.clientX-pointerDown[0],e.clientY-pointerDown[1])>5||e.button!==0)return;
  const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);
  const hits=raycaster.intersectObjects(batches.filter(b=>b.mesh.visible).map(b=>b.mesh),false);if(hits.length){const h=hits[0];selectPart(h.object.userData.batch.parts[h.instanceId].id);}
});
function frame(time){
  requestAnimationFrame(frame);
  const dt=lastFrameTime===null?0:Math.min(.1,Math.max(0,(time-lastFrameTime)/1000));lastFrameTime=time;
  if(tourPlaying)advanceTour(dt);
  if(playing){const j=currentJoint(),t=(time-startTime)/1000,amplitude=(j.upper-j.lower)/2,omega=Math.min(Math.PI/5,SWEEP_MAX_RATE_RAD_S/amplitude);angles.set(j.name,(j.lower+j.upper)/2+amplitude*Math.sin(phaseOffset+t*omega));applyAngles();updateJointUI();}
  controls.update();renderer.render(scene,camera);
}
resize();requestAnimationFrame(frame);initialize().catch(fail);
