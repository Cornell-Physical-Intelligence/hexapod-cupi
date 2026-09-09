import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import URDFLoader from 'urdf-loader';
import { animationPose, phaseForJoint } from './inspection_motion.js';

const $ = id => document.getElementById(id);
const BASE = new URL('../', import.meta.url);
const LEGS = ['lf', 'lm', 'lr', 'rf', 'rm', 'rr'];
const KINDS = ['coxa_yaw', 'femur_pitch', 'tibia_pitch'];
const NAMES = {lf: 'Left front', lm: 'Left middle', lr: 'Left rear', rf: 'Right front', rm: 'Right middle', rr: 'Right rear'};
const TINT = {body: 0xcad7e5, coxa: 0xb49ddc, femur: 0x8dbdff, tibia: 0x80ddba, tibia_push_lever: 0xf3bc72, tibia_pushrod: 0xea9490};
const state = {ready: false, errors: [], meshes: 0, pose: 'V2 reset stance', q: {}, robot: null, model: 'linkage', playing: false};
window.__inspection = state;
const error = message => {
  const text = String(message);
  state.errors.push(text); $('errors').style.display = 'block';
  $('errors').textContent = state.errors.join('\n');
  $('status').textContent = 'Model inspection needs attention';
};
window.addEventListener('error', event => error(event.message));
window.addEventListener('unhandledrejection', event => error(event.reason));

const stage = $('stage');
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x121820);
const camera = new THREE.PerspectiveCamera(36, 1, 0.005, 30); camera.up.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace; stage.prepend(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.12; controls.minDistance = 0.2; controls.maxDistance = 4;
controls.target.set(0, 0, 0.12);
scene.add(new THREE.HemisphereLight(0xecf5ff, 0x43505f, 2));
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(1, -1.3, 2); scene.add(key);
const rim = new THREE.DirectionalLight(0x9fcfff, 1.0); rim.position.set(-1, 0.6, 1); scene.add(rim);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshStandardMaterial({color: 0x17212c, roughness: 1, side: THREE.DoubleSide}));
// Numerical ground is z=0; polygon offset prevents grid z-fighting.
ground.material.polygonOffset = true; ground.material.polygonOffsetFactor = 1; ground.material.polygonOffsetUnits = 1; scene.add(ground);
const grid = new THREE.GridHelper(2, 40, 0x506171, 0x2b3b4c); grid.rotation.x = Math.PI / 2; scene.add(grid);
scene.add(new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, -0.3, 0.002), 0.18, 0x80ddba, 0.035, 0.025));
const resize = () => {const {width, height} = stage.getBoundingClientRect(); camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height);};
new ResizeObserver(resize).observe(stage); resize();
const view = position => {camera.position.set(...position); controls.target.set(0, 0, 0.12); controls.update();};
$('view-iso').onclick = () => view([0.90, -1.05, 0.68]);
$('view-top').onclick = () => view([0, -0.001, 1.40]);
$('view-front').onclick = () => view([0, -1.35, 0.28]);
$('view-side').onclick = () => view([1.35, 0, 0.28]);
view([0.90, -1.05, 0.68]);

let showCollisions = false, showVisuals = true, useTints = false;
let stance, limits, modelSpecs, footSpheres = [], pendingMeshes = 0, urdfLoaded = false;
let loadGeneration = 0, animationPhase = 0, animationBase = {}, lastAnimationTime = 0;
const cachedGeometries = new Set();
const cache = new Map(), collisionMaterial = new THREE.MeshBasicMaterial({color: 0xff8b91, wireframe: true, transparent: true, opacity: 0.8, depthTest: false});
const colliderAncestor = object => {for (let node = object; node; node = node.parent) if (node.isURDFCollider) return node; return null;};
const linkAncestor = object => {for (let node = object; node; node = node.parent) if (node.isURDFLink) return node; return null;};
function paint() {
  if (!state.robot) return;
  state.robot.traverse(object => {
    if (!object.isMesh) return;
    const collision = Boolean(colliderAncestor(object)); object.userData.collision = collision;
    object.visible = collision ? showCollisions : showVisuals;
    if (collision) {object.material = collisionMaterial; object.renderOrder = 1; return;}
    if (!object.userData.cadColor) {object.material = object.material.clone(); object.userData.cadColor = object.material.color.clone();}
    const link = linkAncestor(object)?.name || 'body'; const kind = link === 'body' ? 'body' : link.split('_').slice(1).join('_');
    if (useTints) object.material.color.setHex(TINT[kind] || TINT.body);
    else object.material.color.copy(object.userData.cadColor);
  });
}
$('collisions').onclick = () => {showCollisions = !showCollisions; $('collisions').setAttribute('aria-pressed', showCollisions); paint();};
$('tints').onclick = () => {useTints = !useTints; $('tints').setAttribute('aria-pressed', useTints); paint();};
$('visuals').onclick = () => {showVisuals = !showVisuals; $('visuals').setAttribute('aria-pressed', showVisuals); paint();};
function updateFootClearance() {
  if (!state.robot || !footSpheres.length) return;
  state.robot.updateMatrixWorld(true);
  const lowest = Math.min(...footSpheres.map(({link, center, radius}) => center.clone().applyMatrix4(state.robot.links[link].matrixWorld).z - radius));
  state.footClearanceM = lowest; $('foot-clearance').textContent = `${(lowest * 1000).toFixed(3)} mm`;
  $('foot-clearance').style.color = lowest < 0 ? '#ff9ba7' : '#c7f4df';
}
function setPose(values, label) {
  state.pose = label; $('pose-label').textContent = label;
  for (const leg of LEGS) for (const kind of KINDS) {
    const name = `${leg}_${kind}`, value = values[name] ?? 0; state.q[name] = value;
    state.robot?.joints[name]?.setJointValue(value);
    const input = $(`s_${name}`), output = $(`v_${name}`);
    if (input) input.value = value; if (output) output.textContent = value.toFixed(3);
  }
  // Explicit mimic values also expose the expected relationship for inspection.
  for (const leg of LEGS) {
    const knee = state.q[`${leg}_tibia_pitch`];
    state.robot?.joints[`${leg}_tibia_lever_pivot`]?.setJointValue(knee);
    state.robot?.joints[`${leg}_tibia_rod_pivot`]?.setJointValue(-knee);
  }
  updateFootClearance();
}
function resetStance() {if (stance) setPose(Object.fromEntries(LEGS.flatMap(leg => KINDS.map(kind => [`${leg}_${kind}`, stance[`${kind}_rad`]]))), 'V2 reset stance');}
$('reset-stance').onclick = () => {pauseAnimation(); resetStance();}; $('cad-zero').onclick = () => {pauseAnimation(); setPose({}, 'CAD zero');};
function makeSliders() {
  $('sliders').replaceChildren();
  for (const leg of LEGS) {
    const section = document.createElement('section'); section.className = 'leg';
    const title = document.createElement('h3'); title.textContent = `${NAMES[leg]} · ${leg}`; section.append(title);
    for (const kind of KINDS) {
      const name = `${leg}_${kind}`, row = document.createElement('div'); row.className = 'joint';
      const label = document.createElement('label'); label.htmlFor = `s_${name}`; label.textContent = kind.split('_')[0];
      const input = document.createElement('input'); input.type = 'range'; input.id = `s_${name}`; input.min = limits[kind].lower; input.max = limits[kind].upper; input.step = '0.000001'; input.value = stance[`${kind}_rad`]; input.setAttribute('aria-label', `${NAMES[leg]} ${kind}`);
      const output = document.createElement('output'); output.id = `v_${name}`; output.htmlFor = input.id;
      input.oninput = () => {pauseAnimation(); state.q[name] = Number(input.value); setPose(state.q, 'Manual joint pose');};
      row.append(label, input, output); section.append(row);
    }
    $('sliders').append(section);
  }
}
const fetchChecked = async url => {const response = await fetch(url); if (!response.ok) throw new Error(`${response.status}: ${url}`); return response;};
function finishLoading() {
  if (!urdfLoaded || pendingMeshes !== 0 || state.errors.length || state.ready) return;
  paint(); setPose(state.q, state.pose); state.ready = true; document.body.dataset.ready = 'true';
  $('status').textContent = `Ready · ${state.meshes} local meshes · ${state.model === 'linkage' ? '18 joints + 12 mimics' : '18 independent joints'}`;
  for (const mode of ['linkage', 'serial']) $(`model-${mode}`).disabled = false;
  if (state.pose === 'V2 reset stance' && state.footClearanceM < stance.reset_clearance_m - 1e-8) error('Rendered stance violates the 5 mm reset clearance.');
}
function disposeRobot(robot) {
  if (!robot) return;
  scene.remove(robot);
  const geometries = new Set(), materials = new Set();
  robot.traverse(object => {
    if (object.geometry && !cachedGeometries.has(object.geometry)) geometries.add(object.geometry);
    if (object.material) for (const material of (Array.isArray(object.material) ? object.material : [object.material])) if (material !== collisionMaterial) materials.add(material);
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
}
async function loadModel(mode = 'linkage') {
  pauseAnimation(); const generation = ++loadGeneration;
  for (const model of ['linkage', 'serial']) $(`model-${model}`).disabled = true;
  state.ready = false; document.body.dataset.ready = 'false'; urdfLoaded = false;
  if (!stance) {
    [stance, limits, modelSpecs] = await Promise.all([
      ...['stance_v2.json', 'joint_limits.json'].map(async name => (await fetchChecked(new URL(name, BASE))).json()),
      fetchChecked(new URL('./inspection_models.json', import.meta.url)).then(response => response.json()),
    ]);
    if (modelSpecs.serial.sha256 !== stance.source_urdf_sha256) throw new Error('Serial model pin differs from the v2 stance source.');
    makeSliders(); resetStance();
  }
  const spec = modelSpecs[mode], url = new URL(spec.urdf, BASE), text = await (await fetchChecked(url)).text();
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  if (generation !== loadGeneration) return;
  state.urdfSha256 = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
  if (state.urdfSha256 !== spec.sha256) throw new Error(`${mode} URDF differs from its inspection source pin.`);
  const xml = new DOMParser().parseFromString(text, 'application/xml');
  if (xml.querySelector('parsererror')) throw new Error('URDF XML parse failed');
  const links = [...xml.querySelectorAll('robot > link')], joints = [...xml.querySelectorAll('robot > joint')];
  const independent = joints.filter(joint => !joint.querySelector('mimic')), mimics = joints.filter(joint => joint.querySelector('mimic'));
  if (links.length !== spec.links || joints.length !== spec.joints || independent.length !== 18 || mimics.length !== spec.mimic_joints) throw new Error('Model link/joint/mimic counts differ from its inspection contract.');
  for (const joint of independent) {
    const name = joint.getAttribute('name'), kind = name.split('_').slice(1).join('_'), bounds = joint.querySelector('limit');
    if (!limits[kind] || Number(bounds.getAttribute('lower')) !== limits[kind].lower || Number(bounds.getAttribute('upper')) !== limits[kind].upper) throw new Error(`Joint limit mismatch: ${name}`);
  }
  for (const joint of mimics) {
    const name = joint.getAttribute('name'), leg = name.split('_')[0], mimic = joint.querySelector('mimic');
    const multiplier = name.endsWith('_tibia_lever_pivot') ? 1 : name.endsWith('_tibia_rod_pivot') ? -1 : NaN;
    if (mimic.getAttribute('joint') !== `${leg}_tibia_pitch` || Number(mimic.getAttribute('multiplier')) !== multiplier || Number(mimic.getAttribute('offset')) !== 0) throw new Error(`Unexpected linkage mimic: ${name}`);
  }
  footSpheres = [];
  for (const link of links.filter(link => link.getAttribute('name').endsWith('_tibia'))) {
    for (const collision of link.querySelectorAll(':scope > collision')) {
      const sphere = collision.querySelector('geometry > sphere'); if (!sphere) continue;
      const xyz = (collision.querySelector('origin')?.getAttribute('xyz') || '0 0 0').trim().split(/\s+/).map(Number);
      footSpheres.push({link: link.getAttribute('name'), center: new THREE.Vector3(...xyz), radius: Number(sphere.getAttribute('radius'))});
    }
  }
  if (footSpheres.length !== 12) throw new Error('Expected twelve foot-pad spheres');
  $('counts').textContent = `${links.length} / ${joints.length}${mode === 'linkage' ? ' (12 mimic)' : ''}`;
  $('mass').textContent = `${[...xml.querySelectorAll('inertial > mass')].reduce((sum, node) => sum + Number(node.getAttribute('value')), 0).toFixed(3)} kg`;
  $('plate-height').textContent = `${(stance.reset_root_height_m * 1000).toFixed(3)} mm`;
  const loader = new URDFLoader(); loader.packages = {hexapod_mkii_assy: BASE.href}; loader.parseCollision = true;
  loader.loadMeshCb = (path, manager, done) => {
    if (!cache.has(path)) cache.set(path, new Promise((resolve, reject) => new STLLoader(manager).load(path, geometry => {geometry.computeVertexNormals(); cachedGeometries.add(geometry); state.meshes++; resolve(geometry);}, undefined, reject)));
    pendingMeshes++;
    cache.get(path).then(geometry => {const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: 0xc0ccdb, metalness: 0.28, roughness: 0.5})); mesh.userData.file = path.split('/').pop(); done(mesh);}).catch(reason => {error(`Mesh failed: ${path}\n${reason}`); done(null);}).finally(() => {pendingMeshes--; if (!state.errors.length) $('status').textContent = `Loading local CAD meshes · ${state.meshes} / 77`; finishLoading();});
  };
  const previous = state.robot; state.robot = loader.parse(text, BASE.href); state.model = mode;
  state.robot.position.z = stance.reset_root_height_m; scene.add(state.robot); disposeRobot(previous);
  for (const model of ['linkage', 'serial']) $(`model-${model}`).setAttribute('aria-pressed', model === mode);
  $('model-note').textContent = mode === 'linkage'
    ? 'Push lever and rod follow the knee; simulation still uses the serial approximation.'
    : 'Simulation approximation: push lever and rod are fixed to the femur, so their visual connection opens as the knee moves.';
  urdfLoaded = true; setPose(state.q, state.pose); finishLoading();
}
for (const mode of ['linkage', 'serial']) $(`model-${mode}`).onclick = () => {if (state.ready && state.model !== mode) loadModel(mode).catch(error);};
function pauseAnimation() {
  state.playing = false; $('animation-play').textContent = 'Play'; $('animation-play').setAttribute('aria-pressed', 'false');
}
function beginAnimation() {
  if (!state.ready) return;
  animationBase = {...state.q};
  const mode = $('animation-mode').value, kind = mode === 'knee' ? 'tibia_pitch' : $('animation-joint').value;
  const leg = $('animation-leg').value === 'all' ? 'lf' : $('animation-leg').value;
  animationPhase = mode === 'pose' ? 0 : phaseForJoint(state.q[`${leg}_${kind}`], limits[kind].lower, limits[kind].upper);
  lastAnimationTime = performance.now(); state.playing = true;
  $('animation-play').textContent = 'Pause'; $('animation-play').setAttribute('aria-pressed', 'true');
}
$('animation-play').onclick = () => state.playing ? pauseAnimation() : beginAnimation();
for (const id of ['animation-mode', 'animation-leg', 'animation-joint']) $(id).onchange = () => {
  const wasPlaying = state.playing; pauseAnimation();
  $('animation-joint').disabled = $('animation-mode').value !== 'joint';
  $('animation-leg').disabled = $('animation-mode').value === 'pose';
  if (wasPlaying) beginAnimation();
};
$('animation-speed').oninput = () => {$('animation-speed-label').textContent = `${Number($('animation-speed').value)}×`;};
function animate(now) {
  if (!state.playing || !state.ready) return;
  const elapsed = Math.min(0.1, Math.max(0, (now - lastAnimationTime) / 1000)); lastAnimationTime = now;
  animationPhase += elapsed * Number($('animation-speed').value) * 2 * Math.PI / 8;
  const stancePose = Object.fromEntries(LEGS.flatMap(leg => KINDS.map(kind => [`${leg}_${kind}`, stance[`${kind}_rad`]])));
  const mode = $('animation-mode').value;
  setPose(animationPose({mode, phase: animationPhase, basePose: animationBase, stancePose, legs: LEGS, kinds: KINDS, limits, leg: $('animation-leg').value, joint: $('animation-joint').value}), mode === 'pose' ? 'Kinematic pose cycle' : 'Kinematic joint sweep');
}
const raycaster = new THREE.Raycaster(); let pointerStart;
renderer.domElement.addEventListener('pointerdown', event => {pointerStart = [event.clientX, event.clientY];});
renderer.domElement.addEventListener('pointerup', event => {
  if (!state.robot || !pointerStart || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) return;
  const rect = renderer.domElement.getBoundingClientRect(); raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
  const hit = raycaster.intersectObject(state.robot, true).find(hit => hit.object.isMesh && hit.object.visible && !hit.object.userData.collision);
  if (hit) $('selected').textContent = `Link: ${linkAncestor(hit.object)?.name} · Mesh: ${hit.object.userData.file || 'primitive'}`;
});
renderer.setAnimationLoop(now => {animate(now); controls.update(); renderer.render(scene, camera);});
loadModel().catch(error);

// Shared STL geometry is owned once per URL; dispose each GPU object once.
window.addEventListener('pagehide', () => {
  renderer.setAnimationLoop(null); controls.dispose();
  const geometries = new Set(), materials = new Set();
  scene.traverse(object => {
    if (object.geometry) geometries.add(object.geometry);
    if (object.material) for (const material of (Array.isArray(object.material) ? object.material : [object.material])) materials.add(material);
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
  renderer.dispose();
});
