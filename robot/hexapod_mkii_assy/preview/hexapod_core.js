// Shared engine for the Hexapod MKII pages (viewer and showcase). Inlined by
// robot/tools/pack_urdf_viewer.py; expects THREE (r160 module) to be passed in.
function createHexapodCore(THREE, MODEL, BLOB, container, opts = {}) {
  const LEGS = ["lf","lm","lr","rf","rm","rr"];
  const JOINTS = ["coxa_yaw","femur_pitch","tibia_pitch"];
  const BODIES = ["coxa","femur","tibia","tibia_push_lever","tibia_pushrod"];
  const CLASS_TINT = {body:0xc9ced6, coxa:0xb48ee6, femur:0x5aa9e6, tibia:0x6fd39a, tibia_push_lever:0xf2b544, tibia_pushrod:0xf07c5a};
  const LEG_TINTS = {body:0xc9ced6, lf:0xf07c5a, lm:0xf2b544, lr:0x8fd35a, rf:0x5aa9e6, rm:0xb48ee6, rr:0x5ad3c9};
  const FASTENER = /screw|nut|washer|standoff__|cirpattern|revolve|mirror1|pem_|cut_extrude|bearing_insert/i;
  const linkClass = n => n === "body" ? "body" : n.replace(/^(lf|lm|lr|rf|rm|rr)_/, "");
  const legOf = n => n === "body" ? null : n.slice(0, 2);
  const partClass = n => FASTENER.test(n) ? "fastener" : /^motor_/.test(n) ? "motor" : /cover|screw_head_cap/.test(n) ? "cover" : "structure";

  // ---------- unpack meshes: 16-bit positions, indexed triangles, feature edges; PCA axis per part
  function decode(b64){ const bin = atob(b64); const out = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i); return out.buffer; }
  const buf = decode(BLOB);
  function pca(pos, n){
    const step = Math.max(1, Math.floor(n / 3000)); let cnt = 0; const c = [0,0,0];
    for (let i = 0; i < n; i += step) { c[0] += pos[i*3]; c[1] += pos[i*3+1]; c[2] += pos[i*3+2]; cnt++; }
    c[0] /= cnt; c[1] /= cnt; c[2] /= cnt;
    const S = [[0,0,0],[0,0,0],[0,0,0]];
    for (let i = 0; i < n; i += step) { const d = [pos[i*3]-c[0], pos[i*3+1]-c[1], pos[i*3+2]-c[2]]; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += d[a]*d[b]; }
    // Jacobi eigen-decomposition of the 3x3 symmetric matrix
    const A = S.map(r => r.slice()), V = [[1,0,0],[0,1,0],[0,0,1]];
    for (let it = 0; it < 40; it++) {
      let p = 0, q = 1, m = Math.abs(A[0][1]);
      if (Math.abs(A[0][2]) > m) { p = 0; q = 2; m = Math.abs(A[0][2]); }
      if (Math.abs(A[1][2]) > m) { p = 1; q = 2; m = Math.abs(A[1][2]); }
      if (m < 1e-14) break;
      const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]), cs = Math.cos(th), sn = Math.sin(th);
      for (let k = 0; k < 3; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = cs*akp - sn*akq; A[k][q] = sn*akp + cs*akq; }
      for (let k = 0; k < 3; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = cs*apk - sn*aqk; A[q][k] = sn*apk + cs*aqk; }
      for (let k = 0; k < 3; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = cs*vkp - sn*vkq; V[k][q] = sn*vkp + cs*vkq; }
    }
    const ev = [0,1,2].map(i => ({l: A[i][i] / cnt, v: new THREE.Vector3(V[0][i], V[1][i], V[2][i]).normalize()})).sort((a, b) => b.l - a.l);
    const cen = new THREE.Vector3(...c);
    // head heuristic for elongated parts: the end with the wider cross-section is the head
    let axis = null, kind = "radial";
    if (ev[0].l > 3 * ev[1].l) {
      kind = "long"; axis = ev[0].v.clone();
      const ts = [], rs = [];
      for (let i = 0; i < n; i += step) { const d = new THREE.Vector3(pos[i*3], pos[i*3+1], pos[i*3+2]).sub(cen); const t = d.dot(axis); ts.push(t); rs.push(d.addScaledVector(axis, -t).length()); }
      const order = ts.map((t, i) => i).sort((a, b) => ts[a] - ts[b]); const k = Math.max(3, Math.floor(order.length * 0.2));
      const rBot = order.slice(0, k).reduce((s, i) => s + rs[i], 0) / k, rTop = order.slice(-k).reduce((s, i) => s + rs[i], 0) / k;
      if (rTop > rBot * 1.15) { /* head at +axis */ } else if (rBot > rTop * 1.15) axis.negate(); else kind = "long-sym";
    } else if (ev[2].l < 0.15 * ev[1].l) { kind = "flat"; axis = ev[2].v.clone(); }
    return {cen, axis, kind};
  }
  const geoms = MODEL.meshes.map(m => {
    const q = new Int16Array(buf, m.op, m.nv * 3);
    const pos = new Float32Array(m.nv * 3);
    for (let i = 0; i < m.nv; i++) for (let k = 0; k < 3; k++) pos[i*3+k] = m.min[k] + (q[i*3+k] + 32768) / 65535 * m.span[k];
    const idx = m.i32 ? new Uint32Array(buf, m.oi, m.nt * 3) : new Uint16Array(buf, m.oi, m.nt * 3);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx.slice(), 1));
    g.computeVertexNormals(); g.computeBoundingSphere();
    const eIdx = m.i32 ? new Uint32Array(buf, m.oe, m.ne * 2) : new Uint16Array(buf, m.oe, m.ne * 2);
    const eg = new THREE.BufferGeometry();
    eg.setAttribute("position", g.getAttribute("position"));
    eg.setIndex(new THREE.BufferAttribute(eIdx.slice(), 1));
    g.userData.edges = eg; g.userData.pca = pca(pos, m.nv);
    return g;
  });

  // ---------- renderer, scene, lights
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true, powerPreference:"high-performance", preserveDrawingBuffer:!!opts.preserveDrawingBuffer});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, container.clientWidth / container.clientHeight, 0.01, 60);
  camera.up.set(0,0,1);
  const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x2a2622, 0.85); scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 1.7); key.position.set(1.0, -1.3, 1.9); scene.add(key);
  const fill = new THREE.DirectionalLight(0x8fb7ff, 0.45); fill.position.set(-1.1, 0.9, 0.6); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffd9a0, 0.35); rim.position.set(0.3, 1.2, -0.4); scene.add(rim);
  key.castShadow = true; key.shadow.mapSize.set(2048, 2048);
  // studio environment: a procedural softbox room baked with PMREM, for product-shot material response
  let envTex = null;
  function studioEnvironment(){
    if (envTex) return envTex;
    const pm = new THREE.PMREMGenerator(renderer); pm.compileEquirectangularShader();
    const room = new THREE.Scene();
    room.add(new THREE.Mesh(new THREE.BoxGeometry(10, 10, 10), new THREE.MeshBasicMaterial({color:0x3a3f47, side:THREE.BackSide})));
    const soft = (w, h, x, y, z, ry, i) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({color:new THREE.Color(i, i, i)})); m.position.set(x, y, z); m.lookAt(0, 0, 0); room.add(m); };
    soft(6, 4, 0, 0, 4.9, 0, 3.2);      // top light
    soft(3, 5, -4.9, -2, 1.5, 0, 1.6);  // key softbox
    soft(3, 5, 4.9, 2.5, 1.0, 0, 0.9);  // fill
    soft(6, 2, 0, 4.9, 0.5, 0, 0.5);    // rim strip
    envTex = pm.fromScene(room, 0.06).texture; pm.dispose();
    return envTex;
  }
  Object.assign(key.shadow.camera, {left:-1.2, right:1.2, top:1.2, bottom:-1.2, near:0.1, far:8}); key.shadow.bias = -0.0005; key.shadow.radius = 3;

  // ---------- materials and styles
  const PO = {polygonOffset:true, polygonOffsetFactor:1, polygonOffsetUnits:1};
  const cadMats = MODEL.colors.map(c => new THREE.MeshStandardMaterial({color:new THREE.Color(c), metalness:0.28, roughness:0.58, flatShading:true, ...PO}));
  const tintMats = Object.fromEntries(Object.entries(CLASS_TINT).map(([k, c]) => [k, new THREE.MeshStandardMaterial({color:new THREE.Color(c), metalness:0.15, roughness:0.65, flatShading:true, ...PO})]));
  const legMats = Object.fromEntries(Object.entries(LEG_TINTS).map(([k, c]) => [k, new THREE.MeshStandardMaterial({color:new THREE.Color(c), metalness:0.15, roughness:0.65, flatShading:true, ...PO})]));
  const whiteFlat = new THREE.MeshBasicMaterial({color:0xffffff, ...PO});
  const whiteShaded = new THREE.MeshStandardMaterial({color:0xf4f6f8, metalness:0.0, roughness:0.9, flatShading:true, ...PO});
  const hiddenMat = new THREE.MeshBasicMaterial({colorWrite:false, ...PO});
  const ghostMat = new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:0.10, depthWrite:false, ...PO});
  const edgeMat = new THREE.LineBasicMaterial({color:0xe3e9f2, transparent:true, opacity:0.8});
  const collMat = new THREE.MeshBasicMaterial({color:0xf2b544, wireframe:true, transparent:true, opacity:0.55, depthTest:false});
  const hiMat = new THREE.MeshStandardMaterial({color:0xffffff, emissive:new THREE.Color(0xff3b6b), emissiveIntensity:0.9, flatShading:true, ...PO});
  const STYLES = {
    cad:      {label:"CAD colours",   look:"dark",      edges:false, mat:m => m.userData.cad},
    tint:     {label:"Tint by link",  look:"dark",      edges:false, mat:m => tintMats[m.userData.cls]},
    tintLeg:  {label:"Tint by leg",   look:"dark",      edges:false, mat:m => legMats[m.userData.leg || "body"]},
    white:    {label:"White + edges", look:"light",     edges:true,  mat:() => whiteFlat},
    shaded:   {label:"White shaded",  look:"light",     edges:true,  mat:() => whiteShaded},
    hidden:   {label:"Hidden line",   look:"light",     edges:true,  mat:() => hiddenMat},
    ghost:    {label:"Ghost",         look:"light",     edges:true,  mat:() => ghostMat},
    blueprint:{label:"Blueprint",     look:"blueprint", edges:true,  mat:() => hiddenMat},
  };
  const LOOK_BG = {dark:0x0f1318, light:0xfafbfc, blueprint:0x0b2545};

  // ---------- kinematic tree
  const linkObj = {}, jointObj = {}, meshes = [], collObjs = [], axisObjs = [];
  const robot = new THREE.Group(); scene.add(robot);
  for (const [name, L] of Object.entries(MODEL.links)) {
    const g = new THREE.Group(); g.name = name; linkObj[name] = g;
    const cls = linkClass(name);
    for (const [mi, p, q, ci] of L.v) {
      const m = new THREE.Mesh(geoms[mi], cadMats[ci]);
      m.position.set(...p); m.quaternion.set(...q);
      const e = new THREE.LineSegments(geoms[mi].userData.edges, edgeMat);
      e.position.copy(m.position); e.quaternion.copy(m.quaternion); e.visible = false;
      const pn = MODEL.meshes[mi].n;
      m.userData = {part: pn, link: name, cls, leg: legOf(name), cad: cadMats[ci], ci, fastener: FASTENER.test(pn), pclass: partClass(pn), edge: e, rest: m.position.clone(), hidden: false, mi};
      g.add(m); g.add(e); meshes.push(m);
    }
    for (const [t, p, q, s] of L.c) {
      let geo;
      if (t === "box") geo = new THREE.BoxGeometry(s[0], s[1], s[2]);
      else if (t === "cyl") { geo = new THREE.CylinderGeometry(s[0], s[0], s[1], 28, 1, true); geo.rotateX(Math.PI/2); }
      else geo = new THREE.SphereGeometry(s[0], 14, 10);
      const c = new THREE.Mesh(geo, collMat); c.position.set(...p); c.quaternion.set(...q); c.visible = false; c.userData.coll = true;
      g.add(c); collObjs.push(c);
    }
  }
  const jointVal = {};
  for (const J of MODEL.joints) {
    const jg = new THREE.Group(); jg.name = J.n;
    jg.position.set(...J.p); jg.quaternion.set(...J.q);
    jg.userData = {base: jg.quaternion.clone(), axis: new THREE.Vector3(...J.axis).normalize(), spec: J};
    linkObj[J.parent].add(jg); jg.add(linkObj[J.child]); jointObj[J.n] = jg;
    jointVal[J.n] = 0;
    const ax = new THREE.ArrowHelper(jg.userData.axis, new THREE.Vector3(), 0.05, 0xf2b544, 0.014, 0.008);
    ax.visible = false; jg.add(ax); axisObjs.push(ax);
  }
  robot.add(linkObj.body);
  function applyJoints(){
    for (const J of MODEL.joints) {
      let v = jointVal[J.n];
      if (J.mimic) v = jointVal[J.mimic.j] * J.mimic.k + J.mimic.o, jointVal[J.n] = v;
      const jg = jointObj[J.n];
      jg.quaternion.copy(jg.userData.base).multiply(new THREE.Quaternion().setFromAxisAngle(jg.userData.axis, v));
    }
  }
  applyJoints();
  robot.updateMatrixWorld(true);

  // ---------- rest geometry for the explode: link centroids and per-part directions
  const tmpV = new THREE.Vector3(), tmpQ = new THREE.Quaternion();
  const linkRest = {};
  const bodyCentre = new THREE.Vector3();
  {
    const acc = {};
    for (const m of meshes) {
      const cLocal = m.geometry.boundingSphere.center.clone().applyQuaternion(m.quaternion).add(m.position);
      m.userData.cLocal = cLocal;
      (acc[m.userData.link] ??= []).push(cLocal);
    }
    for (const [name, list] of Object.entries(acc)) {
      const centreLocal = list.reduce((s, v) => s.add(v), new THREE.Vector3()).multiplyScalar(1 / list.length);
      linkRest[name] = {centreLocal, centreWorld: linkObj[name].localToWorld(centreLocal.clone())};
    }
    bodyCentre.copy(linkRest.body.centreWorld);
    for (const [name, r] of Object.entries(linkRest)) {
      const d = r.centreWorld.clone().sub(bodyCentre); d.z = 0;
      if (d.lengthSq() < 1e-9) d.set(1, 0, 0);
      d.normalize(); d.z = 0.28; d.normalize();
      r.dirWorld = d;
      r.level = name === "body" ? 0 : ({coxa:1, femur:1, tibia:1, tibia_push_lever:0.55, tibia_pushrod:0.55})[linkClass(name)];
    }
    for (const m of meshes) {
      const off = m.userData.cLocal.clone().sub(linkRest[m.userData.link].centreLocal);
      const n = off.length() > 1e-6 ? off.clone().normalize() : new THREE.Vector3(0, 0, 1);
      const p = m.geometry.userData.pca;
      if (m.userData.fastener && p.axis) {
        // fasteners leave along their own axis: screws toward their head, washers and nuts along the hole axis outward
        const a = p.axis.clone().applyQuaternion(m.quaternion);
        if (p.kind !== "long" && a.dot(n) < 0) a.negate();
        m.userData.dir = a.multiplyScalar(0.06 + 0.35 * off.length());
      } else {
        m.userData.dir = off.multiplyScalar(0.7).addScaledVector(n, 0.05); // damped proportional plus a minimum gap
      }
    }
  }
  const state = {style:"cad", edgesOn:false, explodeLinks:0, explodeParts:0, labelsOn:false, shadows:false, faceColor:"#f4f6f8", edgeColor:null, background:"auto", gridOn:true, comOn:false, section:{axis:"off", t:0, flip:1}};
  const LINK_STEP = 0.14;
  function applyExplode(){
    robot.updateMatrixWorld(true);
    for (const [name, g] of Object.entries(linkObj)) {
      if (name === "body") continue;
      g.parent.getWorldQuaternion(tmpQ);
      tmpV.copy(linkRest[name].dirWorld).applyQuaternion(tmpQ.invert());
      g.position.copy(tmpV).multiplyScalar(state.explodeLinks * LINK_STEP * linkRest[name].level);
    }
    for (const m of meshes) {
      m.position.copy(m.userData.rest).addScaledVector(m.userData.dir, state.explodeParts);
      m.userData.edge.position.copy(m.position);
    }
  }

  // ---------- floor, helpers, shadows, centre of mass, labels
  const box = new THREE.Box3().setFromObject(robot);
  const centre = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2;
  const size = box.getSize(new THREE.Vector3());
  const grid = new THREE.GridHelper(3.0, 30, 0x3a4352, 0x232a35);
  grid.rotation.x = Math.PI/2; grid.position.set(0, 0, box.min.z - 0.002); grid.material.transparent = true; scene.add(grid);
  const axes = new THREE.AxesHelper(0.12); axes.position.z = box.min.z; scene.add(axes);
  const shadowPlane = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.ShadowMaterial({opacity:0.22}));
  shadowPlane.position.z = box.min.z - 0.001; shadowPlane.receiveShadow = true; shadowPlane.visible = false; scene.add(shadowPlane);
  const comGroup = new THREE.Group(); comGroup.visible = false; scene.add(comGroup);
  comGroup.add(new THREE.Mesh(new THREE.SphereGeometry(0.012, 20, 14), new THREE.MeshBasicMaterial({color:0xff3b6b, depthTest:false})));
  for (const [a, b] of [[[-0.05,0,0],[0.05,0,0]], [[0,-0.05,0],[0,0.05,0]], [[0,0,-0.05],[0,0,0.05]]]) comGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]), new THREE.LineBasicMaterial({color:0xff3b6b, depthTest:false})));
  const totalMass = Object.values(MODEL.links).reduce((s, l) => s + l.m, 0);
  function updateCom(){
    if (!comGroup.visible) return;
    const acc = new THREE.Vector3();
    for (const [name, L] of Object.entries(MODEL.links)) acc.addScaledVector(linkObj[name].localToWorld(new THREE.Vector3(...(L.com || [0,0,0]))), L.m);
    comGroup.position.copy(acc.multiplyScalar(1 / totalMass));
  }
  function textSprite(text, fg, bg, scale = 2200){
    const c = document.createElement("canvas"); const ctx = c.getContext("2d"); const font = "600 40px Barlow, Helvetica, Arial, sans-serif";
    ctx.font = font; const w = Math.ceil(ctx.measureText(text).width) + 36; c.width = w; c.height = 60;
    ctx.font = font; ctx.fillStyle = bg; ctx.beginPath(); ctx.roundRect(0, 0, w, 60, 10); ctx.fill();
    ctx.fillStyle = fg; ctx.textBaseline = "middle"; ctx.fillText(text, 18, 32);
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:tex, depthTest:false, transparent:true, sizeAttenuation:false}));
    sp.scale.set(w / scale, 60 / scale, 1); sp.renderOrder = 998; return sp;
  }
  const labels = {};
  function makeLabels(){
    const light = currentLook !== "dark";
    for (const sp of Object.values(labels)) scene.remove(sp);
    for (const name of Object.keys(MODEL.links)) labels[name] = textSprite(name === "body" ? "body" : name.replace("_", " · ").replace("tibia_push_lever", "lever").replace("tibia_pushrod", "rod"), light ? "#f7f8fa" : "#1b2129", light ? "rgba(27,33,41,.88)" : "rgba(242,244,247,.9)");
    for (const sp of Object.values(labels)) { sp.visible = false; scene.add(sp); }
  }
  function updateLabels(){
    for (const [name, sp] of Object.entries(labels)) {
      const anyVisible = state.labelsOn && meshes.some(m => m.userData.link === name && m.visible);
      sp.visible = anyVisible;
      if (sp.visible) sp.position.copy(linkObj[name].localToWorld(linkRest[name].centreLocal.clone())).addScaledVector(new THREE.Vector3(0, 0, 1), 0.045);
    }
  }

  // ---------- camera
  const view = {target: centre.clone(), theta: -1.15, phi: 1.08, r: 1};
  function fitDistance(r = radius){
    const vHalf = THREE.MathUtils.degToRad(camera.fov) / 2, hHalf = Math.atan(Math.tan(vHalf) * camera.aspect);
    return r / Math.sin(Math.min(vHalf, hHalf)) * 1.08;
  }
  view.r = fitDistance();
  function updateCamera(){
    const sp = Math.sin(view.phi), cp = Math.cos(view.phi);
    camera.position.set(view.target.x + view.r*sp*Math.cos(view.theta), view.target.y + view.r*sp*Math.sin(view.theta), view.target.z + view.r*cp);
    camera.lookAt(view.target);
  }
  function frameVisible(margin = 0.9){
    const b = new THREE.Box3(); const s = new THREE.Sphere();
    for (const m of meshes) { if (!m.visible) continue; s.copy(m.geometry.boundingSphere).applyMatrix4(m.matrixWorld); b.expandByPoint(s.center.clone().addScalar(s.radius)); b.expandByPoint(s.center.clone().subScalar(s.radius)); }
    if (b.isEmpty()) return;
    view.target.copy(b.getCenter(new THREE.Vector3()));
    view.r = fitDistance(b.getSize(new THREE.Vector3()).length() / 2);
    const v = new THREE.Vector3(), right = new THREE.Vector3(), up = new THREE.Vector3();
    for (let it = 0; it < 4; it++) {
      updateCamera(); camera.updateMatrixWorld(true); camera.updateProjectionMatrix();
      right.setFromMatrixColumn(camera.matrixWorld, 0); up.setFromMatrixColumn(camera.matrixWorld, 1);
      let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
      for (const m of meshes) {
        if (!m.visible) continue;
        s.copy(m.geometry.boundingSphere).applyMatrix4(m.matrixWorld);
        for (const [ox, oy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
          v.copy(s.center).addScaledVector(right, ox * s.radius).addScaledVector(up, oy * s.radius).project(camera);
          minX = Math.min(minX, v.x); maxX = Math.max(maxX, v.x); minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y);
        }
      }
      if (minX > maxX) return;
      const vHalf = THREE.MathUtils.degToRad(camera.fov) / 2, hHalf = Math.atan(Math.tan(vHalf) * camera.aspect);
      view.target.addScaledVector(right, (minX + maxX) / 2 * view.r * Math.tan(hHalf)).addScaledVector(up, (minY + maxY) / 2 * view.r * Math.tan(vHalf));
      view.r *= Math.max((maxX - minX) / 2, (maxY - minY) / 2) / margin;
    }
    updateCamera();
  }

  // ---------- look, style, visibility
  let currentLook = "dark";
  const linkVisible = Object.fromEntries(Object.keys(MODEL.links).map(n => [n, true]));
  const classVisible = {fastener:true, motor:true, cover:true, structure:true};
  let isolate = null, highlighted = null;
  const currentMat = m => STYLES[state.style].mat(m);
  function applyLook(){
    const look = STYLES[state.style].look; currentLook = look;
    const light = look === "light", bp = look === "blueprint";
    if (state.background === "none") { scene.background = null; renderer.setClearColor(0x000000, 0); }
    else if (state.background === "auto") { scene.background = new THREE.Color(LOOK_BG[look]); renderer.setClearColor(LOOK_BG[look], 1); }
    else { scene.background = new THREE.Color(state.background); renderer.setClearColor(new THREE.Color(state.background), 1); }
    const autoEdge = bp ? 0x9fd3ff : light ? 0x141a21 : 0xe3e9f2;
    edgeMat.color.set(state.edgeColor ? new THREE.Color(state.edgeColor) : autoEdge); edgeMat.opacity = bp ? 0.95 : light ? 0.95 : 0.8;
    grid.material.color.set(bp ? 0x1c4a86 : light ? 0xe1e5ea : 0x2e3540); grid.material.opacity = light ? 0.8 : 1;
    grid.visible = state.gridOn;
    hemi.intensity = light ? 1.35 : 0.85; key.intensity = light ? 1.25 : 1.7;
    axes.visible = !light && !bp && state.gridOn;
    makeLabels();
    if (opts.onLook) opts.onLook(look);
  }
  function applyMode(){
    for (const m of meshes) {
      const u = m.userData;
      if (m !== highlighted) m.material = api.materialOverride ? api.materialOverride(m) : currentMat(m);
      m.visible = linkVisible[u.link] && classVisible[u.pclass] && !u.hidden && (!isolate || isolate.has(m));
      u.edge.visible = state.edgesOn && m.visible;
      m.castShadow = state.shadows;
    }
    shadowPlane.visible = state.shadows;
    comGroup.visible = state.comOn;
  }
  const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);
  function applySection(){
    const s = state.section;
    if (s.axis === "off") { renderer.clippingPlanes = []; return null; }
    const n = new THREE.Vector3(s.axis === "x" ? 1 : 0, s.axis === "y" ? 1 : 0, s.axis === "z" ? 1 : 0).multiplyScalar(s.flip);
    const lo = box.min[s.axis] - 0.05, hi = box.max[s.axis] + 0.05, at = lo + (s.t + 1) / 2 * (hi - lo);
    plane.set(n, -at * s.flip); renderer.clippingPlanes = [plane];
    return at;
  }
  const api = {
    materialOverride: null, // optional (mesh) => material, used by applyMode when set
    THREE, MODEL, LEGS, JOINTS, BODIES, STYLES, renderer, scene, camera, robot, meshes, linkObj, jointObj, jointVal, linkRest, geoms, box, size, radius, centre, key, hemi, PO,
    materials: {cadMats, tintMats, legMats, whiteFlat, whiteShaded, hiddenMat, ghostMat, edgeMat, hiMat},
    state, view, linkVisible, classVisible, collObjs, axisObjs, comGroup, totalMass,
    get isolate(){ return isolate; }, set isolate(v){ isolate = v; },
    get highlighted(){ return highlighted; }, set highlighted(v){ highlighted = v; },
    currentMat, applyMode, applyLook, applyJoints, applyExplode, updateCom, updateLabels, updateCamera, fitDistance, frameVisible, applySection, textSprite,
    setStyle(name){ state.style = name; state.edgesOn = STYLES[name].edges; applyLook(); applyMode(); },
    setEdges(on){ state.edgesOn = on; applyMode(); },
    setFaceColor(hex){ state.faceColor = hex; whiteShaded.color.set(hex); whiteFlat.color.set(hex); },
    setEdgeColor(hex){ state.edgeColor = hex; applyLook(); },
    setBackground(v){ state.background = v; applyLook(); },
    setGrid(on){ state.gridOn = on; applyLook(); },
    setShadows(on){ state.shadows = on; applyMode(); },
    setStudio(on, intensity = 0.7){ scene.environment = on ? studioEnvironment() : null; for (const m of [whiteShaded, ...cadMats, ...Object.values(tintMats), ...Object.values(legMats)]) { m.envMapIntensity = intensity; m.needsUpdate = true; } hemi.intensity = on ? 0.5 : (currentLook !== "dark" ? 1.35 : 0.85); },
    setFov(f){ camera.fov = f; camera.updateProjectionMatrix(); },
    setLabels(on){ state.labelsOn = on; },
    setCom(on){ state.comOn = on; applyMode(); },
    setSection(axis, t, flip){ Object.assign(state.section, {axis, t: t ?? state.section.t, flip: flip ?? state.section.flip}); return applySection(); },
    setExplode(l, p){ state.explodeLinks = l; state.explodeParts = p; },
    setPose(map){ for (const n in jointVal) if (!MODEL.joints.find(j => j.n === n).mimic) jointVal[n] = map[n] ?? 0; },
    showAll(){ isolate = null; Object.keys(linkVisible).forEach(n => linkVisible[n] = true); Object.keys(classVisible).forEach(c => classVisible[c] = true); meshes.forEach(m => m.userData.hidden = false); applyMode(); },
    update(){ applyJoints(); applyExplode(); updateCom(); updateLabels(); updateCamera(); },
    render(){ renderer.render(scene, camera); },
    resize(w, h, css = true){ renderer.setSize(w, h, css); camera.aspect = w / h; camera.updateProjectionMatrix(); },
    capture(w, h, {transparent = false} = {}){
      const pr = renderer.getPixelRatio(); const sz = new THREE.Vector2(); renderer.getSize(sz);
      const bgSave = state.background;
      if (transparent) { state.background = "none"; applyLook(); }
      renderer.setPixelRatio(1); api.resize(w, h, false); api.update(); renderer.render(scene, camera);
      const url = renderer.domElement.toDataURL("image/png");
      if (transparent) { state.background = bgSave; applyLook(); }
      renderer.setPixelRatio(pr); api.resize(sz.x, sz.y, false);
      return url;
    },
    gait(t, base = MODEL.stance || {}, amp = {yaw:0.22, femur:0.40, tibia:0.25}, hz = 0.7){
      const w = t * 2 * Math.PI * hz, out = {}, A = new Set(["lf","lr","rm"]);
      for (const leg of LEGS) {
        const ph = A.has(leg) ? w : w + Math.PI, swing = Math.max(0, Math.sin(ph));
        out[`${leg}_coxa_yaw`] = amp.yaw * Math.cos(ph) * (leg[0] === "l" ? 1 : -1);
        out[`${leg}_femur_pitch`] = (base[`${leg}_femur_pitch`] ?? 0) + amp.femur * swing;
        out[`${leg}_tibia_pitch`] = (base[`${leg}_tibia_pitch`] ?? 0) + amp.tibia * swing;
      }
      return out;
    },
  };
  applyLook(); applyMode();
  return api;
}
