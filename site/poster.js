const $ = (s) => document.querySelector(s);
const esc = (x) => String(x ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const short = (x) => `${x.slice(0, 12)}…`;
const date = (x) => new Date(x).toLocaleString(undefined, {dateStyle:'medium',timeStyle:'short'});
const link = (url, label) => `<a class="text-link" href="${esc(url)}" target="_blank" rel="noopener">${esc(label)} ↗</a>`;
let data, selected = 'policy', paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
const capabilityStates = {needs_definition:'Plan incomplete',ready:'Ready to start',active:'In progress',blocked:'Blocked',accepted:'Reference accepted'};
const states = {implemented:'Code exists',in_progress:'Tests incomplete',prepared:'Prototype',planned:'To build'};
const nodeIcon = {boundary:'polygon',coverage:'route',commands:'arrows',policy:'robot',sensors:'scan',estimate:'cross',terrain:'terrain',motors:'joint',simulation:'grid',evaluation:'check',hardware:'joint',survey:'map'};
function icon(kind) {
  const paths = {
    polygon:'M4 5L17 3L21 14L13 21L3 15Z', route:'M4 5H20V10H4V15H20V20H4',
    arrows:'M12 3V21M3 12H21M8 7L12 3L16 7M8 17L12 21L16 17M7 8L3 12L7 16M17 8L21 12L17 16',
    robot:'M8 7H16V17H8ZM3 4L8 8M16 8L21 4M2 12H8M16 12H22M3 20L8 16M16 16L21 20',
    scan:'M12 12L5 3M12 12L21 8M12 12L18 21M12 12L3 17M5 3L21 8L18 21L3 17Z',
    cross:'M12 2V7M12 17V22M2 12H7M17 12H22M7 7H17V17H7Z',
    terrain:'M2 17L7 7L12 14L17 4L22 17ZM3 21H21',joint:'M4 6L12 6L19 18M2 3H7V9H2ZM16 15H22V21H16Z',
    grid:'M3 3H9V9H3ZM15 3H21V9H15ZM3 15H9V21H3ZM15 15H21V21H15Z',
    check:'M4 4H20V20H4ZM7 12L11 16L18 8',map:'M3 5L9 3L15 6L21 3V19L15 22L9 19L3 21ZM9 3V19M15 6V22',
  };
  return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[kind]||paths.map}"/></svg>`;
}
function media(item, label='', caption=item?.caption) {
  if (!item) return '';
  const poster=data.media.find(m=>m.id===item.poster_id);
  const content=item.type==='video'
    ? `<video controls playsinline preload="none" ${poster?`poster="${esc(poster.url)}"`:''} aria-label="${esc(item.title)}"><source src="${esc(item.url)}" type="video/mp4"><a href="${esc(item.url)}">Open recording</a></video>`
    : `<a href="${esc(item.url)}" target="_blank" rel="noopener"><img src="${esc(item.url)}" alt="${esc(item.caption)}" loading="lazy"></a>`;
  return `<figure class="recorded-media">${label?`<div class="media-topline">${esc(label)}</div>`:''}${content}<figcaption>${esc(caption)} ${link(item.evidence_url,'Source')}</figcaption></figure>`;
}
function definitionCard(m) {
  const d=m.definition;
  if(!d)return '';
  return `<article class="definition-card" id="increment-${esc(d.id)}"><strong>${esc(d.id)}</strong><h3>${esc(d.title)}</h3><span class="scope">Owner needed</span>${link(data.architecture_url+'#m1-stationary-simulated-scan-export-and-reload','Test specification')}</article>`;
}
function geometryDiagram() {
  return `<svg viewBox="0 0 530 270" role="img" aria-label="Leg dimensions in the approved robot model: 49 millimeters from the yaw joint to the hip, 73.502 millimeters from hip to knee, and 130 millimeters from knee to the distal reference."><defs><marker id="dim" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse"><path d="M0 0 L6 3 L0 6" fill="#777"/></marker></defs><path d="M45 95 L145 95 L285 65 L450 205" fill="none" stroke="#ddd" stroke-width="16" stroke-linecap="round"/><path d="M45 95 L145 95 L285 65 L450 205" fill="none" stroke="#222" stroke-width="3" stroke-linecap="round"/>${[[45,95],[145,95],[285,65],[450,205]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="8" fill="white" stroke="#222" stroke-width="3"/>`).join('')}<path d="M50 142 H140 M154 44 L275 20 M330 75 L472 196" fill="none" stroke="#777" marker-start="url(#dim)" marker-end="url(#dim)"/><text x="66" y="166">${esc(data.geometry.coxa_joint_centers_mm)} mm</text><text x="184" y="21">${esc(data.geometry.femur_hip_to_knee_mm)} mm</text><text x="376" y="117" transform="rotate(40 376 117)">${esc(data.geometry.tibia_knee_to_distal_reference_mm)} mm</text><text x="28" y="77">yaw</text><text x="130" y="122">hip</text><text x="265" y="106">knee</text><text x="332" y="244">distal reference</text></svg>`;
}

function roadmap() {
  return `<section id="roadmap" class="section first-section" aria-labelledby="roadmap-heading"><div class="section-title"><h2 id="roadmap-heading">Roadmap</h2></div><ol class="major-stages">${data.milestones.map((m,i)=>{
    const item=data.media.find(v=>v.id===m.media_id);
    return `<li class="major-stage state-${esc(m.status)}" id="marker-${esc(m.id)}"><div class="stage-top"><span class="stage-number">${String(i+1).padStart(2,'0')}</span><span class="status-pill ${esc(m.status)}">${esc(capabilityStates[m.status])}</span></div><h3>${esc(m.title)}</h3>${item?media(item,'','Earlier robot model · real-time playback.'):''}<div class="stage-bottom"><p class="stage-status">${esc(m.card_text)}</p>${link(data.status_url,'Details')}</div></li>`;
  }).join('')}</ol>${data.milestones.map(definitionCard).join('')}</section>`;
}
function connections() {
  const adjacent=new Set([selected]);
  for(const [a,b] of data.edges)if(a===selected||b===selected){adjacent.add(a);adjacent.add(b);}
  return adjacent;
}
function graph() {
  const adjacent=connections();
  const nodes=data.nodes.map((n,i)=>`<button class="node ${n.id===selected?'selected':adjacent.has(n.id)?'connected':''} state-${esc(n.state)}" style="left:${22+n.column*300}px;top:${32+n.row*180}px" data-node="${esc(n.id)}" aria-pressed="${n.id===selected}" aria-controls="detail"><span class="node-head">${icon(nodeIcon[n.id])}<span class="node-state">${esc(states[n.state])}</span></span><strong>${esc(n.title)}</strong><span class="node-label">${esc(n.label)}</span><span class="node-number">${String(i+1).padStart(2,'0')}</span></button>`).join('');
  const paths=data.edges.map(([a,b])=>{
    const from=data.nodes.find(n=>n.id===a),to=data.nodes.find(n=>n.id===b);
    let x1=22+from.column*300+120,y1=32+from.row*180+70,x2=22+to.column*300+120,y2=32+to.row*180+70;
    let path;
    if(from.row===to.row){const direction=Math.sign(to.column-from.column);x1+=direction*120;x2-=direction*120;path=`M${x1} ${y1} H${x2}`;}
    else {const direction=Math.sign(to.row-from.row);y1+=direction*70;y2-=direction*70;const middle=(y1+y2)/2;path=`M${x1} ${y1} C${x1} ${middle},${x2} ${middle},${x2} ${y2}`;}
    const active=a===selected||b===selected;
    return `<path class="edge ${active?'active':''}" data-edge="${esc(a)}:${esc(b)}" d="${path}" marker-end="url(#arrow${active?'-active':''})"/>`;
  }).join('');
  return `<svg class="wires" viewBox="0 0 1184 570" aria-hidden="true"><defs>${['','-active'].map(s=>`<marker id="arrow${s}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L8 4L0 8" fill="${s?'#255a80':'#a9b6c0'}"/></marker>`).join('')}</defs>${paths}</svg>${nodes}`;
}
function detail() {
  const n=data.nodes.find(n=>n.id===selected),m=data.milestones.find(m=>m.id===n.milestone_id);
  const incoming=data.edges.filter(([,b])=>b===selected).map(([a])=>data.nodes.find(n=>n.id===a));
  const outgoing=data.edges.filter(([a])=>a===selected).map(([,b])=>data.nodes.find(n=>n.id===b));
  const buttons=items=>items.length?items.map(v=>`<button class="connection-link" data-node="${esc(v.id)}">${esc(v.title)} ↗</button>`).join(''):'<span class="subtle">No connection shown</span>';
  return `<div class="detail-heading">${icon(nodeIcon[n.id])}<div><span class="eyebrow">SELECTED COMPONENT · ${esc(states[n.state])}</span><h3>${esc(n.title)}</h3></div>${link(n.source_url,'Source')}</div><p>${esc(n.meaning)}</p><div class="io"><div><span>Receives</span><p>${esc(n.input)}</p>${buttons(incoming)}</div><div><span>Produces</span><p>${esc(n.output)}</p>${buttons(outgoing)}</div></div>${m?`<div class="node-proof"><a href="#marker-${esc(m.id)}">${esc(m.title)} · ${esc(capabilityStates[m.status])} ↗</a><p>${esc(m.blocker)}</p></div>`:''}${n.id==='sensors'?'<a class="text-link" href="#increment-M1">M1 · the agreed first mapping test ↑</a>':''}`;
}
function system() {
  return `<section id="system" class="section" aria-labelledby="system-heading"><div class="section-title"><div class="eyebrow">SYSTEM CONNECTIONS</div><h2 id="system-heading">How the parts connect</h2><p>Select a box to see what it receives, what it produces and where its code or design lives. “Code exists” means an implementation is present; test results are shown separately.</p></div><div class="system-board"><div class="graph-toolbar"><div class="legend"><span><i class="exists"></i>Code exists</span><span><i class="prototype"></i>Prototype / tests incomplete</span><span><i class="pending"></i>To build</span></div><button class="motion-toggle" id="motion-toggle" aria-pressed="${paused}">${paused?'Animate arrows':'Stop animation'}</button></div><div class="graph-scroll" tabindex="0" role="region" aria-label="System dependency graph; scroll horizontally on small screens"><div class="graph" id="graph">${graph()}</div></div><div class="detail" id="detail" aria-live="polite">${detail()}</div></div><p class="diagram-note">Arrows show how components connect. They do not show live robot activity or a schedule for assigning work.</p></section>`;
}
function findings() {
  return `<section id="findings" class="section"><div class="section-title"><div class="eyebrow">TEST RESULTS</div><h2>What has passed and what has failed</h2><p>Each result identifies the robot model used. Earlier-model results do not establish how the approved model behaves.</p></div><div class="findings">${data.progress.facts.map(f=>`<article class="finding"><div class="finding-top"><span class="status-pill ${esc(f.result)}">${esc(f.result)}</span><span class="scope">${esc(f.backend)}</span></div><h3>${esc(f.title)}</h3>${f.metric?`<div class="stat">${f.metric.value} / ${f.metric.total}<span> ${esc(f.metric.units)}</span></div><p class="metric-window">${esc(f.metric.window)}</p>`:''}<p>${esc(f.text)}</p>${link(f.source_url,'Test report')}</article>`).join('')}</div><p class="diagram-note">Results recorded: ${esc(date(data.progress.as_of))}. The publication date at the top is when this page was built.</p></section>`;
}
function render() {
  const cad=data.media.find(m=>m.role==='cad_preview');
  const shown=new Set([...data.milestones.map(m=>m.media_id),cad?.id]);
  const supplementary=data.media.filter(m=>!shown.has(m.id)&&(m.type==='video'||m.role==='research_image'));
  $('#poster').innerHTML=`<section class="intro"><div><div class="eyebrow">CORNELL PHYSICAL INTELLIGENCE · HEXAPOD MKII</div><h1>Hexapod development roadmap</h1><p>${esc(data.description)}</p><div class="mission-tags">${data.summary_tags.map(t=>`<span>${esc(t)}</span>`).join('')}</div></div><div class="version"><span class="pause-label">Research paused</span><p>Current hardware: robot not built</p>${link(data.architecture_url,'Architecture')}<a href="${esc(data.version.commit_url)}">Published ${esc(date(data.version.built_utc))} · ${esc(data.version.revision.slice(0,7))} ↗</a></div></section>${roadmap()}${system()}${findings()}<section id="robot" class="section"><div class="section-title"><div class="eyebrow">ROBOT MODEL</div><h2>The approved hexapod design</h2><p>19 simulated rigid bodies · 18 powered joints · physical robot not built</p></div><div class="robot-grid"><div>${media(cad,'CAD ANIMATION · APPROVED ROBOT MODEL')}</div><article class="robot-diagram"><h3>Leg dimensions and model mass</h3>${geometryDiagram()}<div class="hardware-note"><div><strong>${Number(data.geometry.mass_kg).toFixed(3)} kg</strong><span>total model mass, including corrected motor weights</span></div><div><strong>${esc(data.geometry.motors)}</strong><span>RS05 motors</span></div></div><p>${esc(data.geometry.note)}</p></article></div></section><section id="recordings" class="section"><div class="section-title"><div class="eyebrow">VIDEOS AND DESIGN STUDIES</div><h2>Earlier work</h2><p>Recordings and design proposals from earlier development.</p></div><div class="media-gallery">${supplementary.map(m=>media(m,m.type==='video'?'EARLIER SIMULATION':'DESIGN STUDY')).join('')}</div></section><section id="notebook" class="section"><div class="section-title"><h2>Project records</h2><p>${link(data.status_url,'Written status')} · ${link(data.research_url,'Research references')}</p></div><div class="notebook-grid"><details class="updates"><summary>Change history</summary>${data.updates.slice(0,8).map(u=>`<article class="update"><time datetime="${esc(u.date)}">${esc(date(u.date))}</time><h3>${esc(u.title)}</h3><div class="update-links">${u.evidence_links.slice(0,4).map(l=>link(l.url,l.label)).join('')}</div></article>`).join('')}</details><aside class="glossary"><h3>Terms used on this page</h3>${data.glossary.map(g=>`<details><summary>${esc(g.term)}</summary><p>${esc(g.meaning)}</p></details>`).join('')}</aside></div></section>`;
  document.body.classList.toggle('paused',paused);
  $('#motion-toggle').addEventListener('click',()=>{paused=!paused;document.body.classList.toggle('paused',paused);$('#motion-toggle').textContent=paused?'Animate arrows':'Stop animation';$('#motion-toggle').setAttribute('aria-pressed',String(paused));});
  const selectNode=e=>{const button=e.target.closest('[data-node]');if(!button)return;selected=button.dataset.node;$('#graph').innerHTML=graph();$('#detail').innerHTML=detail();$(`[data-node="${selected}"]`).focus({preventScroll:true});};
  $('#graph').addEventListener('click',selectNode);$('#detail').addEventListener('click',selectNode);
}
try {
  const response=await fetch('project-data.json',{cache:'no-store'});
  if(!response.ok)throw new Error(`Research data returned ${response.status}`);
  data=await response.json();render();
  setInterval(async()=>{try{const r=await fetch(`version.json?t=${Date.now()}`,{cache:'no-store'});if(!r.ok)return;const version=await r.json();if(version.revision!==data.version.revision||version.built_utc!==data.version.built_utc){if(![...document.querySelectorAll('video')].some(v=>!v.paused)){location.reload();return;}$('#update-button').hidden=false;}}catch{}},60000);
  $('#update-button').addEventListener('click',()=>location.reload());
} catch(error) {
  $('#poster').innerHTML=`<div class="loading"><h1>The project data could not load.</h1><p>${esc(error.message)}</p><p><a href="https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/main/STATUS.md">Read the current status on GitHub ↗</a></p></div>`;
}
