const $ = (s) => document.querySelector(s);
const esc = (x) => String(x ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const short = (x) => `${x.slice(0, 12)}…`;
const date = (x) => new Date(x).toLocaleString(undefined, {dateStyle:'medium',timeStyle:'short'});
const link = (url, label) => `<a class="text-link" href="${esc(url)}" target="_blank" rel="noopener">${esc(label)} ↗</a>`;
let data, selected = 'policy', paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
const capabilityStates = {needs_definition:'Needs definition', ready:'Ready', active:'Active', blocked:'Blocked', accepted:'Accepted within stated scope'};
const states = {implemented:'Implemented',in_progress:'In progress',prepared:'Prepared',planned:'Not qualified yet'};

function mediaCard(item, top = 'ACTUAL SIMULATION RECORDING') {
  if (!item) return `<article class="panel checkpoint"><h3>No matching recording yet</h3><p>The latest saved weights have not been recorded. Older videos are identified separately below.</p></article>`;
  const poster = data.media.find(m => m.id === item.poster_id);
  const body = item.type === 'video'
    ? `<video controls playsinline preload="none" ${poster ? `poster="${esc(poster.url)}"` : ''} aria-label="${esc(item.title)}"><source src="${esc(item.url)}" type="video/mp4">Your browser cannot play this video. <a href="${esc(item.url)}">Open the recording</a>.</video>`
    : `<a href="${esc(item.url)}"><img src="${esc(item.url)}" alt="${esc(item.caption)}" loading="lazy"></a>`;
  return `<article class="panel media-panel"><div class="media-topline">${esc(top)}</div>${body}<div class="media-caption"><h3>${esc(item.title)}</h3><p>${esc(item.caption)}</p>${link(item.evidence_url,'Read its evidence')}</div></article>`;
}
function sectionTitle(number, title, description='') {
  return `<div class="section-title"><h2><span class="number">${number}</span>${title}</h2>${description ? `<p>${description}</p>`:''}</div>`;
}
function graph() {
  const nodes = data.nodes.map((n, i) => `<button class="node ${n.id === selected ? 'selected':''}" style="left:${2+n.column*25}%;top:${27+n.row*140}px" data-node="${n.id}" aria-pressed="${n.id === selected}"><span class="node-number">${String(i+1).padStart(2,'0')}<span class="node-state">${esc(states[n.state])}</span></span><strong>${esc(n.title)}</strong><span class="node-label">${esc(n.label)}</span></button>`).join('');
  const paths = data.edges.map(([a,b]) => {
    const from=data.nodes.find(n=>n.id===a),to=data.nodes.find(n=>n.id===b);
    let x1=from.column*250+125,y1=from.row*140+73,x2=to.column*250+125,y2=to.row*140+73;
    let path;
    if (from.row===to.row) {x1+=105; x2-=105; path=`M ${x1} ${y1} H ${x2}`;}
    else {y1+=to.row>from.row?47:-47;y2+=to.row>from.row?-47:47;const mid=(y1+y2)/2;path=`M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`;}
    return `<path class="edge ${a===selected||b===selected?'active':''} ${to.row<from.row?'feedback':''}" d="${path}" vector-effect="non-scaling-stroke" marker-end="url(#arrow)"/>`;
  }).join('');
  return `<svg viewBox="0 0 1000 445" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#777"/></marker></defs>${paths}</svg>${nodes}`;
}
function detail() {
  const n=data.nodes.find(n=>n.id===selected);
  return `<div class="detail-heading"><h3>${esc(n.title)}</h3>${link(n.source_url,'Open implementation')}</div><p>${esc(n.meaning)}</p><div class="io"><div><span>Input</span><p>${esc(n.input)}</p></div><div><span>Output</span><p>${esc(n.output)}</p></div></div><p class="next-note">This block describes a software boundary. Qualification and current evidence are recorded in the roadmap above.</p>`;
}
function geometryDiagram() {
  return `<svg viewBox="0 0 530 270" role="img" aria-label="Kinematic reference diagram: 49 millimeter yaw to hip, 73.502 millimeter hip to knee, 130 millimeter knee to distal surface. Schematic of the canonical direct-drive CAD."><defs><marker id="dim" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse"><path d="M0 0 L6 3 L0 6" fill="#777"/></marker></defs><path d="M45 95 L145 95 L285 65 L450 205" fill="none" stroke="#ddd" stroke-width="16" stroke-linecap="round"/><path d="M45 95 L145 95 L285 65 L450 205" fill="none" stroke="#222" stroke-width="3" stroke-linecap="round"/>${[[45,95],[145,95],[285,65],[450,205]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="8" fill="white" stroke="#222" stroke-width="3"/>`).join('')}<path d="M50 142 H140 M154 44 L275 20 M330 75 L472 196" fill="none" stroke="#777" marker-start="url(#dim)" marker-end="url(#dim)"/><text x="66" y="166">${esc(data.geometry.coxa_joint_centers_mm)} mm</text><text x="184" y="21">${esc(data.geometry.femur_hip_to_knee_mm)} mm</text><text x="376" y="117" transform="rotate(40 376 117)">${esc(data.geometry.tibia_knee_to_distal_reference_mm)} mm</text><text x="28" y="77">yaw</text><text x="130" y="122">hip</text><text x="265" y="106">knee</text><text x="332" y="244">distal reference</text></svg>`;
}

function roadmap() {
  return `<section id="roadmap" class="section" aria-labelledby="roadmap-heading">
    <div class="section-title"><h2 id="roadmap-heading">Roadmap against the architecture</h2>
    <p>${esc(data.progress.definition_policy)}</p></div>
    <ol class="roadmap-list">${data.milestones.map((m,i)=>{
      const deps=m.depends_on.map(id=>data.milestones.find(x=>x.id===id));
      const step=m.next_step;
      return `<li class="roadmap-marker" id="marker-${esc(m.id)}">
        <div class="marker-top"><span class="stage-label">${String(i+1).padStart(2,'0')}</span><span class="stage-state">${esc(capabilityStates[m.status])}</span></div>
        <h3>${esc(m.title)}</h3><p>${esc(m.description)}</p>
        <p class="marker-scope">${esc(m.backend)}</p>
        <p><strong>Current boundary:</strong> ${esc(m.blocker)}</p>
        <p><strong>Next step:</strong> ${step?esc(step.outcome):'Needs definition with the team.'}</p>
        <details><summary>Evidence, ownership and open definition</summary>
          <p><strong>Required proof:</strong> ${esc(m.gate)}</p>
          ${m.acceptance?`<p><strong>Accepted by ${esc(m.acceptance.by)}:</strong> ${esc(m.acceptance.scope)}</p>`:''}
          <p><strong>Owner:</strong> ${esc(m.owner)}</p>
          <p><strong>Depends on:</strong> ${deps.length?deps.map(d=>`<a href="#marker-${esc(d.id)}">${esc(d.title)}</a>`).join(', '):'Historical reference; no prior marker.'}</p>
          <p><strong>Question to define:</strong> ${esc(m.question)}</p>
          ${step?`<p>${link(step.issue,'Assigned issue')} · Reviewer: ${esc(step.reviewer)}</p><p>Acceptance: ${esc(step.acceptance)}</p>`:'<p>No implementation packet has been assigned by this roadmap.</p>'}
          <p>${link(m.source_url,'Recorded evidence')} · ${link(data.architecture_url,`Architecture: ${m.requirements.join(', ')}`)}</p>
        </details>
      </li>`;
    }).join('')}</ol></section>`;
}

function render() {
  const progress=data.progress;
  const evidence=data.media.filter(m=>m.role==='benchmark'||m.id===data.primary_video_id||m.role==='cad_preview');
  $('#poster').innerHTML=`
    <section class="intro"><div><h1>${esc(data.mission)}</h1><p>${link(data.architecture_url,'Read the architecture')} · Requirements, boundaries and team workflow</p></div>
      <div class="version">Published snapshot<a href="${esc(data.version.commit_url)}">${data.version.revision.slice(0,7)} ↗</a>${esc(date(data.version.built_utc))}</div></section>
    <section class="current-evidence" aria-labelledby="current-heading"><h2 id="current-heading">Where the evidence stops</h2><p>${esc(progress.summary)}</p><p class="subtle">Evidence recorded ${esc(date(progress.as_of))}. Repository cleanup does not establish new robot capability.</p></section>
    ${roadmap()}
    <section id="findings" class="section">${sectionTitle('','Measured results','Each attempt retains its model, scope and original result.')}
      <div class="findings">${progress.facts.map(f=>`<article class="panel finding"><div class="label">${esc(f.result)} · ${esc(f.backend)}</div><h3>${esc(f.title)}</h3>${f.metric?`<div class="stat">${f.metric.value} / ${f.metric.total}<span class="metric-unit"> ${esc(f.metric.units)}</span></div><p class="subtle">${esc(f.metric.window)}</p>`:''}<p>${esc(f.text)}</p>${link(f.source_url,'Read the evidence')}</article>`).join('')}</div></section>
    <section id="system" class="section"><details class="system-details"><summary>System boundaries and existing source</summary><p>Source implementation and demonstrated robot capability are recorded separately.</p>
      <div class="panel-head"><h2>System connections</h2><button class="motion-toggle" id="motion-toggle" aria-pressed="${paused}">${paused?'Resume flow':'Pause flow'}</button></div>
      <div class="graph-scroll"><div class="graph" id="graph">${graph()}</div></div><div class="detail" id="detail" aria-live="polite">${detail()}</div>
      <p class="subtle">The flow describes architecture. It is not live telemetry.</p></details></section>
    <section id="robot" class="section"><details><summary>Approved robot and recorded media</summary><p>${esc(data.geometry.note)}</p>
      <div class="robot-grid"><article class="panel robot-diagram"><h3>Direct-drive model · 19 bodies / 18 joints</h3>${geometryDiagram()}<p>${Number(data.geometry.mass_kg).toFixed(3)} kg with nominal motor mass overrides. Lengths are kinematic references.</p></article></div>
      <div class="findings">${evidence.map(m=>mediaCard(m,m.role==='benchmark'?'HISTORICAL FORWARD BENCHMARK':m.id===data.primary_video_id?'HISTORICAL POLICY RECORDING':'KINEMATIC CAD PREVIEW')).join('')}</div>
      <details><summary>All preserved figures and recordings</summary><div class="findings">${data.media.filter(m=>!evidence.includes(m)&&['video','research_image'].includes(m.type==='video'?'video':m.role)).map(m=>mediaCard(m,'RECORDED EVIDENCE')).join('')}</div></details></details></section>
    <section id="notebook" class="section"><details><summary>Change history and research references</summary><div class="updates">${data.updates.slice(0,8).map(u=>`<article class="update"><time datetime="${esc(u.date)}">${esc(date(u.date))}</time><h3>${esc(u.title)}</h3><p>${esc(u.summary)}</p><div class="update-links">${u.evidence_links.slice(0,4).map(l=>link(l.url,l.label)).join('')}</div></article>`).join('')}</div>
      <p>${link(data.research_url,'Research literature and prior experiments')}</p><p>${link(data.status_url,'Generated text progress')}</p></details></section>`;
  document.body.classList.toggle('paused',paused);
  $('#motion-toggle').addEventListener('click',()=>{paused=!paused;document.body.classList.toggle('paused',paused);$('#motion-toggle').textContent=paused?'Resume flow':'Pause flow';$('#motion-toggle').setAttribute('aria-pressed',String(paused));});
  $('#graph').addEventListener('click',e=>{const button=e.target.closest('[data-node]');if(!button)return;selected=button.dataset.node;$('#graph').innerHTML=graph();$('#detail').innerHTML=detail();$(`[data-node="${selected}"]`).focus({preventScroll:true});});
}
try {
  const response=await fetch('project-data.json',{cache:'no-store'});
  if(!response.ok)throw new Error(`Research data returned ${response.status}`);
  data=await response.json();render();
  setInterval(async()=>{
    try{const r=await fetch(`version.json?t=${Date.now()}`,{cache:'no-store'});if(!r.ok)return;const version=await r.json();if(version.revision!==data.version.revision || version.built_utc!==data.version.built_utc){if(![...document.querySelectorAll('video')].some(v=>!v.paused)){location.reload();return;}$('#update-button').hidden=false;}}catch{}
  },60000);
  $('#update-button').addEventListener('click',()=>location.reload());
} catch(error) {
  $('#poster').innerHTML=`<div class="loading"><h1>The research snapshot could not load.</h1><p>${esc(error.message)}</p><p><a href="https://github.com/Cornell-Physical-Intelligence/hexapod-cupi/blob/main/STATUS.md">Read the current status on GitHub ↗</a></p></div>`;
}
