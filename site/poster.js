const $ = (s) => document.querySelector(s);
const esc = (x) => String(x ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const short = (x) => `${x.slice(0, 12)}…`;
const date = (x) => new Date(x).toLocaleString(undefined, {dateStyle:'medium',timeStyle:'short'});
const link = (url, label) => `<a class="text-link" href="${esc(url)}" target="_blank" rel="noopener">${esc(label)} ↗</a>`;
let data, selected = 'policy', paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
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
  return `<div class="detail-heading"><h3>${esc(n.title)}</h3>${link(n.source_url,'Open implementation')}</div><p>${esc(n.meaning)}</p><div class="io"><div><span>Input</span><p>${esc(n.input)}</p></div><div><span>Output</span><p>${esc(n.output)}</p></div></div><p class="next-note"><strong>Evidence today:</strong> ${esc(n.today)}</p><p class="next-note"><strong>Next proof:</strong> ${esc(n.next)}</p>`;
}
function geometryDiagram() {
  return `<svg viewBox="0 0 530 270" role="img" aria-label="Kinematic reference diagram: fixed coxa, 72.5 millimeter hip to knee, 126 millimeter knee to distal reference. Not a production linkage drawing."><defs><marker id="dim" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse"><path d="M0 0 L6 3 L0 6" fill="#777"/></marker></defs><path d="M45 95 L145 95 L285 65 L450 205" fill="none" stroke="#ddd" stroke-width="16" stroke-linecap="round"/><path d="M45 95 L145 95 L285 65 L450 205" fill="none" stroke="#222" stroke-width="3" stroke-linecap="round"/>${[[45,95],[145,95],[285,65],[450,205]].map(([x,y])=>`<circle cx="${x}" cy="${y}" r="8" fill="white" stroke="#222" stroke-width="3"/>`).join('')}<path d="M50 142 H140 M154 44 L275 20 M330 75 L472 196" fill="none" stroke="#777" marker-start="url(#dim)" marker-end="url(#dim)"/><text x="66" y="166">FIXED</text><text x="184" y="21">72.5 mm</text><text x="376" y="117" transform="rotate(40 376 117)">126 mm</text><text x="28" y="77">yaw</text><text x="130" y="122">hip</text><text x="265" y="106">knee</text><text x="332" y="244">distal reference</text></svg>`;
}

function majorStages() {
  return `<section aria-labelledby="checkpoint-heading"><h2 class="major-heading" id="checkpoint-heading">Major checkpoints</h2><div class="major-stages">${data.milestones.map((m,i)=>{
    const media=data.media.find(v=>v.id===m.media_id),poster=media&&data.media.find(v=>v.id===media.poster_id);
    return `<article class="major-stage"><div class="stage-label">Checkpoint ${i+1}</div><h2>${esc(m.title)}</h2><div class="stage-state">${esc(m.state)}</div><p>${esc(m.description)}</p>${media?`<figure class="stage-media" style="margin-left:0;margin-right:0"><video controls playsinline preload="none" ${poster?`poster="${esc(poster.url)}"`:''} aria-label="${esc(media.title)}"><source src="${esc(media.url)}" type="video/mp4"></video><figcaption>${esc(media.caption)}</figcaption></figure>`:''}<details><summary>Evidence and completion criteria</summary><p>${esc(m.gate)}</p><div class="stage-evidence">${link(m.source_url,'Evidence')}</div></details></article>`;
  }).join('')}</div></section>`;
}

function render() {
  const checkpoint=data.checkpoint_data, stop=data.stop_data;
  const videos=data.media.filter(m=>m.type==='video');
  const historical=videos.find(m=>m.id===data.primary_video_id);
  const benchmark=videos.find(m=>m.role==='benchmark');
  const supplementary=data.media.filter(m=>m.role==='research_image');
  $('#poster').innerHTML=`
  <section class="intro"><div><h1>${esc(data.mission)}</h1><p>${esc(data.description)}</p></div><div class="version">Published research snapshot<a href="${esc(data.version.commit_url)}">main / ${data.version.revision.slice(0,7)} ↗</a>${esc(date(data.version.commit_date))}<br>Rebuilt on every push to main</div></section>
  <div class="priority"><div><strong>${esc(data.focus.title)}</strong><p>${esc(data.focus.text)}</p></div><span class="badge">${esc(data.focus.badge)}</span></div>
  ${majorStages()}
  <div class="layout"><section id="system" class="panel"><div class="panel-head"><div><h2>System implementation</h2><p class="subtle">Select a block to see what exists and what must be proved.</p></div><button class="motion-toggle" id="motion-toggle" aria-pressed="${paused}">${paused?'Resume flow':'Pause flow'}</button></div><div class="graph-scroll"><div class="graph" id="graph">${graph()}</div></div><div class="detail" id="detail" aria-live="polite">${detail()}</div><p class="subtle" style="padding:0 24px 20px;font-size:.75rem">Animated lines explain information flow. They are not live sensor or GPU activity.</p></section>
  <aside class="rail"><article class="panel checkpoint"><div class="eyebrow">Latest published policy</div><h3>${esc(data.checkpoint.label)}</h3><div class="stat">${checkpoint.updates}<span style="font-size:.8rem;color:var(--muted);letter-spacing:0"> updates</span></div><p>${esc(data.checkpoint.qualification)}. Strict reload: ${checkpoint.reload_passed?'passed':'failed'}.</p><code title="${checkpoint.sha256}">${short(checkpoint.sha256)}</code><p><strong>${data.checkpoint.video_matches ? 'Matching recording available.' : 'No matching video yet.'}</strong> ${data.checkpoint.video_matches ? 'The selected clip records these exact weights.' : 'The omnidirectional clip shows the earlier policy.'}</p>${link(checkpoint.evidence_url,'Checkpoint evidence')}</article><article class="panel checkpoint failure"><div class="eyebrow">Quiet-stop evaluation</div><div class="stat">${stop.passing}<span style="font-size:1.2rem;color:var(--muted)"> / ${stop.replicas}</span></div><h3>Replicas passed quiet standing</h3><p>Measured over the final ten seconds after stopping. Drift, jitter and requested torque remain open failures.</p>${link(stop.evidence_url,'See every failed bound')}</article></aside></div>
  <section id="findings" class="section">${sectionTitle('01','Results','Software integration, walking quality and physical deployment are different milestones.')}<div class="findings">${data.findings.map(f=>`<article class="panel finding"><div class="label">${esc(f.label)}</div><h3>${esc(f.title)}</h3><p>${esc(f.text)}</p>${link(f.source_url,'Read the evidence')}</article>`).join('')}</div><div class="comparison-note">The accepted forward clip is a visual reference. It is not an all-direction, terrain or hardware qualification. Latest results are measured at the common 0.040 rad per 20 ms limiter.</div></section>
  <section id="robot" class="section">${sectionTitle('02','Robot geometry','Selected C geometry · 19 bodies / 18 joints. The physical four-bar assembly remains a separate model.')}<div class="robot-grid"><article class="panel robot-diagram"><h3>Kinematic lengths, not material cut lengths</h3>${geometryDiagram()}<p>This schematic shows reference relationships, not the production four-bar layout or a prescribed walking posture.</p></article><article class="panel dimensions"><table aria-label="Selected C leg dimensions"><tbody><tr><th>Coxa · joint-center distance</th><td>55.454 mm</td></tr><tr><th>Coxa · horizontal projection</th><td>47.718 mm</td></tr><tr><th>Femur · hip to knee</th><td>72.5 mm</td></tr><tr><th>Tibia · knee to distal reference</th><td>126 mm</td></tr><tr><th>Study tibia · full mesh extent</th><td>135.75 mm</td></tr></tbody></table><p class="subtle">${esc(data.geometry.note)}</p><div class="hardware-note"><div><strong>8.261 kg</strong><span>study assembly mass</span></div><div><strong>18</strong><span>RS05 motor coordinates</span></div><div><strong>1.6 N·m</strong><span>applied study cap</span></div></div></article></div></section>
  <section id="roadmap" class="section">${sectionTitle('03','Next experiments','A stage closes only when its evidence passes. Preparation does not count as completion.')}<div class="next-grid">${data.next.map((n,i)=>`<article class="panel next-item"><span>${String(i+1).padStart(2,'0')} →</span><div><h3>${esc(n.title)}</h3><p>${esc(n.text)}</p></div></article>`).join('')}</div></section>
  <section class="section">${sectionTitle('04','Figures','Actual recordings and study figures, with their scope kept explicit.')}<div class="findings">${supplementary.map(m=>mediaCard(m,'RESEARCH FIGURE')).join('')}</div></section>
  <section class="section">${sectionTitle('05','Related work',`${link(data.research_url,'Read the 32-work review')}`)}<div class="papers">${data.papers.map(p=>`<a class="panel paper" href="${esc(p.url)}" target="_blank" rel="noopener"><h3>${esc(p.title)}</h3><p>${esc(p.role)}</p><span>Primary source ↗</span></a>`).join('')}</div></section>
  <section id="notebook" class="section">${sectionTitle('06','Research record','Every repository change must explain its effect on this central poster framework.')}<div class="notebook-grid"><div class="panel updates">${data.updates.slice(0,8).map(u=>`<article class="update"><time datetime="${u.date}">${esc(date(u.date))}</time><h3>${esc(u.title)}</h3><p>${esc(u.summary)}</p>${u.no_project_impact?`<p>${esc(u.reason)}</p>`:''}<p class="next-action"><strong>Next:</strong> ${esc(u.next)}</p><div class="update-links">${u.evidence_links.slice(0,4).map(l=>link(l.url,l.label)).join('')}</div></article>`).join('')}</div><aside class="panel glossary"><h3>Technical words, plain English</h3>${data.glossary.map(g=>`<details><summary>${esc(g.term)}</summary><p>${esc(g.meaning)}</p></details>`).join('')}<details><summary>What does “live” mean here?</summary><p>This poster rebuilds from verified repository content on each push to main and checks for a new revision every minute. It is a published research snapshot, not a direct Spark telemetry feed.</p></details><details><summary>How do contributors keep it current?</summary><p>Every change needs a central update record and any relevant registry changes. CI checks evidence, media hashes and coverage of changed files.</p>${link(`https://github.com/${data.repository}/blob/main/docs/PROJECT_SITE.md`,'Required contributor contract')}</details></aside></div></section>`;
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
