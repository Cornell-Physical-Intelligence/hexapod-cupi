const $ = (s) => document.querySelector(s);
const esc = (x) => String(x ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const day = (x) => new Date(`${x}T12:00:00Z`).toLocaleDateString('en-GB', {day:'numeric',month:'long',year:'numeric',timeZone:'UTC'});
const nodeStates = {working:'Working',in_progress:'In progress',not_started:'Not started'};
const stepStates = {done:'Done',now:'Now',next:'Next',later:'Later'};
const WIDE = {w:186, h:64, gx:224, gy:112, pad:16}, NARROW = {w:100, h:76, gx:120, gy:100, pad:6};
const narrow = matchMedia('(max-width:760px)');
let data, selected = 'controller';

function figure(item) {
  if (!item) return '';
  const poster = data.media.find((m) => m.id === item.poster_id);
  const content = item.type === 'video'
    ? `<video controls playsinline preload="none" ${poster ? `poster="${esc(poster.url)}"` : ''} aria-label="${esc(item.caption)}"><source src="${esc(item.url)}" type="video/mp4"><a href="${esc(item.url)}">Open the video</a></video>`
    : `<img src="${esc(item.url)}" alt="${esc(item.caption)}" loading="lazy">`;
  return `<figure>${content}<figcaption>${esc(item.caption)}</figcaption></figure>`;
}

function latest() {
  const l = data.latest, m = l.metric;
  return `<section id="latest" class="section" aria-labelledby="latest-heading"><h2 id="latest-heading">Latest progress</h2><div class="latest-head"><div><p class="date">${esc(day(l.date))}</p><h3>${esc(l.title)}</h3><p>${esc(l.text)}</p></div><div class="metric"><strong>${esc(m.value)}<span> / ${esc(m.total)}</span></strong><p>${esc(m.label)}</p></div></div><div class="media-row">${l.media_ids.map((id) => figure(data.media.find((x) => x.id === id))).join('')}</div><p class="latest-next"><strong>Next:</strong> ${esc(l.next)}</p></section>`;
}

function roadmap() {
  const steps = data.roadmap.map((s, i) => `<li class="step ${esc(s.state)}" id="step-${esc(s.id)}"><span class="dot" aria-hidden="true">${s.state === 'done' ? '✓' : i + 1}</span><span class="step-title">${esc(s.title)}</span><span class="step-state">${s.state === 'now' ? 'We are here' : esc(stepStates[s.state])}</span></li>`).join('');
  const done = data.accomplishments.map((a) => `<li><time datetime="${esc(a.date)}">${esc(day(a.date))}</time><p>${esc(a.text)}</p></li>`).join('');
  return `<section id="roadmap" class="section" aria-labelledby="roadmap-heading"><h2 id="roadmap-heading">Roadmap</h2><ol class="track">${steps}</ol><h3>Done so far</h3><ol class="timeline">${done}</ol></section>`;
}

// On a phone the diagram turns on its side: columns become rows, so it fits the screen width.
const layout = () => narrow.matches ? NARROW : WIDE;
const place = (n) => narrow.matches ? {c: n.row, r: n.column} : {c: n.column, r: n.row};
const box = (n) => { const l = layout(), p = place(n); return {x: l.pad + p.c * l.gx, y: l.pad + p.r * l.gy}; };
function adjacent() {
  const near = new Set([selected]);
  for (const [a, b] of data.edges) if (a === selected || b === selected) { near.add(a); near.add(b); }
  return near;
}
function graph() {
  const near = adjacent(), l = layout(), W = l.w, H = l.h;
  const width = l.pad * 2 + Math.max(...data.nodes.map((n) => place(n).c)) * l.gx + W;
  const height = l.pad * 2 + Math.max(...data.nodes.map((n) => place(n).r)) * l.gy + H;
  const paths = data.edges.map(([a, b]) => {
    const from = box(data.nodes.find((n) => n.id === a)), to = box(data.nodes.find((n) => n.id === b));
    let path;
    if (from.y === to.y) {
      const direction = Math.sign(to.x - from.x), y = from.y + H / 2;
      path = `M${from.x + W / 2 + direction * W / 2} ${y} H${to.x + W / 2 - direction * (W / 2 + 4)}`;
    } else {
      const direction = Math.sign(to.y - from.y);
      const x1 = from.x + W / 2, x2 = to.x + W / 2;
      const y1 = from.y + H / 2 + direction * H / 2, y2 = to.y + H / 2 - direction * (H / 2 + 4), middle = (y1 + y2) / 2;
      path = `M${x1} ${y1} C${x1} ${middle},${x2} ${middle},${x2} ${y2}`;
    }
    const active = a === selected || b === selected;
    return `<path class="edge ${active ? 'active' : ''}" data-edge="${esc(a)}:${esc(b)}" d="${path}" marker-end="url(#arrow${active ? '-active' : ''})"/>`;
  }).join('');
  const markers = ['', '-active'].map((s) => `<marker id="arrow${s}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8"/></marker>`).join('');
  const buttons = data.nodes.map((n) => {
    const {x, y} = box(n);
    const role = n.id === selected ? 'selected' : near.has(n.id) ? 'connected' : '';
    return `<button class="node ${esc(n.state)} ${role}" style="left:${x}px;top:${y}px;width:${W}px;height:${H}px" data-node="${esc(n.id)}" aria-pressed="${n.id === selected}" aria-controls="detail" aria-label="${esc(n.title)}, ${esc(nodeStates[n.state])}">${esc(n.title)}</button>`;
  }).join('');
  return `<div class="graph-canvas" style="width:${width}px;height:${height}px"><svg class="wires" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" aria-hidden="true"><defs>${markers}</defs>${paths}</svg>${buttons}</div>`;
}
function detail() {
  const n = data.nodes.find((x) => x.id === selected);
  const names = (ids) => ids.map((id) => esc(data.nodes.find((x) => x.id === id).title)).join(', ') || 'None';
  const from = data.edges.filter(([, b]) => b === selected).map(([a]) => a);
  const to = data.edges.filter(([a]) => a === selected).map(([, b]) => b);
  return `<h3>${esc(n.title)} <span class="badge ${esc(n.state)}">${esc(nodeStates[n.state])}</span></h3><p>${esc(n.text)} <a href="${esc(n.source_url)}" target="_blank" rel="noopener">Read more ↗</a></p><p><strong>Depends on:</strong> ${names(from)}. <strong>Needed by:</strong> ${names(to)}.</p>`;
}
function system() {
  const legend = Object.entries(nodeStates).map(([state, label]) => `<span class="badge ${state}">${label}</span>`).join('');
  return `<section id="system" class="section" aria-labelledby="system-heading"><h2 id="system-heading">System connections</h2><p>Each box is one step toward a finished survey. An arrow points to the step that depends on it. Select a box to read more.</p><div class="legend">${legend}</div><div class="graph-scroll" tabindex="0" role="region" aria-label="System diagram"><div class="graph" id="graph">${graph()}</div></div><div class="detail" id="detail" aria-live="polite">${detail()}</div></section>`;
}

function leg(r) {
  return `<svg viewBox="0 0 530 270" role="img" aria-label="One leg: ${esc(r.coxa_mm)} mm from the swing joint to the hip, ${esc(r.femur_mm)} mm from hip to knee and ${esc(r.tibia_mm)} mm from knee to foot."><defs><marker id="dim" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto-start-reverse"><path d="M0 0 L6 3 L0 6"/></marker></defs><path class="limb-shadow" d="M45 95 L145 95 L285 65 L450 205"/><path class="limb" d="M45 95 L145 95 L285 65 L450 205"/>${[[45,95],[145,95],[285,65],[450,205]].map(([x,y]) => `<circle cx="${x}" cy="${y}" r="8"/>`).join('')}${['M50 142 H140', 'M154 44 L275 20', 'M330 75 L472 196'].map((d) => `<path class="dimension" d="${d}" marker-start="url(#dim)" marker-end="url(#dim)"/>`).join('')}<text x="66" y="166">${esc(r.coxa_mm)} mm</text><text x="160" y="16">${esc(Number(r.femur_mm).toFixed(1))} mm</text><text x="388" y="104" transform="rotate(40 388 104)">${esc(r.tibia_mm)} mm</text><text x="18" y="77">swing</text><text x="130" y="122">hip</text><text x="265" y="106">knee</text><text x="440" y="244">foot</text></svg>`;
}
function robot() {
  const r = data.robot, cad = data.media.find((m) => m.id === r.media_id);
  return `<section id="robot" class="section" aria-labelledby="robot-heading"><h2 id="robot-heading">The robot</h2><p>${esc(r.caption)}</p><div class="robot-grid">${figure(cad)}<figure class="leg">${leg(r)}<figcaption>One of the six legs, with its three joints.</figcaption></figure></div></section>`;
}

function terms() {
  return `<section id="terms" class="section" aria-labelledby="terms-heading"><h2 id="terms-heading">Terms used on this page</h2><dl class="terms">${data.glossary.map((g) => `<div><dt>${esc(g.term)}</dt><dd>${esc(g.meaning)}</dd></div>`).join('')}</dl></section>`;
}

function render() {
  $('#poster').innerHTML = `<section class="intro"><h1>${esc(data.title)}</h1><p>${esc(data.mission)}</p></section>${latest()}${roadmap()}${system()}${robot()}${terms()}`;
  $('#graph').addEventListener('click', (e) => {
    const button = e.target.closest('[data-node]');
    if (!button) return;
    selected = button.dataset.node;
    $('#graph').innerHTML = graph();
    $('#detail').innerHTML = detail();
    $(`[data-node="${selected}"]`).focus({preventScroll: true});
  });
  narrow.addEventListener('change', () => { $('#graph').innerHTML = graph(); });
}

try {
  const response = await fetch('project-data.json', {cache: 'no-store'});
  if (!response.ok) throw new Error(`Project data returned ${response.status}`);
  data = await response.json();
  render();
  setInterval(async () => {
    try {
      const r = await fetch(`version.json?t=${Date.now()}`, {cache: 'no-store'});
      if (!r.ok) return;
      const version = await r.json();
      if (version.revision !== data.version.revision || version.built_utc !== data.version.built_utc) {
        if (![...document.querySelectorAll('video')].some((v) => !v.paused)) { location.reload(); return; }
        $('#update-button').hidden = false;
      }
    } catch {}
  }, 60000);
  $('#update-button').addEventListener('click', () => location.reload());
} catch (error) {
  $('#poster').innerHTML = `<div class="loading"><h1>The page could not load.</h1><p>${esc(error.message)}</p><p><a href="https://github.com/Cornell-Physical-Intelligence/hexapod-cupi">Open the repository on GitHub ↗</a></p></div>`;
}
