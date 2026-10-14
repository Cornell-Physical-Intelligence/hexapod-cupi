// Exercise the built page and its load-error state without a browser or new dependency.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const code = await readFile(new URL('./poster.js', import.meta.url), 'utf8');
const data = JSON.parse(await readFile(new URL('./dist/project-data.json', import.meta.url), 'utf8'));
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
async function render(payload, failure=false, phone=false) {
  const elements = new Map();
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, {
        innerHTML:'', events:{}, attributes:{}, hidden:true,
        addEventListener(name, callback){this.events[name]=callback;},
        setAttribute(name, value){this.attributes[name]=value;}, focus(){},
      });
      return elements.get(selector);
    },
    querySelectorAll() { return []; },
  };
  const matchMedia = () => ({matches:phone, addEventListener(){}});
  await new AsyncFunction('document','matchMedia','fetch','setInterval',code)(
    document, matchMedia, async () => ({ok:!failure, status:failure?503:200, json:async()=>payload}), () => {});
  return {html:elements.get('#poster').innerHTML, elements};
}
const {html, elements} = await render(data);
assert.ok(!html.includes('could not load'), 'Valid built registry must render');

// The page holds five sections and nothing that the presentation standard removed.
for (const id of ['latest', 'roadmap', 'system', 'robot', 'terms']) assert.ok(html.includes(`id="${id}"`), `Missing section ${id}`);
for (const removed of ['id="findings"', 'id="recordings"', 'id="notebook"', 'Change history', 'Earlier work',
  'class="eyebrow"', 'diagram-note', 'class="subtle"', 'status-pill', 'major-stage', '<small', '<details']) {
  assert.ok(!html.includes(removed), `Removed content returned: ${removed}`);
}
const words = html.replace(/<svg[\s\S]*?<\/svg>/g, ' ').replace(/<[^>]+>/g, ' ').split(/\s+/).filter(Boolean).length;
assert.ok(words <= 650, `The page shows ${words} words; keep it short (docs/PROJECT_SITE.md)`);

// Latest progress: number, text and every video with its caption, all expanded.
const latest = data.latest;
assert.ok(html.includes(`${latest.metric.value}<span> / ${latest.metric.total}</span>`));
assert.ok(html.includes(latest.title) && html.includes(latest.next));
for (const id of latest.media_ids) {
  const item = data.media.find((m) => m.id === id);
  assert.ok(html.includes(`src="${item.url}"`), `Latest video ${id} must render`);
  assert.ok(html.includes(`<figcaption>${item.caption}</figcaption>`), `Latest video ${id} needs its caption`);
}

// Roadmap: each step, one current step, each accomplishment.
for (const step of data.roadmap) assert.ok(html.includes(`id="step-${step.id}"`) && html.includes(step.title));
assert.equal((html.match(/We are here/g) || []).length, 1, 'Exactly one current step');
for (const item of data.accomplishments) assert.ok(html.includes(item.text));

// System diagram: each box selects, shows its sentence, source and connections in text, and lights its arrows.
const title = (id) => data.nodes.find((n) => n.id === id).title;
for (const node of data.nodes) {
  elements.get('#graph').events.click({target:{closest:()=>({dataset:{node:node.id}})}});
  const panel = elements.get('#detail').innerHTML;
  assert.ok(panel.includes(node.text) && panel.includes(node.source_url));
  for (const [a,b] of data.edges) {
    if (b === node.id) assert.ok(panel.split('Needed by:')[0].includes(title(a)), `${node.id} must name its input ${a}`);
    if (a === node.id) assert.ok(panel.split('Needed by:')[1].includes(title(b)), `${node.id} must name its output ${b}`);
  }
  assert.ok(elements.get('#graph').innerHTML.includes(`data-node="${node.id}" aria-pressed="true"`));
  for (const [a,b] of data.edges.filter(([a,b])=>a===node.id||b===node.id)) {
    assert.ok(elements.get('#graph').innerHTML.includes(`class="edge active" data-edge="${a}:${b}"`));
  }
}

// On a phone the diagram turns on its side and fits a 390-pixel screen without scrolling.
const {html:phone} = await render(data, false, true);
const canvas = phone.match(/graph-canvas" style="width:(\d+)px/);
assert.ok(canvas && Number(canvas[1]) <= 358, `Phone diagram is ${canvas?.[1]} px wide`);
for (const node of data.nodes) assert.ok(phone.includes(`data-node="${node.id}"`));

// Robot and terms.
const cad = data.media.find((m) => m.id === data.robot.media_id);
assert.ok(html.includes(`src="${cad.url}"`) && html.includes(data.robot.caption));
for (const term of data.glossary) assert.ok(html.includes(`<dt>${term.term}</dt>`));

assert.ok(!/(?:href|src|poster)="(?:undefined|null)"|>undefined</.test(html), 'No missing field may become a link or displayed value');
const unsafe = structuredClone(data);
unsafe.latest.title = '<script>bad()</script>';
const {html:escaped} = await render(unsafe);
assert.ok(escaped.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!escaped.includes('<script>bad()</script>'));
assert.ok((await render(data,true)).html.includes('could not load'));
console.log(`Five sections, ${words} words, videos, roadmap, system diagram, escaping and load-error state verified`);
