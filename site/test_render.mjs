// Exercise the built page and its load-error state without a browser or new dependency.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const code = await readFile(new URL('./poster.js', import.meta.url), 'utf8');
const data = JSON.parse(await readFile(new URL('./dist/project-data.json', import.meta.url), 'utf8'));
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
async function render(payload, failure=false) {
  const elements = new Map();
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, {
        innerHTML:'', events:{}, attributes:{},
        addEventListener(name, callback){this.events[name]=callback;},
        setAttribute(name, value){this.attributes[name]=value;}, focus(){},
      });
      return elements.get(selector);
    },
    body: {classList:{toggle(){}}},
  };
  await new AsyncFunction('document','matchMedia','fetch','setInterval',code)(
    document, () => ({matches:true}),
    async () => ({ok:!failure, status:failure?503:200, json:async()=>payload}), () => {});
  return {html:elements.get('#poster').innerHTML, elements};
}
const {html, elements} = await render(data);
assert.ok(!html.includes('could not load'), 'Valid built registry must render');
for (const marker of data.milestones) {
  assert.ok(html.includes(`id="marker-${marker.id}"`));
  assert.ok(html.includes(marker.title));
}
const roadmap = html.slice(html.indexOf('<ol class="major-stages">'), html.indexOf('</ol>'));
for (const marker of data.milestones) {
  const card = roadmap.split(`id="marker-${marker.id}"`)[1].split('</li>')[0];
  assert.equal((card.match(/<p\b/g)||[]).length,1,'Each stage has one summary line');
  assert.ok(!card.includes('<details'), 'Stage details belong in the linked document');
  if (!marker.media_id) assert.ok(!/<(?:video|img|svg)\b/.test(card),'Unrecorded stages have no visual');
}
assert.ok(!html.includes('mission-figure'));
assert.ok(!html.includes('exact_mount_comparison.png'));
for (const fact of data.progress.facts) {
  if (fact.metric) assert.ok(html.includes(`${fact.metric.value} / ${fact.metric.total}`));
}
assert.ok(!/(?:href|src)="undefined"|>undefined</.test(html), 'No missing field may become a link or displayed value');
// The main visuals must be visible on arrival, independent of optional prose.
const tags = html.match(/<\/?(?:details|video|div)\b[^>]*>/g);
let collapsed = 0;
for (const tag of tags) {
  if (tag.startsWith('<details')) collapsed++;
  if (tag.startsWith('</details')) collapsed--;
  if (tag.startsWith('<video') || /id="graph"/.test(tag)) assert.equal(collapsed,0,'Visuals must remain expanded');
}
for (const item of data.media.filter(m=>m.type==='video')) assert.ok(html.includes(`src="${item.url}"`));
assert.ok(html.includes('id="increment-M1"'));
for (const node of data.nodes) {
  elements.get('#graph').events.click({target:{closest:()=>({dataset:{node:node.id}})}});
  assert.ok(elements.get('#detail').innerHTML.includes(node.source_url));
  assert.ok(elements.get('#graph').innerHTML.includes(`data-node="${node.id}" aria-pressed="true"`));
  for (const [a,b] of data.edges.filter(([a,b])=>a===node.id||b===node.id)) {
    assert.ok(elements.get('#graph').innerHTML.includes(`class="edge active" data-edge="${a}:${b}"`));
  }
}
elements.get('#detail').events.click({target:{closest:()=>({dataset:{node:'sensors'}})}});
assert.ok(elements.get('#detail').innerHTML.includes('#increment-M1'));
elements.get('#motion-toggle').events.click();
assert.equal(elements.get('#motion-toggle').attributes['aria-pressed'],'false');
const unsafe = structuredClone(data);
unsafe.milestones[1].card_text = '<script>bad()</script>';
const {html:escaped} = await render(unsafe);
assert.ok(escaped.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!escaped.includes('<script>bad()</script>'));
assert.ok((await render(data,true)).html.includes('could not load'));
console.log('Expanded videos, interactive dependency graph, M1, evidence, escaping and load-error state verified');
