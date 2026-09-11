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
      if (!elements.has(selector)) elements.set(selector, {innerHTML:'', addEventListener(){}, setAttribute(){}, focus(){}});
      return elements.get(selector);
    },
    body: {classList:{toggle(){}}},
  };
  await new AsyncFunction('document','matchMedia','fetch','setInterval',code)(
    document, () => ({matches:true}),
    async () => ({ok:!failure, status:failure?503:200, json:async()=>payload}), () => {});
  return elements.get('#poster').innerHTML;
}
const html = await render(data);
assert.ok(!html.includes('could not load'), 'Valid built registry must render');
for (const marker of data.milestones) {
  assert.ok(html.includes(`id="marker-${marker.id}"`));
  assert.ok(html.includes(marker.title));
}
assert.ok(html.includes('Needs definition with the team.'));
assert.ok(html.includes('10 / 32'));
assert.ok(html.includes('Historical forward visual benchmark only'));
assert.ok(!/(?:href|src)="undefined"|>undefined</.test(html), 'No missing field may become a link or displayed value');
const unsafe = structuredClone(data);
unsafe.milestones[1].question = '<script>bad()</script>';
const escaped = await render(unsafe);
assert.ok(escaped.includes('&lt;script&gt;bad()&lt;/script&gt;'));
assert.ok(!escaped.includes('<script>bad()</script>'));
assert.ok((await render(data,true)).includes('could not load'));
console.log('Built roadmap, evidence, unresolved definitions, escaping and load-error state verified');
