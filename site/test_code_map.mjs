// Test generated map interactions without a browser dependency.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const page = await readFile(new URL('./dist/code/index.html', import.meta.url), 'utf8');
const markup = page.split('<script>')[0];
assert.ok(markup.includes('href="../poster.css"'));
assert.ok(markup.includes('class="masthead"'));
assert.ok(markup.includes('class="brand"'));
assert.ok(markup.includes('class="intro"'));
assert.ok(!/<small\b|class="muted"|class="badge"/.test(page));
assert.ok(!/#[a-f\d]{3,8}\b|font-size\s*:/.test(markup));
const code = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value:'', checked:false, textContent:'', innerHTML:'', events:{},
      append(){}, addEventListener(name, callback){this.events[name]=callback;}});
    return elements.get(id);
  },
  createElement(){return {};},
};
new Function('document', code)(document);
const results = () => elements.get('results').innerHTML;
const input = (id, value) => {
  const element = elements.get(id);
  if (typeof value === 'boolean') element.checked = value;
  else element.value = value;
  element.events.input();
};
assert.ok(results().includes('locomotion/backends/isaac/env.py'));
assert.ok(!results().includes('Compatibility import'));
assert.ok(!results().includes('Source identity'));
assert.ok(!results().includes('No docstring.'));
input('query', 'ProximityGuard');
assert.ok(results().includes('locomotion/tasks/task.py'));
assert.ok(results().includes('ProximityGuard'));
assert.ok(/blob\/[a-f0-9]{40}\/locomotion\/tasks\/task.py#L\d+/.test(results()));
input('area', 'learning');
assert.ok(results().includes('No matching modules.'));
input('query', 'VanillaVecEnv');
assert.ok(results().includes('locomotion/learning/ppo.py'));
input('area', '');
input('query', 'Compatibility import for ppo');
assert.ok(results().includes('No matching modules.'));
input('compatibility', true);
assert.ok(results().includes('locomotion/ppo.py'));
input('query', 'test_native_ranges');
assert.ok(results().includes('No matching modules.'));
input('tests', true);
assert.ok(results().includes('locomotion/tests/test_native_ranges.py'));
input('query', 'emitted_target');
assert.ok(results().includes('locomotion/core/motor.py'));
console.log('Code-map class search, function search, filters and revision links verified');
