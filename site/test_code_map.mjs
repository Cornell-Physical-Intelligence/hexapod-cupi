// Test the generated map's drawings and interactions without a browser dependency.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

const page = await readFile(new URL('./dist/code/index.html', import.meta.url), 'utf8');
const markup = page.split('<script>')[0];
assert.ok(markup.includes('href="../poster.css"'));
assert.ok(markup.includes('class="masthead"'));
assert.ok(markup.includes('class="brand"'));
assert.ok(markup.includes('class="intro"'));
assert.ok(!/<small\b|class="muted"|class="badge"/.test(page));
assert.ok(!/#[a-f\d]{3,8}\b|font-size\s*:/.test(markup), 'Colours and type come from poster.css');
assert.ok(!/box-shadow|border-radius|border-left\s*:/.test(markup), 'No card styles');
const code = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value:'', checked:false, textContent:'', innerHTML:'', events:{},
      setAttribute(){}, addEventListener(name, callback){(this.events[name] ||= []).push(callback);}});
    return elements.get(id);
  },
};
new Function('document', code)(document);
const html = (id) => elements.get(id).innerHTML;
const fire = (id, name, event={}) => elements.get(id).events[name].forEach((f) => f(event));
const input = (id, value) => {
  const element = elements.get(id);
  if (typeof value === 'boolean') element.checked = value;
  else element.value = value;
  fire(id, 'input');
};
const click = (id, data) => fire(id, 'click', {target:{closest:() => ({dataset:data})}});

// Packages: one square each, sized and labelled, with arrows for imports between packages.
for (const name of ['gait_clock', 'behavior_cloning', 'priors', 'surrogate', 'tools']) {
  assert.ok(html('packages').includes(`data-package="${name}"`), `Missing package ${name}`);
}
assert.ok(!html('packages').includes('data-package="tests"'), 'Tests stay hidden by default');
assert.ok(/class="use[^"]*" data-edge="[^"]+&gt;[^"]+"/.test(html('packages')), 'Package arrows are drawn');
assert.ok(html('packages').includes('data-edge="gait_clock&gt;priors"'), 'The gait-clock environment imports motion features');
assert.ok(/<rect x="[\d.]+" y="[\d.]+" width="[\d.]+" height="[\d.]+"\/>/.test(html('packages')));

// Selecting a package draws its files; selecting a file draws its definitions bar.
click('packages', {package:'gait_clock'});
assert.ok(html('packages').includes('class="pkg selected" data-package="gait_clock"'));
assert.ok(html('files').includes('data-module="locomotion/gait_clock/task.py"'));
click('files', {module:'locomotion/gait_clock/task.py'});
assert.equal(elements.get('file-heading').textContent, 'gait_clock/task.py');
assert.ok(/class="sym class[^"]*"/.test(html('strip')), 'Classes appear as blocks');
assert.ok(/blob\/[a-f0-9]{40}\/locomotion\/gait_clock\/task.py#L\d+/.test(html('strip')), 'Blocks link to source lines');
assert.ok(html('file').includes('Imported by:') && html('file').includes('Imports:'));

// Search marks matches in red and opens the first matching file.
input('query', 'StrideWindow');
assert.ok(elements.get('count').textContent.startsWith('1 file matches'));
assert.ok(html('results').includes('data-file="locomotion/gait_clock/task.py"'));
assert.ok(/class="pkg[^"]* hit" data-package="gait_clock"/.test(html('packages')));
assert.ok(/class="file[^"]* hit" data-module="locomotion\/gait_clock\/task.py"/.test(html('files')));
assert.ok(/class="sym class hit"/.test(html('strip')), 'The matching class block turns red');
input('query', 'VanillaVecEnv');
assert.ok(html('results').includes('locomotion/gait_clock/ppo.py'));
assert.equal(elements.get('file-heading').textContent, 'gait_clock/ppo.py');
click('results', {file:'locomotion/gait_clock/control.py'});
assert.equal(elements.get('file-heading').textContent, 'gait_clock/control.py');

// Retired wrappers have no filter; tests retain their toggle.
assert.ok(!markup.includes('id="compatibility"'));
input('query', 'Compatibility import for ppo');
assert.ok(html('results').includes('No matching files.'));
input('query', 'test_native_ranges');
assert.ok(html('results').includes('No matching files.'));
input('tests', true);
assert.ok(html('results').includes('locomotion/tests/test_native_ranges.py'));
assert.ok(html('packages').includes('data-package="tests"'));
input('query', 'emitted_target');
assert.ok(html('results').includes('locomotion/gait_clock/control.py'));
console.log('Code-map package squares, import arrows, file tiles, definition bar, search marking and toggles verified');
