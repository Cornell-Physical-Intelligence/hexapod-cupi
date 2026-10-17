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
for (const name of ['omniwalk', 'hexapod', 'amp', 'surrogate', 'evaluation', 'spark', 'robots', 'rl', 'scripts']) {
  assert.ok(html('packages').includes(`data-package="${name}"`), `Missing package ${name}`);
}
assert.ok(!html('packages').includes('data-package="tests"'), 'Tests stay hidden by default');
assert.ok(/class="use[^"]*" data-edge="[^"]+&gt;[^"]+"/.test(html('packages')), 'Package arrows are drawn');
assert.ok(html('packages').includes('data-edge="amp&gt;omniwalk"'), 'The motion priors import the omni task');
assert.ok(/<rect x="[\d.]+" y="[\d.]+" width="[\d.]+" height="[\d.]+"\/>/.test(html('packages')));

// Selecting a package draws its files; selecting a file draws its definitions bar.
click('packages', {package:'omniwalk'});
assert.ok(html('packages').includes('class="pkg selected" data-package="omniwalk"'));
assert.ok(html('files').includes('data-module="src/locomotion/omniwalk/task.py"'));
click('files', {module:'src/locomotion/omniwalk/task.py'});
assert.equal(elements.get('file-heading').textContent, 'omniwalk/task.py');
assert.ok(/class="sym class[^"]*"/.test(html('strip')), 'Classes appear as blocks');
assert.ok(/blob\/[a-f0-9]{40}\/src\/locomotion\/omniwalk\/task.py#L\d+/.test(html('strip')), 'Blocks link to source lines');
assert.ok(html('file').includes('Imported by:') && html('file').includes('Imports:'));

// Search marks matches in red and opens the first matching file.
input('query', 'StrideWindow');
assert.ok(elements.get('count').textContent.startsWith('1 file matches'));
assert.ok(html('results').includes('data-file="src/locomotion/omniwalk/mdp/rewards.py"'));
assert.ok(/class="pkg[^"]* hit" data-package="omniwalk"/.test(html('packages')));
assert.ok(/class="file[^"]* hit" data-module="src\/locomotion\/omniwalk\/mdp\/rewards.py"/.test(html('files')));
assert.ok(/class="sym class hit"/.test(html('strip')), 'The matching class block turns red');
input('query', 'OmniVecEnv');
assert.ok(html('results').includes('src/locomotion/rl/wrapper.py'));
assert.equal(elements.get('file-heading').textContent, 'rl/wrapper.py');
click('results', {file:'src/locomotion/omniwalk/mdp/actions.py'});
assert.equal(elements.get('file-heading').textContent, 'omniwalk/mdp/actions.py');

// Retired wrappers have no filter; tests retain their toggle.
assert.ok(!markup.includes('id="compatibility"'));
input('query', 'Compatibility import for ppo');
assert.ok(html('results').includes('No matching files.'));
input('query', 'test_native_ranges');
assert.ok(html('results').includes('No matching files.'));
input('tests', true);
assert.ok(html('results').includes('src/locomotion/tests/test_native_ranges.py'));
assert.ok(html('packages').includes('data-package="tests"'));
input('query', 'emitted_target');
assert.ok(html('results').includes('src/locomotion/omniwalk/mdp/actions.py'));
console.log('Code-map package squares, import arrows, file tiles, definition bar, search marking and toggles verified');
