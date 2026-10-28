// Read the shared run-file fixture that tests/test_run_view.py checks against viewer/run_view.py.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { center, describeFailure, point, readRun, touching } from '../src/runs/runFile.js';

const fixture = await readFile(new URL('./fixtures/sample_run.json', import.meta.url), 'utf8');

test('a run file decodes into leg points, contacts and timing', async () => {
  const run = await readRun(fixture);
  assert.equal(run.source, 'native');
  const [forward, stop] = run.cases;
  assert.equal(forward.frames, 3);
  assert.equal(forward.seconds, 0.12);
  // The left-front coxa sits 201 mm ahead of the origin, 80 mm left and 128 mm up after the first control,
  // and moves forward 2 mm per frame (two controls of 1 mm).
  assert.deepEqual(point(forward, 0, 0, 0), [0.201, 0.08, 0.128]);
  assert.deepEqual(point(forward, 1, 0, 0), [0.203, 0.08, 0.128]);
  assert.ok(Math.abs(point(forward, 0.5, 0, 0)[0] - 0.202) < 1e-9);
  assert.ok(Math.abs(center(forward, 2)[0] - center(forward, 0)[0] - 0.004) < 1e-9);
  // Legs lf, lr and rm stand on the floor; the second case's points follow the first case's.
  assert.equal(touching(forward, 0, 0), true);
  assert.equal(touching(forward, 0, 1), false);
  assert.deepEqual(point(stop, 2, 0, 0), [0.203, 0.08, 0.128]);
});

test('a file of another schema is refused and failures read in plain words', async () => {
  await assert.rejects(readRun({ schema: 'other' }), /Not a run file/);
  assert.equal(describeFailure({ check: 'terminated', value: 1, bound: 0 }), 'fell');
  assert.equal(
    describeFailure({ check: 'planar_error_mps', value: 0.098859, bound: 0.025 }),
    'planar speed error, m/s 0.0989 (limit 0.025)',
  );
});
