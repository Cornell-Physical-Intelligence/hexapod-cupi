#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
record=json.loads((here/'review.json').read_text())
target=here.parent/'full_ppo_continuation_v1'
assert record['candidate_bytes_unchanged'] is True and record['tests_pass'] is True and record['exit_code']==0
assert record['candidate_sha256_before']==record['candidate_sha256_after']
for name,expected in record['candidate_sha256_after'].items():
 assert hashlib.sha256((target/name).read_bytes()).hexdigest()==expected,name
text=(here/'cpu_tests.stderr.txt').read_text()
assert 'Ran 24 tests' in text and '\nOK\n' in text
print('PASS: reviewed coordinator/test bytes and independent 24-test result')
