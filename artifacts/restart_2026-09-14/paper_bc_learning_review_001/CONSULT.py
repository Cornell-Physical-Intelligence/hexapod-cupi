"""One bounded, tools-disabled Claude consultation from the exact recorded request."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OUT = Path(__file__).resolve().parent
request = json.loads((OUT/"REQUEST.json").read_text())
prompt = (OUT/request["stdin"]).read_text()
assert hashlib.sha256(prompt.encode()).hexdigest() == request["input_sha256"]
for path,digest in request["source_inputs"].items():
    assert hashlib.sha256((Path(request["cwd"])/path).read_bytes()).hexdigest() == digest
start = datetime.datetime.now(datetime.timezone.utc).isoformat()
started = time.monotonic()
process = subprocess.Popen(request["argv"], cwd=request["cwd"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, start_new_session=True)
timeout = False
try:
    stdout,stderr = process.communicate(prompt,timeout=request["timeout_seconds"])
except subprocess.TimeoutExpired:
    timeout = True
    os.killpg(process.pid,signal.SIGTERM)
    try:
        stdout,stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        stdout,stderr = process.communicate()
for name,content in (("STDOUT.txt",stdout),("STDERR.txt",stderr)):
    with (OUT/name).open("x") as stream:
        stream.write(content)
parsed = None
try:
    parsed = json.loads(stdout)
except json.JSONDecodeError:
    pass
if parsed is not None:
    with (OUT/"RESPONSE.json").open("x") as stream:
        json.dump(parsed,stream,indent=2);stream.write("\n")
record = {"started_utc":start,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "elapsed_seconds":time.monotonic()-started,"exit_code":process.returncode,
          "timed_out":timeout,"requested_model":request["model"],"requested_effort":request["effort"],
          "requested_session":request["session_id"],"parsed_json":parsed is not None,
          "consult_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if isinstance(parsed,dict):
    record.update({key:parsed.get(key) for key in ("session_id","is_error","subtype","usage","modelUsage","total_cost_usd","duration_ms")})
    record["response_words"] = len(str(parsed.get("result","")).split())
with (OUT/"RECEIPT.json").open("x") as stream:
    json.dump(record,stream,indent=2);stream.write("\n")
print(json.dumps(record))
