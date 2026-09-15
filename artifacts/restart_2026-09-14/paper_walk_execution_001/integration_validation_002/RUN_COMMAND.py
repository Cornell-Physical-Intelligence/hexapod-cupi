from pathlib import Path
import datetime, json, os, subprocess, sys, time
root=Path(__file__).resolve().parents[4]
out=Path(__file__).resolve().parent
name=sys.argv[1]
commands=json.loads((out/"COMMANDS.json").read_text())
argv=commands[name]
receipt=out/(name+".json")
assert not receipt.exists()
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
t0=time.monotonic()
with (out/(name+".stdout.txt")).open("xb") as stdout, (out/(name+".stderr.txt")).open("xb") as stderr:
 result=subprocess.run(argv,cwd=root,stdout=stdout,stderr=stderr,check=False)
r={"name":name,"argv":argv,"cwd":str(root),"started_at_utc":start,"ended_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"wall_seconds":time.monotonic()-t0,"exit_code":result.returncode,"stdout":stdout.name,"stderr":stderr.name}
with receipt.open("x") as stream: json.dump(r,stream,indent=2);stream.write("\n")
print(json.dumps(r))
sys.exit(result.returncode)
