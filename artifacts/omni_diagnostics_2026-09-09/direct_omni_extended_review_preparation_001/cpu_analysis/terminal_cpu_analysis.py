"""Prepared terminal-only CPU executor. Root must authorize/copy/run it; never auto-dispatches."""
import argparse,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path

ANALYZER_FREEZE='be6625b977aa8ee333ecf1aa744bed99ba16d991dca0f3679c53414bbdc8bd94'
PYTHON='/opt/wx/venv-models/bin/python'
SOURCE='aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def inventory(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic input path')
    return {p.relative_to(root).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(root.rglob('*')) if p.is_file()}

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('campaign','cold','analyzer','output','terminal-audit'):parser.add_argument('--'+key,type=Path,required=True)
    for key in ('campaign-sha256','terminal-audit-sha256'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();campaign,cold,analyzer,output=[getattr(args,k).resolve() for k in ('campaign','cold','analyzer','output')]
    if output.exists() or any(root==output or root in output.parents or output in root.parents for root in (campaign,cold,analyzer)):raise ValueError('Fresh separate output required')
    if sha(campaign/'campaign.json')!=args.campaign_sha256 or sha(args.terminal_audit)!=args.terminal_audit_sha256:raise ValueError('Changed terminal campaign/audit')
    record=json.loads((campaign/'campaign.json').read_text())
    if record.get('status') not in ('completed','failed') or record.get('allocation')!='extended' or record.get('branch')!='caps' or record.get('identity',{}).get('source_manifest_sha256')!=SOURCE:raise ValueError('Exact terminal extended CAPS campaign required')
    if sha(analyzer/'FREEZE_SHA256.json')!=ANALYZER_FREEZE:raise ValueError('Wrong frozen analyzer')
    analyzer_before=inventory(analyzer);expected=json.loads((analyzer/'FREEZE_SHA256.json').read_text())
    if {k:v['sha256'] for k,v in analyzer_before.items() if k!='FREEZE_SHA256.json'}!=expected:raise ValueError('Changed/unlisted analyzer payload')
    # Root's audit remains responsible for exact owner IDs, cleanup, source/assets and restoration.
    # This utility validates its pinned bytes; it does not substitute for that audit.
    before={'campaign':inventory(campaign),'cold':inventory(cold)}
    output.mkdir(parents=True)
    write(output/'remote_inputs_before.json',before)
    write(output/'terminal_audit_binding.json',{'path':str(args.terminal_audit),'sha256':args.terminal_audit_sha256,'scope':'Externally reviewed root terminal audit; this executor only pins its bytes.'})
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    cmd=[PYTHON,'-B',str(analyzer/'analyze.py'),'--campaign',str(campaign),'--cold',str(cold),'--output',str(output/'analysis')]
    started=time.time();result={'command':cmd,'started_unix':started,'analyzer_freeze_sha256':ANALYZER_FREEZE,'campaign_sha256':args.campaign_sha256,'python':PYTHON,'threads':1,'GPU_dispatched':False,'Stage2_complete':False,'remote_full_analysis_attempted':True}
    try:
        with (output/'stdout.log').open('wb') as stdout,(output/'stderr.log').open('wb') as stderr:
            process=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=900)
        result['exit_code']=process.returncode
    except Exception as exc:result['executor_error']=repr(exc)
    finally:
        result['finished_unix']=time.time();result['wall_seconds']=time.time()-started
        result['max_rss_kib_linux']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        result.update(complete_raw_trees_unchanged=False,analyzer_unchanged=False,terminal_audit_unchanged=False)
        try:
            after={'campaign':inventory(campaign),'cold':inventory(cold)};write(output/'remote_inputs_after.json',after)
            result['complete_raw_trees_unchanged']=after==before
            result['analyzer_unchanged']=inventory(analyzer)==analyzer_before
            result['terminal_audit_unchanged']=sha(args.terminal_audit)==args.terminal_audit_sha256
        except Exception as exc:result['final_integrity_error']=repr(exc)
        report_path=output/'analysis/report.json'
        if report_path.is_file():
            try:
                report=json.loads(report_path.read_text());result['analyzer_evidence_verified']=report.get('evidence_verified') is True;result['analyzer_errors']=report.get('errors');result['analyzer_report_sha256']=sha(report_path)
            except Exception as exc:result['report_read_error']=repr(exc)
        expected_outputs=[output/'analysis'/name for name in ('REPORT.md','report.json','INPUTS_SHA256.json')]
        result['analysis_outputs_sha256']={p.name:sha(p) for p in expected_outputs if p.is_file()}
        result['analysis_outputs_complete']=len(result['analysis_outputs_sha256'])==3 and 'report_read_error' not in result
        result['execution_receipt_passed']=result.get('exit_code')==0 and result['analysis_outputs_complete'] and result['complete_raw_trees_unchanged'] and result['analyzer_unchanged'] and result['terminal_audit_unchanged']
        result['scope']='CPU analysis receipt, not policy quality, root terminal-audit replacement or local replay of omitted raw.'
        write(output/'execution_receipt.json',result)
    print(json.dumps(result,indent=2))
    return 0 if result['execution_receipt_passed'] else 1

if __name__=='__main__':sys.exit(main())
