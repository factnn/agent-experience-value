"""Four GPU lanes, fixed source-group jobs; stop future launches on failure, no retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time
from supervise_bfcl_stage_fork import ROOT,read,write,supervise
from bfcl_diagnostic_preflight import preflight


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True);args=ap.parse_args();p=read(args.protocol)
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest()==args.protocol_sha256
    for path,sha in p['input_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    for eid in p['required_review_events']:assert read(ROOT/'experiments/reviews'/(eid+'.json'))['status']=='reviewed'
    initial={gpu:preflight(gpu,p) for gpu in p['allowed_gpus']};assert len(initial)<=4
    root=ROOT/p['output_root'];root.mkdir(parents=True,exist_ok=False)
    write(root/'launch.json',{'supervisor_pid':os.getpid(),'started_unix':time.time(),'protocol_sha256':args.protocol_sha256,'preflights':initial,'max_simultaneous_gpus':4})
    lock=threading.Lock();stop=threading.Event();state={'active':{},'complete':[],'pending':[c['name'] for c in p['cases']],'failures':[]}
    def publish():write(root/'queue.json',state)
    def audit(case=None):
        cmd=[str(ROOT/'.venv-rl/bin/python'),str(ROOT/'pipeline/report_bfcl_task_probe.py'),'--protocol',str(args.protocol.resolve())]
        if case:cmd+=['--case',case]
        with (root/((case or 'batch')+'.audit.console.log')).open('x') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        assert r.returncode==0,'Probe audit failed: '+str(case)
    def worker(gpu):
        for name in p['gpu_lanes'][str(gpu)]:
            if stop.is_set():return
            case=next(c for c in p['cases'] if c['name']==name)
            try:
                checked=preflight(gpu,p)
                with lock:
                    if stop.is_set():return
                    state['pending'].remove(name);state['active'][str(gpu)]=name;publish()
                write(root/(name+'.preflight.json'),checked)
                result=supervise([str(ROOT/'.venv-rl/bin/python'),'-u',str(ROOT/'pipeline/run_bfcl_task_probe.py'),'--protocol',str(args.protocol.resolve()),'--case',name],root/name,root/(name+'.console.log'),case['hard_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8'))
                assert result['returncode']==0 and not result['hard_timeout'],'Probe execution failed: '+name
                audit(name)
                with lock:state['active'].pop(str(gpu));state['complete'].append(name);publish()
            except Exception as e:
                with lock:
                    state['active'].pop(str(gpu),None);state['failures'].append({'case':name,'gpu':gpu,'type':type(e).__name__,'message':str(e)});stop.set();publish();write(root/'failures.json',state['failures'])
                return
    publish()
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,p['allowed_gpus']))
    if not state['failures'] and not state['pending']:
        try:audit()
        except Exception as e:write(root/'failures.json',[{'type':type(e).__name__,'message':str(e)}]);raise
    write(root/'supervisor.json',{'status':'audited_complete' if not state['failures'] and not state['pending'] else 'incomplete_requires_review','complete_cases':len(state['complete']),'pending_cases':state['pending'],'automatic_retry':False})

if __name__=='__main__':main()
