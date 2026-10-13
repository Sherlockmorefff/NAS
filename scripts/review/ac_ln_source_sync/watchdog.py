"""Supplemental process supervision; never changes scientific inputs or code."""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path('/home/fangxy/D-VAE/results/ac_open_graph_mix_20261012v1')
PYTHON='/home/fangxy/.conda/envs/nas/bin/python'
PROJECT=Path('/home/fangxy/D-VAE')
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,x):
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def log(x):
    with (ROOT/'ERRORS.jsonl').open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def owned(pid):
    if not pid:return False
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and str(ROOT).encode() in p.read_bytes()
def workers():
    result=set()
    for p in list(ROOT.glob('worker?.json'))+list(ROOT.glob('gp_worker?.json')):
        x=read(p)
        for key in ['pid','numerical_pid']:
            if owned(x.get(key)):result.add(x[key])
    return result
def launch(action,extra=()):
    log_file=(ROOT/('controller.log' if action=='run' else 'acceptance.log')).open('a')
    child=subprocess.Popen([PYTHON,'-u','-B','-m','analyse.ac_open_graph_mix',action,'--run-dir',str(ROOT),*extra],
        cwd=PROJECT,stdout=log_file,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    log_file.close();return child.pid

def main():
    lock=(ROOT/'supervision.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(ROOT/'SUPERVISION_MANIFEST.json',{'file':str(Path(__file__)),'sha256':digest,'added_after_freeze':True,
      'purpose':'routine engineering heartbeat and exact checkpoint recovery only; scientific protocol/source/initialization/seeds unchanged',
      'at':now(),'freeze_identity':read(ROOT/'freeze.json')['identity']})
    repairs=0
    while True:
        status=read(ROOT/'STATUS.json')
        try:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True,timeout=10).stdout
        except subprocess.TimeoutExpired:gpu='GPU query timed out; recorded as monitoring issue'
        write(ROOT/'SUPERVISION_HEARTBEAT.json',{'at':now(),'pid':os.getpid(),'controller_pid':status.get('pid'),'controller_owned_alive':owned(status.get('pid')),
          'worker_pids':sorted(workers()),'state':status['state'],'recoveries':repairs,'GPU_snapshot':gpu,
          'controller_log_mtime':(ROOT/'controller.log').stat().st_mtime,'task_progress':status.get('counts')})
        if (ROOT/'acceptance_schedule.json').exists() and (ROOT/'acceptance_status.json').exists():
            acceptance=read(ROOT/'acceptance_status.json');schedule=read(ROOT/'acceptance_schedule.json')
            if acceptance.get('state') in ['WAITING_FIRST_20_FORMAL_ROUNDS','WAITING_UNTIL_DUE'] and not schedule['first_triggered']:
                acceptance.update(state='WAITING_UNTIL_DUE',due=schedule['due'],updated_at=now());write(ROOT/'acceptance_status.json',acceptance)
        if status['state']=='COMPLETE':return
        if status['state'] in ['FAILED','RECOVERING']:
            # The original acceptance process handles recognized infrastructure failures.
            time.sleep(60);continue
        unhealthy=not owned(status.get('pid')) or time.time()-status.get('heartbeat',time.time())>180
        for p in ROOT.glob('worker?.json'):
            x=read(p)
            if x.get('state')=='running' and x.get('current_task'):
                checkpoint=ROOT/'training'/x['current_task']/'resume.pt'
                last=checkpoint.stat().st_mtime if checkpoint.exists() else x.get('heartbeat',time.time())
                if time.time()-last>3600:unhealthy=True
        for p in ROOT.glob('gp_worker?.json'):
            x=read(p)
            if x.get('state')=='GP_and_acquisition' and time.time()-x['started']>7200:unhealthy=True
        if unhealthy:
            if repairs>=3:
                status.update(state='FAILED',error='Process supervision exhausted three same-protocol recoveries');write(ROOT/'STATUS.json',status);return
            repairs+=1;log({'at':now(),'phase':'supplemental_process_recovery','attempt':repairs,'reason':'lost controller/heartbeat or stalled checkpoint',
               'scientific_protocol_unchanged':True,'completed_tasks_preserved':True})
            oldpid=status.get('pid');(ROOT/'STOP').write_text(now());status.update(state='RECOVERING');write(ROOT/'STATUS.json',status)
            end=time.monotonic()+600
            while workers() and time.monotonic()<end:time.sleep(2)
            for pid in workers():
                if owned(pid):os.kill(pid,signal.SIGTERM)
            if owned(oldpid):os.kill(oldpid,signal.SIGTERM)
            time.sleep(3)
            for pid in workers():
                if owned(pid):os.kill(pid,signal.SIGKILL)
            write(ROOT/'SUPERVISION_RECOVERY.json',{'at':now(),'new_controller_pid':launch('run',['--recover']),'attempt':repairs,
                    'no_completed_training_repeated':True,'no_scientific_code_change':True})
            time.sleep(15)
            a=read(ROOT/'acceptance_status.json') if (ROOT/'acceptance_status.json').exists() else {}
            if a.get('state')=='FAILED_OR_PAUSED' or not owned(a.get('pid')):
                write(ROOT/'SUPERVISION_ACCEPTANCE_RESTART.json',{'pid':launch('timer'),'at':now()})
        time.sleep(60)

if __name__=='__main__':main()
