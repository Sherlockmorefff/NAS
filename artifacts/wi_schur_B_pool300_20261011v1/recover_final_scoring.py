"""Resume only final scoring/reporting with the frozen float32 GNN recipe.

The original controller left torch's default dtype at GP float64. This entry
uses the untouched frozen scoring function in a fresh process with explicit
float32; no candidate selection, fitting, or training is executed.
"""
import fcntl
import os
import sqlite3
import sys
import time
import traceback
from pathlib import Path

REPO=Path('/home/fangxy/D-VAE')
sys.path.insert(0,str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

from analyse.wi_schur_B_pool300.common import read,write,seal,append,sha,now,pid_alive,counts
from analyse.wi_schur_B_pool300.prepare import verify
from analyse.wi_schur_B_pool300.runtime import score_final
from analyse.wi_schur_B_pool300.report import final_report,accept_complete

def main():
    import torch
    root=REPO/'results/wi_schur_B_pool300_20261011v1'
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=verify(root)
    if (root/'COMPLETE.json').exists():
        print(read(root/'COMPLETE.json'),flush=True);return
    old=read(root/'status.json')
    if pid_alive(old.get('pid')):raise RuntimeError('Existing controller is still alive')
    for i in range(2):
        if pid_alive(read(root/f'worker{i}.json')['pid']):raise RuntimeError('Training worker still alive')
    c=counts(root)
    if c['actual']!={'succeeded':3150} or c['logical_enqueued']!={'search':1500,'validation':1500,'final':150}:
        raise RuntimeError('Recovery requires all3150 training tasks already committed')
    if read(root/'worker_exit_codes.json')!=[0,0]:raise RuntimeError('Training workers did not exit cleanly')
    barrier=root/'all_final_models_sealed.json'
    if not barrier.is_file():raise RuntimeError('Existing all-model test barrier is required')
    before=sha(barrier)
    torch.set_default_dtype(torch.float32)
    if torch.get_default_dtype()!=torch.float32:raise RuntimeError('GNN must be float32')
    record={'started_at':now(),'pid':os.getpid(),'entry':str(Path(__file__).resolve()),
        'entry_sha256':sha(__file__),'freeze_identity':frozen['identity'],'barrier_sha256':before,
        'original_failure':old.get('error'),'repair':'explicit float32 default in scoring-only fresh process',
        'scientific_protocol_changed':False,'additional_training_tasks':0,'GP_fits':0,
        'operations':'frozen score_final, final_report and accept_complete only'}
    append(root/'scoring_recovery.jsonl',record)
    status={k:v for k,v in old.items() if k!='error'}
    status.update(state='FINAL_TEST_SCORING',pid=os.getpid(),updated_at=now(),heartbeat=time.time())
    write(root/'status.json',status)
    started=time.perf_counter()
    try:
        scores=score_final(root)
        if len(scores)!=150 or sha(barrier)!=before:raise RuntimeError('Scoring recovery violated frozen barrier')
        write(root/'scoring_recovery_result.json',{'state':'SCORED','scores':150,'additional_training_tasks':0,
            'scoring_seconds':time.perf_counter()-started,'ended_at':now(),'entry_sha256':record['entry_sha256']})
        status.update(state='REPORT_AND_VERIFICATION',updated_at=now(),heartbeat=time.time())
        write(root/'status.json',status)
        final_report(root)
        checks=accept_complete(root)
        seal(root/'COMPLETE.json',{'state':'COMPLETE','accepted':checks,'completed_at':now(),'only_B':True,
            'scoring_recovery_entry_sha256':record['entry_sha256']})
        status.update(state='COMPLETE',updated_at=now(),heartbeat=time.time(),**counts(root))
        write(root/'status.json',status)
        report={'state':'PASS','at':now(),'checks':checks,'trigger':'user-requested status check and authorized scoring recovery',
            'original_scheduled_acceptance_time':read(root/'first_acceptance_schedule.json')['due'],
            'scheduled_time_not_changed':True}
        write(root/'immediate_acceptance_report.json',report)
        (root/'IMMEDIATE_ACCEPTANCE.md').write_text('# B 即时验收通过\n\n'+str(report)+'\n')
        print(report,flush=True)
    except BaseException:
        error=traceback.format_exc()
        append(root/'controller_failures.jsonl',{'at':now(),'entry':'scoring-only recovery','error':error})
        status.update(state='FAILED',error=error,updated_at=now(),heartbeat=time.time())
        write(root/'status.json',status)
        raise
    finally:
        lock.close()

if __name__=='__main__':main()
