"""Two independent numerical CPU workers with external hard supervision."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .common import ROOT,B,AC,ZERO,TERMINAL,read,write,append,db,now,alive,setup

def counts(root):
    with db(root) as c:return {r['state']:r['n'] for r in c.execute('SELECT state,COUNT(*) n FROM runs GROUP BY state')}

def compute(root,worker):
    setup()
    from .checks import verify
    from .geometry import Decoder,WL
    from .problem import Problem
    from .algorithms import Run
    started=time.perf_counter();verify(root);wl=WL(read(B/'groups/0/wl_vocabulary.json'))
    problems={(f,n):Problem(f,n,wl) for n in [50,150,250] for f in ['A','WI']};decoder=Decoder()
    append(root/'preparation_events.jsonl',{'role':'numerical_worker','worker':worker,'pid':os.getpid(),'seconds':time.perf_counter()-started,'at':now()})
    write(root/f'compute{worker}.json',{'pid':os.getpid(),'worker':worker,'threads':1,'device':'CPU','GP_dtype':'float64','VAE_dtype':'float32','at':now()})
    while True:
        with db(root) as c:
            c.execute('BEGIN IMMEDIATE');row=c.execute('SELECT * FROM runs WHERE state="queued" ORDER BY ord LIMIT 1').fetchone()
            if row is None:c.execute('COMMIT');break
            c.execute('UPDATE runs SET state="running",worker=?,pid=? WHERE id=?',(worker,os.getpid(),row['id']));c.execute('COMMIT')
        spec=json.loads(row['spec']);folder=root/'runs'/spec['id'];folder.mkdir(parents=True,exist_ok=True)
        write(folder/'assignment.json',{'worker':worker,'pid':os.getpid(),'at':now(),'spec':spec})
        run=None
        try:
            run=Run(root,spec,problems[(spec['family'],spec['n'])],decoder);result=run.execute()
        except Exception as exc:
            result={**spec,**ZERO,'state':'failed','stop_reason':'NUMERICAL_OR_ENGINEERING_FAILURE','error_type':type(exc).__name__,
              'error':str(exc),'traceback':traceback.format_exc(),'ended_at':now(),'requests':0 if run is None else run.requests,
              'post_requests':0 if run is None else run.post_requests,'elapsed_seconds':0. if run is None or run.started is None else time.perf_counter()-run.started,
              'times':{} if run is None else run.times,'returned':None,'realized_best':None,'improvement_approx_validation':None}
            write(folder/'result.json',result)
        with db(root) as c:c.execute('UPDATE runs SET state=?,start=?,end=?,error=? WHERE id=?',
               (result['state'],None if run is None else run.start_wall,time.time(),result.get('error'),row['id']))
        append(root/'run_events.jsonl',{'id':spec['id'],'state':result['state'],'elapsed_seconds':result['elapsed_seconds'],'worker':worker,'at':now()})
    write(root/f'compute{worker}_exit.json',{'exit_code':0,'pid':os.getpid(),'at':now()})

def interrupted_result(root,row,state,reason):
    spec=json.loads(row['spec']);folder=root/'runs'/row['id'];folder.mkdir(parents=True,exist_ok=True)
    progress=read(folder/'progress.json') if (folder/'progress.json').exists() else {}
    # Preserve any completed result whose budget ended before the hard deadline.
    if (folder/'result.json').exists():
        result=read(folder/'result.json')
        if result['state'] in TERMINAL and result.get('elapsed_seconds',301)<=300:
            with db(root) as c:c.execute('UPDATE runs SET state=?,end=? WHERE id=?',(result['state'],time.time(),row['id']))
            return
    result={**spec,**ZERO,'state':state,'stop_reason':reason,'ended_at':now(),'requests':progress.get('requests',0),
      'post_requests':progress.get('post_requests',0),'elapsed_seconds':max(0.,time.time()-progress.get('start_unix',time.time())) if progress.get('start_unix') else 0.,
      'partial_trace_retained':True,'returned':None,'realized_best':None,'improvement_approx_validation':None,'times':{}}
    write(folder/'result.json',result)
    with db(root) as c:c.execute('UPDATE runs SET state=?,end=?,error=? WHERE id=?',(state,time.time(),reason,row['id']))
    append(root/'run_events.jsonl',{'id':row['id'],'state':state,'reason':reason,'at':now()})

def supervisor(root,worker):
    lock=(root/f'worker{worker}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(root/f'worker{worker}.json',{'pid':os.getpid(),'worker':worker,'role':'external_deadline_supervisor','at':now()})
    while not (root/'READY.json').exists():
        if (root/'FAILED.json').exists():return 1
        time.sleep(.5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    while True:
        with db(root) as c:
            queued=c.execute('SELECT COUNT(*) FROM runs WHERE state="queued"').fetchone()[0]
        if not queued:break
        child=subprocess.Popen([sys.executable,'-u','-B','-m','analyse.acq_optimizer_offline','compute','--run-dir',str(root),'--worker',str(worker)],env=env,start_new_session=True)
        write(root/f'worker{worker}_child.json',{'pid':child.pid,'supervisor_pid':os.getpid(),'at':now()})
        timed_out=False
        while child.poll() is None:
            with db(root) as c:row=c.execute('SELECT * FROM runs WHERE state="running" AND pid=?',(child.pid,)).fetchone()
            if row is not None:
                p=root/'runs'/row['id']/'progress.json'
                if p.exists():
                    progress=read(p);start=progress.get('start_unix')
                    if start is not None and time.time()-start>=300:
                        os.killpg(child.pid,signal.SIGKILL);child.wait();timed_out=True
                        interrupted_result(root,row,'timeout','EXTERNAL_300_SECOND_DEADLINE');break
            time.sleep(.05)
        code=child.wait()
        with db(root) as c:orphans=c.execute('SELECT * FROM runs WHERE state="running" AND pid=?',(child.pid,)).fetchall()
        for row in orphans:interrupted_result(root,row,'failed','INTERRUPTED_NO_EXACT_SOLVER_RECOVERY')
        append(root/'worker_child_exits.jsonl',{'worker':worker,'pid':child.pid,'code':code,'timeout':timed_out,'at':now()})
        if code!=0 and not timed_out and not orphans:
            write(root/'FAILED.json',{'state':'FAILED','reason':'numerical worker failed outside a run','worker':worker,'exit':code,'at':now()});return 1
    write(root/f'worker{worker}_exit.json',{'exit_code':0,'at':now(),'pid':os.getpid()})
    return 0

def controller(root,prompt=None,resume=False):
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():return 0
    if resume:
        with db(root) as c:running=c.execute('SELECT * FROM runs WHERE state="running"').fetchall()
        for row in running:
            if alive(row['pid']):raise RuntimeError('A numerical worker is still alive; refusing duplicate resume')
            interrupted_result(root,row,'failed','INTERRUPTED_NO_EXACT_SOLVER_RECOVERY')
    try:
        setup()
        from .checks import prepare,sanity,verify
        start=time.time();write(root/'status.json',{'state':'CHECKING','pid':os.getpid(),'started_unix':start,'at':now(),**ZERO})
        prepare(root,prompt)
        if not (root/'sanity_report.json').exists():sanity(root)
        if read(root/'sanity_report.json')['state']!='PASS':raise ValueError('Common numerical checks not passed')
        verify(root);write(root/'READY.json',{'at':now(),'checks':'PASS','pid':os.getpid()})
        from .report import update_eta,final_report,accept
        while True:
            if (root/'FAILED.json').exists():raise RuntimeError(read(root/'FAILED.json')['reason'])
            ct=counts(root);complete=sum(ct.get(k,0) for k in TERMINAL)
            write(root/'status.json',{'state':'RUNNING','pid':os.getpid(),'started_unix':start,'heartbeat':time.time(),'counts':ct,'terminal':complete,'total':36,'at':now(),**ZERO})
            if complete>=4:update_eta(root)
            if complete==36:break
            for worker in [0,1]:
                p=root/f'worker{worker}.json'
                if p.exists() and not alive(read(p)['pid']) and not (root/f'worker{worker}_exit.json').exists():raise RuntimeError('External worker supervisor died')
            time.sleep(2)
        for worker in [0,1]:
            deadline=time.monotonic()+30
            while not (root/f'worker{worker}_exit.json').exists():
                if time.monotonic()>deadline:raise RuntimeError('Supervisor did not exit after queue completed')
                time.sleep(.2)
        verify(root);final_report(root);receipt=accept(root)
        state='COMPLETE' if receipt['all_36_valid_complete'] else 'COMPLETE_WITH_FAILURES'
        write(root/'COMPLETE.json',{'state':state,'at':now(),'acceptance':receipt,**ZERO})
        write(root/'status.json',{'state':state,'pid':os.getpid(),'started_unix':start,'counts':counts(root),'at':now(),**ZERO})
        return 0
    except Exception as exc:
        write(root/'FAILED.json',{'state':'FAILED','reason':str(exc),'traceback':traceback.format_exc(),'at':now(),**ZERO})
        write(root/'status.json',{'state':'FAILED','pid':os.getpid(),'error':str(exc),'at':now(),**ZERO});return 1
