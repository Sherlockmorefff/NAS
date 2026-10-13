import fcntl
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from .common import *

def stamp(t):
    return datetime.fromtimestamp(t,timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %z')

def worker(root,index):
    import torch
    from deterministic_runtime import configure_torch_determinism
    from .prepare import bundle_data
    from .training import train_validation_only,training_view
    from .checks import initialization
    torch.set_default_dtype(torch.float32);configure_torch_determinism(torch);torch.set_num_threads(1)
    verify(root)
    bundle,context=bundle_data()
    if context!=read_json(root/'PROTOCOL.json')['dataset_context'] or not torch.cuda.is_available():
        raise ValueError('Formal worker requires original data and GPU')
    device=torch.device('cuda:0');data=training_view(bundle.data.to(device))
    mapping={t['task_id']:t for t in tasks(root)}
    gpu=read_json(root/'resources.json')['GPUs'][index]
    started_worker=time.time()
    write_json(root/('worker'+str(index)+'.json'),{'pid':os.getpid(),'index':index,'GPU':gpu,'started':started_worker,'state':'RUNNING'})
    while not (root/'STOP').exists():
        with connect(root) as db:
            db.execute('BEGIN IMMEDIATE')
            first=(root/'FIRST_10_TIMING.json').exists()
            row=db.execute('SELECT * FROM tasks WHERE status=?'+('' if first else ' AND ordinal<10')+' ORDER BY ordinal LIMIT 1',('queued',)).fetchone()
            if row is None:
                done=db.execute("SELECT count(*) FROM tasks WHERE status='succeeded'").fetchone()[0]
                db.commit()
                if done==300:break
                time.sleep(.2);continue
            task=mapping[row['id']];attempt=row['attempts']+1;begin=time.time()
            db.execute("UPDATE tasks SET status='running',worker=?,pid=?,started=?,attempts=? WHERE id=?",(index,os.getpid(),begin,attempt,task['task_id']))
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?)',(task['task_id'],attempt,index,begin,None,None,None));db.commit()
        f=root/'training'/task['task_id'];f.mkdir(parents=True,exist_ok=True)
        write_json(root/('worker'+str(index)+'.json'),{'pid':os.getpid(),'index':index,'GPU':gpu,'started':started_worker,'state':'RUNNING','task':task['task_id'],'heartbeat':time.time()})
        try:
            init=initialization(task['configuration'],task['seed'],bundle.num_features,bundle.num_classes,device)
            p=f/'INITIALIZATION_CHECK.json'
            if p.exists():
                if read_json(p)!=init:raise ValueError('Same-task initialization evidence changed')
            else:write_json(p,init)
            write_json(f/'TRAINING_ACCESS_BOUNDARY.json',{'full_graph_features_edges':True,'test_mask_in_training_view':hasattr(data,'test_mask'),
                       'test_labels_redacted':True,'historical_test_CSV_loaded':False,'dataset_context':context})
            value=train_validation_only(task['configuration'],data,bundle.num_features,bundle.num_classes,task['seed'],device,f,task['task_id'],max_epochs=300,patience=80)
            result(root,task)
            end=time.time();execution={'task_id':task['task_id'],'attempt':attempt,'worker':index,'pid':os.getpid(),
                'physical_gpu':gpu,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'logical_device':str(device),
                'started_unix':begin,'finished_unix':end,'task_wall_seconds':end-begin,'training_seconds':value['elapsed_seconds'],
                'queue_wait_seconds':begin-read_json(root/'QUEUE_ENQUEUED.json')['all_300_enqueued_unix'],
                'component_variant':'layer_norm','from_scratch':True,'None_training_calls':0,'test_metric_computations':0}
            write_json(f/'execution.json',execution)
            with connect(root) as db:
                db.execute("UPDATE tasks SET status='succeeded',finished=?,elapsed=?,error=NULL WHERE id=?",(end,value['elapsed_seconds'],task['task_id']))
                db.execute('UPDATE attempts SET finished=?,exit_code=0 WHERE task=? AND attempt=?',(end,task['task_id'],attempt))
            print(json.dumps({'method':task['method'],'group':task['group'],'seed_index':task['seed_index'],'task':task['task_id'],'seconds':end-begin,'best_epoch':value['best_epoch'],'stopped_epoch':value['stopped_epoch']}),flush=True)
        except Exception:
            error=traceback.format_exc();end=time.time()
            append_jsonl(root/('worker'+str(index)+'_errors.jsonl'),{'at':now(),'task':task['task_id'],'attempt':attempt,'error':error,'same_seed_resume':True})
            with connect(root) as db:
                db.execute('UPDATE attempts SET finished=?,exit_code=1,error=? WHERE task=? AND attempt=?',(end,error,task['task_id'],attempt))
                db.execute('UPDATE tasks SET status=?,error=?,finished=? WHERE id=?',('queued' if attempt<3 else 'failed',error,end,task['task_id']))
            if attempt>=3:raise
            torch.cuda.empty_cache()
    write_json(root/('worker'+str(index)+'.json'),{'pid':os.getpid(),'index':index,'GPU':gpu,'started':started_worker,'finished':time.time(),'state':'EXITED'})

def update_status(root,state,pid,start):
    with connect(root) as db:
        counts={x['status']:x['n'] for x in db.execute('SELECT status,count(*) n FROM tasks GROUP BY status')}
        failed=db.execute('SELECT count(*) FROM attempts WHERE exit_code!=0').fetchone()[0]
    status={'state':state,'pid':pid,'started_unix':start,'heartbeat':time.time(),'updated_at':now(),
            'LN_completed':counts.get('succeeded',0),'LN_total':300,'tasks':counts,'failed_attempts':failed,
            'None_historical_reused':300,'None_new_training':0,'None_new_test_scoring':0,'LN_training_test_evaluations':0}
    write_json(root/'STATUS.json',status)
    return status

def update_eta(root):
    mapping={t['task_id']:t for t in tasks(root)}
    with connect(root) as db:rows=[dict(x) for x in db.execute('SELECT * FROM tasks')]
    done=[x for x in rows if x['status']=='succeeded']
    if not all(any(x['id']==t['task_id'] for x in done) for t in mapping.values() if t['seed_index']==0):return
    from collections import defaultdict
    measured=defaultdict(list)
    for x in done:
        t=mapping[x['id']];e=read_json(root/'training'/x['id']/'execution.json')
        measured[(t['method'],t['group'])].append(e['task_wall_seconds'])
    pending=[x for x in rows if x['status']!='succeeded'];work=0.;detail=[]
    for m,g in TRAJECTORIES:
        vals=measured[(m,g)];n=sum(mapping[x['id']]['method']==m and mapping[x['id']]['group']==g for x in pending)
        mean=sum(vals)/len(vals);work+=n*mean;detail.append({'method':m,'group':g,'samples':len(vals),'mean_task_wall_seconds':mean,'remaining':n})
    capacity=len(read_json(root/'resources.json')['GPUs']);remain=work/capacity+90;at=time.time()
    eta={'at':now(),'remaining_wall_seconds':remain,'estimated_completion_unix':at+remain,'estimated_completion_Beijing':stamp(at+remain),
         'descriptive_window_Beijing':[stamp(at+remain*.7),stamp(at+remain*1.6)],'measured_tasks':len(done),'by_configuration':detail,
         'assumptions':'current LN measurements by fixed configuration; actual GPU count; 90s scoring/report; window is descriptive'}
    write_json(root/'ETA.json',eta)
    if not (root/'FIRST_10_TIMING.json').exists():
        write_json(root/'FIRST_10_TIMING.json',{'at':now(),'formal_tasks':10,'counts_in_300':True,'first_configuration_seed_only':True,'timings':detail})
        write_json(root/'acceptance_schedule.json',{'first_due_unix':at+remain+900,'first_due_Beijing':stamp(at+remain+900),'buffer_seconds':900,
                   'healthy_repeat_seconds':300,'basis':'first formal 10-task ETA; completion triggers immediate acceptance'})

def controller(root,recover=False):
    from .report import validation_seal,report,accept,package
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(root)
    if (root/'COMPLETE.json').exists():
        package(root);return
    if recover:
        for p in root.glob('worker?.json'):
            v=read_json(p)
            if v.get('state')=='RUNNING' and Path('/proc/'+str(v['pid'])).exists():
                raise RuntimeError('Cannot recover with live original worker')
        with connect(root) as db:
            db.execute("UPDATE tasks SET status='queued' WHERE status IN ('running','failed')")
            db.execute("UPDATE attempts SET finished=?,exit_code=1,error='controller/worker interrupted; same task resumed' WHERE exit_code IS NULL",(time.time(),))
        if (root/'STOP').exists():(root/'STOP').unlink()
    previous=read_json(root/'STATUS.json');start=previous.get('started_unix',time.time());pid=os.getpid()
    workers=[];logs=[];gpu=read_json(root/'resources.json')['GPUs']
    update_status(root,'RUNNING',pid,start)
    try:
        if not (root/'LN_TRAINING_SEAL.json').exists():
            for i,resource in enumerate(gpu):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(resource['physical_index']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                f=(root/('worker'+str(i)+'.log')).open('a');logs.append(f)
                workers.append(subprocess.Popen([sys.executable,'-u','-B','-m',MODULE,'worker','--root',str(root),'--index',str(i)],env=env,stdout=f,stderr=subprocess.STDOUT))
            last=0.;last_eta=0.
            while True:
                status=update_status(root,'RUNNING',pid,start)
                if time.time()-last>=60:
                    append_jsonl(root/'HEARTBEAT.jsonl',status);last=time.time()
                if not (root/'FIRST_10_TIMING.json').exists() or time.time()-last_eta>=300:
                    update_eta(root);last_eta=time.time()
                if status['tasks'].get('failed',0):raise RuntimeError('LN same-seed retry limit reached; see worker errors')
                if any(p.poll() is not None and p.returncode!=0 for p in workers):raise RuntimeError('GPU worker exited with failure')
                if status['LN_completed']==300:break
                if all(p.poll() is not None for p in workers):raise RuntimeError('Workers stopped before 300 tasks complete')
                time.sleep(1)
            codes=[p.wait(timeout=60) for p in workers];write_json(root/'worker_exit_codes.json',codes)
            if any(codes):raise RuntimeError('Unclean LN training worker exit')
            validation_seal(root)
        update_status(root,'SCORING',pid,start)
        if not (root/'LN_SEED_RESULTS.json').exists():
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu[0]['physical_index']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (root/'scoring.log').open('a') as f:
                p=subprocess.run([sys.executable,'-u','-B','-m',MODULE,'score','--root',str(root)],env=env,stdout=f,stderr=subprocess.STDOUT)
            write_json(root/'scoring_exit_code.json',p.returncode)
            if p.returncode:raise RuntimeError('Independent layer_norm scorer failed')
        update_status(root,'REPORTING',pid,start)
        report(root,start)
        accepted=accept(root)
        update_status(root,'COMPLETE',pid,start)
        write_json(root/'COMPLETE.json',{'state':'COMPLETE','completed_at':now(),'accepted':accepted})
        package(root)
        print(json.dumps({'state':'COMPLETE','acceptance':'PASS','completed_at':now()}),flush=True)
    except BaseException:
        (root/'STOP').touch();err=traceback.format_exc()
        for p in workers:
            if p.poll() is None:
                try:p.wait(timeout=360)
                except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
        status=update_status(root,'FAILED',pid,start);status['error']=err;write_json(root/'STATUS.json',status)
        append_jsonl(root/'ERRORS.jsonl',{'at':now(),'phase':'controller','error':err});raise
    finally:
        for f in logs:f.close()

def timer(root):
    # This process monitors liveness and acceptance; only the controller packages.
    import signal
    retries=0;next_due=None
    while True:
        if (root/'COMPLETE.json').exists() and (root/'package_manifest.json').exists():
            write_json(root/'acceptance_status.json',{'state':'PASS','at':now(),'delivery':read_json(root/'package_manifest.json')});return
        if not (root/'STATUS.json').exists():time.sleep(5);continue
        status=read_json(root/'STATUS.json')
        own_pid=status.get('pid');alive=own_pid and Path('/proc/'+str(own_pid)).exists()
        if status['state'] in ('FAILED','RUNNING','SCORING','REPORTING') and (not alive or status['state']=='FAILED'):
            live_workers=[read_json(p) for p in root.glob('worker?.json') if read_json(p).get('state')=='RUNNING' and Path('/proc/'+str(read_json(p)['pid'])).exists()]
            if not live_workers and not alive and retries<3:
                retries+=1;append_jsonl(root/'ERRORS.jsonl',{'at':now(),'phase':'automatic_controller_recovery','retry':retries,'same_tasks_seeds':True})
                with (root/'controller.log').open('a') as f:
                    subprocess.Popen([sys.executable,'-u','-B','-m',MODULE,'run','--root',str(root),'--recover'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(10);continue
            if not alive and not live_workers and retries>=3:
                write_json(root/'acceptance_status.json',{'state':'BLOCKED','at':now(),'error':status.get('error'),'recoveries':retries});return
        schedule=read_json(root/'acceptance_schedule.json') if (root/'acceptance_schedule.json').exists() else None
        if schedule and next_due is None:next_due=schedule['first_due_unix']
        due=next_due is not None and time.time()>=next_due
        write_json(root/'acceptance_status.json',{'state':'HEALTHY_INCOMPLETE' if due else 'WAITING_UNTIL_DUE' if schedule else 'WAITING_FIRST_10',
                   'at':now(),'pid':os.getpid(),'controller_alive':bool(alive),'LN_completed':status.get('LN_completed'),
                   'first_due_Beijing':schedule['first_due_Beijing'] if schedule else None,'recoveries':retries})
        if due:next_due=time.time()+300
        time.sleep(30)
