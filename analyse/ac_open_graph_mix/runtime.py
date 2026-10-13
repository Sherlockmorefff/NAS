import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from .common import *
from .prepare import verify

def initialize_jobs(root):
    with connect(root) as db:
        db.execute('CREATE TABLE IF NOT EXISTS jobs(method TEXT,g INTEGER,n INTEGER,observations_identity TEXT,status TEXT,pid INTEGER,worker INTEGER,started REAL,finished REAL,error TEXT,PRIMARY KEY(method,g,n))')

def training_worker(root,index):
    import torch
    from deterministic_runtime import configure_torch_determinism
    from analyse.wi_schur_B_pool300.prepare import bundle_data
    from analyse.structured_kernel_followup.validation_train import train_validation_only
    verify(root);torch.set_default_dtype(torch.float32);configure_torch_determinism(torch);torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('GNN worker must see one masked GPU')
    bundle,context=bundle_data()
    if context!=read(root/'PROTOCOL.json')['dataset_context']:raise ValueError('Training dataset changed')
    device=torch.device('cuda:0');data=bundle.data.to(device)
    gpu=next(g for g in read(root/'resources.json')['gpus'] if g['index']==index)
    info={'pid':os.getpid(),'worker':index,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
        'logical_device':'cuda:0','physical_gpu_uuid':gpu['uuid'],'physical_gpu':index,'started_at':now()}
    while not (root/'STOP').exists():
        write(root/f'worker{index}.json',dict(info,state='waiting',heartbeat=time.time(),current_task=None))
        with connect(root) as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM tasks WHERE source=? AND status=? ORDER BY rowid LIMIT 1',('OPEN-new','queued')).fetchone()
            if row:
                start=time.time();attempt=row['attempts']+1
                db.execute('UPDATE tasks SET status=?,worker=?,pid=?,started=?,attempts=?,error=NULL WHERE id=?',('running',index,os.getpid(),start,attempt,row['id']))
                db.execute('INSERT INTO attempts(task,attempt,worker,pid,started) VALUES(?,?,?,?,?)',(row['id'],attempt,index,os.getpid(),start))
        if row is None:time.sleep(1);continue
        key=row['id'];out=root/'training'/key
        write(root/f'worker{index}.json',dict(info,state='running',heartbeat=time.time(),current_task=key,stage=row['stage']))
        try:
            value=train_validation_only(json.loads(row['config']),data,bundle.num_features,bundle.num_classes,row['seed'],device,out,key,max_epochs=row['epochs'],patience=row['patience'])
            if value['identity']!=key or value['test_metric_computations']!=0:raise ValueError('Invalid committed new training result')
            end=time.time();execution={**info,'task_id':key,'candidate_id':row['cid'],'stage':row['stage'],'seed':row['seed'],
                'creator_method':row['creator_method'],'creator_group':row['creator_group'],'actual_epochs':value['stopped_epoch'],
                'training_seconds':value['elapsed_seconds'],'task_wall_seconds':end-start,'task_started_unix':start,'completed_at':now(),'attempt':attempt}
            write(out/'execution.json',execution)
            with connect(root) as db:
                db.execute('UPDATE tasks SET status=?,finished=?,elapsed=? WHERE id=?',('succeeded',end,value['elapsed_seconds'],key))
                db.execute('UPDATE attempts SET finished=?,exit_code=0 WHERE task=? AND attempt=?',(end,key,attempt))
            print(json.dumps(execution),flush=True);torch.cuda.empty_cache()
        except BaseException:
            error=traceback.format_exc();end=time.time()
            append(root/'failures.jsonl',{'task':key,'attempt':attempt,'worker':index,'at':now(),'error':error,'elapsed':end-start})
            with connect(root) as db:
                db.execute('UPDATE tasks SET status=?,finished=?,error=? WHERE id=?',('failed',end,error,key))
                db.execute('UPDATE attempts SET finished=?,exit_code=1,error=? WHERE task=? AND attempt=?',(end,error,key,attempt))
            write(root/f'worker{index}.json',dict(info,state='failed',current_task=key,heartbeat=time.time(),error=error));raise
    write(root/f'worker{index}.json',dict(info,state='stopped',current_task=None,ended_at=now()))

def job(root,method,g,n,post_only=False):
    from .search import decision
    observations=read(folder(root,method,g)/'observations.json')
    if len(observations)!=n:raise ValueError('Job observation count changed')
    decision(root,method,g,observations,post_only)

def partial_decision(root,method,g,n,reason='external_deadline'):
    f=folder(root,method,g);ad=f/'acquisition'/f'{n-49:03d}';state=read(ad/'state.json')
    rows=[json.loads(line) for line in (ad/'archive.jsonl').read_text().splitlines()]
    valid=[r['selected'] for r in rows if r['event']=='landed' and r.get('elapsed',301)<=300]
    if not valid:raise RuntimeError('NO_VALID_RETURN after enforced deadline; trajectory paused')
    selected=min(valid,key=lambda r:(-r['realized_qLogEI'],r['candidate_id'],r['source_key']))
    fit=read(f/'fits'/f'n{n:03d}'/'fit.json');cp=f/'fits'/f'n{n:03d}'/f"start{fit['selected_start']}.pt"
    obs=read(f/'observations.json');diag={'events':state['events'],'decoder_calls':state['decoder_calls'],'encoder_calls':0,
        'decoder_hits':state['decoder_hits'],'times':state['times'],'normal_requests':state['main_requests'],
        'timeout_partial':reason=='external_deadline','acquisition_exception':reason!='external_deadline','posterior_points_lower_bound':sum(r['event'] in ['scored','landed'] for r in rows),'gradient_requests':sum(r.get('gradient',False) for r in rows if r['event']=='requested'),
        'counter_lower_bounds_marked':True,'starts':state['start_calls']}
    d={'method':method,'group':g,'step':n-49,'observed_n':n,'observed_ids':[o['candidate_id'] for o in obs],
       'observations_identity':digest(obs),'candidate_id':selected['candidate_id'],'selected_row':selected,'fit_identity':fit['identity'],
       'fit_checkpoint':str(cp.relative_to(f)),'model_sha256':sha(cp),'fit_seconds':fit['fit_seconds'],'transform_seconds':fit['transform_seconds'],
       'acquisition_seconds':min(300.,state['consumed_seconds']),'requests':state['requests'],'post_requests':state['post_requests'],
       'fallback_requests':state['fallback_requests'],'diagnostics':diag,'mc_seed':read(inputs(root,g)/'seeds.json')['qlogei_mc'][n-50],
       'parameters':fit['starts'][fit['selected_start']]['parameters'],'sealed_at':now(),'return_reason':'existing_valid_archive_after_'+reason}
    path=f/'decisions'/f'{n-49:03d}.json'
    if path.exists():write(ad/'superseded_over_deadline_decision.json',read(path))
    write(path,d);return d

def gp_worker(root,index):
    import signal
    verify(root)
    while not (root/'STOP').exists():
        write(root/f'gp_worker{index}.json',{'pid':os.getpid(),'state':'waiting','heartbeat':time.time(),'worker':index})
        with connect(root) as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT * FROM jobs WHERE status=? ORDER BY rowid LIMIT 1',('queued',)).fetchone()
            if row:db.execute('UPDATE jobs SET status=?,pid=?,worker=?,started=? WHERE method=? AND g=? AND n=?',('running',os.getpid(),index,time.time(),row['method'],row['g'],row['n']))
        if row is None:time.sleep(1);continue
        method,g,n=row['method'],row['g'],row['n'];start=time.time();f=folder(root,method,g)
        try:
            obs=read(f/'observations.json')
            if len(obs)!=n or digest(obs)!=row['observations_identity']:raise ValueError('Source observations changed')
            ad=f/'acquisition'/f'{n-49:03d}';post_only=False;forced=False
            (f/'job_logs').mkdir(exist_ok=True)
            for segment in range(2):
                cmd=[sys.executable,'-u','-B','-m','analyse.ac_open_graph_mix','job','--run-dir',str(root),'--method',method,'--group',str(g),'--n',str(n)]
                if post_only:cmd.append('--post-only')
                with (f/'job_logs'/f'n{n:03d}.log').open('a') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    write(root/f'gp_worker{index}.json',{'pid':os.getpid(),'numerical_pid':child.pid,'state':'GP_and_acquisition','method':method,'group':g,'n':n,'started':start,'worker':index})
                    while child.poll() is None:
                        sp=ad/'state.json'
                        if sp.exists():
                            state=read(sp);elapsed=state.get('segment_prior_seconds',0)+time.time()-state['active_unix']
                            cut=270 if state['phase']=='main' else 300
                            if elapsed>=cut:
                                os.killpg(child.pid,signal.SIGKILL);child.wait();forced=True
                                state['consumed_seconds']=min(300.,elapsed);state['phase']='post';state['events'].append({'event':'external_native_call_deadline','cut':cut,'elapsed':elapsed,'segment':segment})
                                state['prior_decoder_calls']=state['decoder_calls'];state['prior_decoder_hits']=state['decoder_hits'];state['recovery_unix']=time.time();write(sp,state)
                                post_only=True;break
                        time.sleep(.05)
                    code=child.wait()
                if code==0 and not forced:break
                if forced and segment==0 and read(ad/'state.json')['consumed_seconds']<300:forced=False;continue
                if forced or (ad/'state.json').exists():
                    partial_decision(root,method,g,n,'external_deadline' if forced else 'child_exception')
                    append(root/'ERRORS.jsonl',{'phase':'acquisition_partial_return','method':method,'group':g,'n':n,'child_exit':code,'external_deadline':forced,'at':now()});break
                raise RuntimeError('GP/job child failed before valid acquisition archive: '+str(code))
            saved=read(f/'decisions'/f'{n-49:03d}.json')
            if saved['observations_identity']!=row['observations_identity']:raise ValueError('Committed decision identity mismatch')
            with connect(root) as db:db.execute('UPDATE jobs SET status=?,finished=? WHERE method=? AND g=? AND n=?',('succeeded',time.time(),method,g,n))
            append(root/'GP_jobs.jsonl',{'method':method,'group':g,'n':n,'elapsed_seconds':time.time()-start,'fit_seconds':saved['fit_seconds'],
               'acquisition_seconds':saved['acquisition_seconds'],'worker':index,'at':now()})
        except BaseException:
            error=traceback.format_exc()
            with connect(root) as db:db.execute('UPDATE jobs SET status=?,finished=?,error=? WHERE method=? AND g=? AND n=?',('failed',time.time(),error,method,g,n))
            append(root/'ERRORS.jsonl',{'phase':'GP/acquisition','method':method,'group':g,'n':n,'error':error,'at':now()});raise
    write(root/f'gp_worker{index}.json',{'pid':os.getpid(),'state':'stopped','ended_at':now(),'worker':index})

def progress(root,method,g):
    f=folder(root,method,g);pool={r['candidate_id']:r for r in read(inputs(root,g)/'candidates.json')};seeds=read(inputs(root,g)/'seeds.json')
    path=f/'observations.json';obs=read(path) if path.exists() else [];initial=read(f/'initialization.json')['ids']
    for o in obs:pool[o['candidate_id']]={k:o[k] for k in ['candidate_id','configuration','architecture_id','latent']}
    for r in pool.values():
        if 'latent' not in r:r['latent']=r['x_raw'][:12] if method=='A' else None
    def add(cid,slot,kind,key,value,decision_hash=None):
        physical=task(root,key)
        obs.append({**{k:pool[cid][k] for k in ['configuration','architecture_id','latent']},'latent_source':'generated_or_original' if method=='A' else 'none_direct_graph','slot':slot,'candidate_id':cid,'task_id':key,'seed':value['seed'],'validation_accuracy':value['best_validation_accuracy'],
            'kind':kind,'cache_source':physical['source'],'physical_result_root':physical['result_root'],'decision_identity':decision_hash,
            'best_seen_validation':max([o['validation_accuracy'] for o in obs]+[value['best_validation_accuracy']]),'committed_at':now()})
        write(path,obs)
    while len(obs)<50:
        slot=len(obs);cid=initial[slot];key=enqueue(root,method,g,'search',slot,pool[cid],seeds['search_training_seed']);r=result(root,key)
        if r is None:return False
        add(cid,slot,'initialization',key,r)
    if len(obs)<300:
        n=len(obs)
        with connect(root) as db:
            job=db.execute('SELECT * FROM jobs WHERE method=? AND g=? AND n=?',(method,g,n)).fetchone()
            if job is None:
                db.execute('INSERT INTO jobs(method,g,n,observations_identity,status) VALUES(?,?,?,?,?)',(method,g,n,digest(obs),'queued'));return False
            if job['observations_identity']!=digest(obs):raise ValueError('GP job does not match trajectory observations')
            if job['status']=='failed':raise RuntimeError('GP job failed: '+str(job['error']))
            if job['status']!='succeeded':return False
        decision_path=f/'decisions'/f'{n-49:03d}.json';d=read(decision_path);cid=d['candidate_id']
        if d['method']!=method or d['observed_ids']!=[o['candidate_id'] for o in obs] or cid in {o['candidate_id'] for o in obs}:raise ValueError('Invalid method/group online decision')
        pool[cid]=d['selected_row']
        key=enqueue(root,method,g,'search',n,pool[cid],seeds['search_training_seed']);r=result(root,key)
        if r is None:return False
        add(cid,n,'qlogei',key,r,sha(decision_path))
        if len(obs)<300:return False
    if len(obs)!=300 or len(set(o['candidate_id'] for o in obs))!=300:raise ValueError('Search budget incomplete')
    seal(f/'search_complete.json',{'method':method,'group':g,'initial':50,'online':250,'total':300,'observations_identity':digest(obs)})
    top=sorted(obs,key=lambda o:(-o['validation_accuracy'],o['candidate_id']))[:10]
    topids=[o['candidate_id'] for o in top]
    seal(f/'top10.json',{'method':method,'group':g,'ids':topids,'validation_training_seeds':seeds['validation_training_seeds'],'test_used':False})
    values=[];ready=True
    for cid in topids:
        scores=[];keys=[]
        for i,s in enumerate(seeds['validation_training_seeds']):
            key=enqueue(root,method,g,'validation',f'{cid}:{i}',pool[cid],s);keys.append(key);r=result(root,key)
            if r is None:ready=False
            else:scores.append(r['best_validation_accuracy'])
        if len(scores)==30:values.append({'candidate_id':cid,'validation_scores':scores,'validation_mean':float(np.mean(scores)),
             'validation_sd':float(np.std(scores,ddof=1)),'seeds':seeds['validation_training_seeds'],'tasks':keys})
    if not ready:return False
    seal(f/'validation_reevaluation.json',values)
    chosen=min(values,key=lambda r:(-r['validation_mean'],r['candidate_id']));cid=chosen['candidate_id']
    if not (f/'champion.json').exists():seal(f/'champion.json',{'method':method,'group':g,'candidate_id':cid,
        'configuration':pool[cid]['configuration'],'validation_mean':chosen['validation_mean'],'validation_sd':chosen['validation_sd'],
        'validation_tasks':chosen['tasks'],'test_used_for_selection':False,'sealed_at':now()})
    if read(f/'champion.json')['candidate_id']!=cid:raise ValueError('Frozen champion changed')
    models=[]
    for i,s in enumerate(seeds['final_training_seeds']):
        key=enqueue(root,method,g,'final',i,pool[cid],s)
        if result(root,key) is not None:models.append(model_record(root,key))
    if len(models)!=30:return False
    if not (f/'final_models.json').exists():seal(f/'final_models.json',{'method':method,'group':g,'candidate_id':cid,
        'models':models,'champion_sha256':sha(f/'champion.json'),'sealed_at':now()})
    return True

def controller(root,recover=False):
    verify(root);lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():return read(root/'COMPLETE.json')
    initialize_jobs(root)
    for role in ('worker','gp_worker'):
        for i in range(2):
            p=root/f'{role}{i}.json'
            if p.exists() and pid_alive(read(p)['pid']):raise RuntimeError('Existing experiment worker alive: '+str(p))
    with connect(root) as db:
        rows=db.execute('SELECT id,status FROM tasks WHERE status IN (?,?)',('running','failed')).fetchall()
        jobs=db.execute('SELECT method,g,n,status FROM jobs WHERE status IN (?,?)',('running','failed')).fetchall()
        if (rows or jobs) and not recover:raise RuntimeError('Use --recover with unchanged protocol/seed')
        if rows or jobs:
            append(root/'recovery.jsonl',{'at':now(),'tasks':[dict(r) for r in rows],'jobs':[dict(r) for r in jobs],'exact_checkpoint_resume':True})
            db.execute('UPDATE tasks SET status=? WHERE source=? AND status IN (?,?)',('queued','OPEN-new','running','failed'))
            db.execute('UPDATE jobs SET status=? WHERE status IN (?,?)',('queued','running','failed'))
    if (root/'STOP').exists():(root/'STOP').unlink()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in gpu.splitlines():
        i,u,n,m,util=[x.strip() for x in line.split(',')];gpus.append({'index':int(i),'uuid':u,'name':n,'memory_used_MiB':int(m),'utilization_percent':int(util)})
    gpus=[x for x in gpus if x['memory_used_MiB']<500 and x['utilization_percent']==0][:2]
    if not gpus:raise RuntimeError('No physical GPU currently available; no other experiment will be stopped')
    write(root/'resources.json',{'gpus':gpus,'controller_pid':os.getpid(),'GP_processes':2,'GP_threads_each':1,'training_workers':len(gpus),'checked_at':now(),'python':sys.executable})
    for method,g in TRAJECTORIES:
        pool={r['candidate_id']:r for r in read(inputs(root,g)/'candidates.json')};s=read(inputs(root,g)/'seeds.json')['search_training_seed']
        for slot,cid in enumerate(read(folder(root,method,g)/'initialization.json')['ids']):enqueue(root,method,g,'search',slot,pool[cid],s)
    old=read(root/'STATUS.json');started=old.get('started_unix',time.time());processes=[];logs=[];last_eta=0;last_print=None
    write(root/'STATUS.json',{'state':'RUNNING','pid':os.getpid(),'started_unix':started,'updated_at':now(),'heartbeat':time.time(),'counts':counts(root)})
    try:
        for role in ('worker','gp_worker'):
            for i in ([x['index'] for x in gpus] if role=='worker' else range(2)):
                log=(root/f'{role}{i}.log').open('a');logs.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i) if role=='worker' else '',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                processes.append(subprocess.Popen([sys.executable,'-u','-B','-m','analyse.ac_open_graph_mix',role,'--run-dir',str(root),'--index',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT))
        while True:
            for p in processes:
                if p.poll() is not None:raise RuntimeError(f'Worker process {p.pid} exited unexpectedly: {p.returncode}')
            finished=[];states={}
            for method,g in TRAJECTORIES:
                if progress(root,method,g):finished.append((method,g))
                f=folder(root,method,g);n=len(read(f/'observations.json')) if (f/'observations.json').exists() else 0
                phase='final_sealed' if (f/'final_models.json').exists() else 'final_training' if (f/'champion.json').exists() else 'validation_reevaluation' if (f/'top10.json').exists() else 'online_search' if n>=50 else 'initialization'
                states[f'{method}{g}']={'observations':n,'phase':phase}
            status={'state':'RUNNING','pid':os.getpid(),'started_unix':started,'updated_at':now(),'heartbeat':time.time(),'trajectories':states,'counts':counts(root)}
            write(root/'STATUS.json',status)
            signature=digest({'trajectories':states,'counts':status['counts']})
            if signature!=last_print:print(json.dumps(status),flush=True);last_print=signature
            if time.time()-last_eta>=60:
                write(root/'HEARTBEAT.json',{'at':now(),'controller_pid':os.getpid(),'counts':status['counts'],'trajectories':states})
                from .report import update_eta
                update_eta(root);last_eta=time.time()
            if len(finished)==10:break
            time.sleep(1)
        (root/'STOP').write_text(now());exits=[p.wait() for p in processes];write(root/'worker_exit_codes.json',exits)
        if any(rc!=0 for rc in exits):raise RuntimeError('Worker exit codes not clean')
        manifest=[]
        for method,g in TRAJECTORIES:
            f=folder(root,method,g);m=read(f/'final_models.json')
            if len(m['models'])!=30:raise ValueError('Final logical models incomplete')
            manifest.append({'method':method,'group':g,'champion_sha256':sha(f/'champion.json'),'models_manifest_sha256':sha(f/'final_models.json')})
        if not (root/'AC_all_final_models_sealed.json').exists():seal(root/'AC_all_final_models_sealed.json',{'trajectories':manifest,'logical_models':300,'sealed_at':now(),'extension_after_Btest':True})
        status.update(state='FINAL_TEST_SCORING',updated_at=now(),heartbeat=time.time());write(root/'STATUS.json',status)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[0]['index']),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (root/'scoring.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u','-B','-m','analyse.ac_open_graph_mix','score','--run-dir',str(root)],env=env,stdout=log,stderr=subprocess.STDOUT)
            write(root/'scoring_process.json',{'pid':p.pid,'started_at':now(),'dtype':'float32','GP_isolated':True});rc=p.wait()
        write(root/'scoring_exit_code.json',rc)
        if rc:raise RuntimeError('Isolated scoring failed; see scoring.log')
        status.update(state='REPORT_AND_VERIFICATION',updated_at=now(),heartbeat=time.time());write(root/'STATUS.json',status)
        from .report import final_report,accept_complete,pass_report
        final_report(root);accepted=accept_complete(root)
        seal(root/'COMPLETE.json',{'state':'COMPLETE','accepted':accepted,'completed_at':now(),'methods':['A','C'],'B_read_only':True})
        status.update(state='COMPLETE',updated_at=now(),heartbeat=time.time(),counts=counts(root));write(root/'STATUS.json',status)
        pass_report(root,accepted,'pipeline immediate completion check')
    except BaseException:
        error=traceback.format_exc();append(root/'controller_failures.jsonl',{'at':now(),'error':error})
        (root/'STOP').write_text(now());write(root/'STATUS.json',{'state':'FAILED','pid':os.getpid(),'started_unix':started,'updated_at':now(),'error':error,'counts':counts(root)})
        write(root/'worker_exit_codes.json',[p.wait() for p in processes]);raise
    finally:
        for f in logs:f.close()
        lock.close()
