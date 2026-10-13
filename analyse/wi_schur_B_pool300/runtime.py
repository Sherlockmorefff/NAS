"""One durable controller, two exclusive GPU workers, and automatic stage transitions."""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from .common import *
from .prepare import verify, bundle_data

def worker(root, index):
    import torch
    from deterministic_runtime import configure_torch_determinism
    from analyse.structured_kernel_followup.validation_train import train_validation_only
    configure_torch_determinism(torch);torch.set_num_threads(1)
    verify(root)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Worker must see exactly one CUDA GPU')
    device=torch.device('cuda:0');bundle,context=bundle_data();data=bundle.data.to(device)
    if context!=read(root/'protocol.json')['dataset_context']:raise ValueError('Dataset context changed')
    gpu=next(g for g in read(root/'resources.json')['gpus'] if g['index']==index)
    info={'pid':os.getpid(),'worker':index,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'logical_device':'cuda:0','physical_gpu_uuid':gpu['uuid'],'physical_gpu':index,'started_at':now()}
    write(root/f'worker{index}.json',dict(info,state='ready',heartbeat=time.time(),current_task=None))
    while not (root/'STOP').exists():
        write(root/f'worker{index}.json',dict(info,state='waiting',heartbeat=time.time(),current_task=None))
        with connect(root) as db:
            db.execute('BEGIN IMMEDIATE')
            task=db.execute("SELECT * FROM tasks WHERE status='queued' ORDER BY rowid LIMIT 1").fetchone()
            if task:
                started=time.time();attempt=task['attempts']+1
                db.execute("UPDATE tasks SET status='running',worker=?,pid=?,started=?,attempts=?,error=NULL WHERE id=?",(index,os.getpid(),started,attempt,task['id']))
                db.execute('INSERT INTO attempts(task,attempt,worker,pid,started) VALUES(?,?,?,?,?)',(task['id'],attempt,index,os.getpid(),started))
        if not task:time.sleep(1);continue
        key=task['id'];folder=root/'training'/key
        write(root/f'worker{index}.json',dict(info,state='running',heartbeat=time.time(),current_task=key,stage=task['stage'],attempt=attempt))
        try:
            value=train_validation_only(json.loads(task['config']),data,bundle.num_features,bundle.num_classes,
                task['seed'],device,folder,key,max_epochs=task['epochs'],patience=task['patience'])
            if value['test_metric_computations']!=0 or value['identity']!=key:raise ValueError('Invalid training result')
            ended=time.time();metadata=dict(info,task_id=key,candidate_id=task['cid'],stage=task['stage'],seed=task['seed'],
                actual_epochs=value['stopped_epoch'],training_seconds=value['elapsed_seconds'],task_wall_seconds=ended-started,
                checkpoint_resumed=(folder/'resume.pt').exists() and attempt>1,completed_at=now(),attempt=attempt)
            write(folder/'execution.json',metadata)
            with connect(root) as db:
                db.execute("UPDATE tasks SET status='succeeded',finished=?,elapsed=? WHERE id=?",(ended,value['elapsed_seconds'],key))
                db.execute('UPDATE attempts SET finished=?,exit_code=0 WHERE task=? AND attempt=?',(ended,key,attempt))
            print(json.dumps(metadata),flush=True)
            torch.cuda.empty_cache()
        except Exception as exc:
            error=traceback.format_exc();ended=time.time()
            append(root/'failures.jsonl',{'task_id':key,'attempt':attempt,'worker':index,'time':now(),'error':error,'spent_seconds':ended-started})
            with connect(root) as db:
                db.execute("UPDATE tasks SET status='failed',finished=?,error=? WHERE id=?",(ended,error,key))
                db.execute('UPDATE attempts SET finished=?,exit_code=1,error=? WHERE task=? AND attempt=?',(ended,error,key,attempt))
            write(root/f'worker{index}.json',dict(info,state='failed',current_task=key,error=str(exc),heartbeat=time.time()))
            raise
    write(root/f'worker{index}.json',dict(info,state='stopped',current_task=None,ended_at=now(),heartbeat=time.time()))

def progress_group(root,g):
    folder=root/'groups'/str(g);rows=read(folder/'candidates.json');index={r['candidate_id']:r for r in rows}
    seeds=read(folder/'seeds.json');initial=read(folder/'initialization.json')['ids']
    opath=folder/'observations.json';obs=read(opath) if opath.exists() else []
    while len(obs)<50:
        slot=len(obs);cid=initial[slot];key=enqueue(root,g,'search',slot,index[cid],seeds['search_training_seed']);r=result(root,key)
        if r is None:return False
        obs.append({'slot':slot,'candidate_id':cid,'task_id':key,'seed':r['seed'],'validation_accuracy':r['best_validation_accuracy'],
            'kind':'RBF-Schur-initial','best_seen_validation':max([o['validation_accuracy'] for o in obs]+[r['best_validation_accuracy']]),'committed_at':r['completed_at']})
        write(opath,obs)
    if len(obs)<300:
        n=len(obs);step=n-49;dpath=folder/'decisions'/f'{step:03d}.json'
        if not dpath.exists():
            write(root/'controller_activity.json',{'state':'GP_and_qLogEI','group':g,'observed_n':n,'started':time.time(),'pid':os.getpid()})
            from .search import decision
            d=decision(root,g,obs)
        else:
            d=read(dpath)
            if d['observations_identity']!=digest(obs):raise ValueError('Saved online decision source changed')
        cid=d['candidate_id']
        if cid in {o['candidate_id'] for o in obs}:raise ValueError('Online decision requested an already seen configuration')
        key=enqueue(root,g,'search',n,index[cid],seeds['search_training_seed']);r=result(root,key)
        if r is None:return False
        obs.append({'slot':n,'step':step,'candidate_id':cid,'task_id':key,'seed':r['seed'],'validation_accuracy':r['best_validation_accuracy'],
             'kind':'qlogei','decision_identity':sha(dpath),'best_seen_validation':max([o['validation_accuracy'] for o in obs]+[r['best_validation_accuracy']]),'committed_at':r['completed_at']})
        write(opath,obs)
        if len(obs)<300:return False
    if len(obs)!=300 or len(set(o['candidate_id'] for o in obs))!=300:raise ValueError('Invalid search query budget')
    seal(folder/'search_complete.json',{'group':g,'initial':50,'online':250,'total':300,'observations_identity':digest(obs),
        'decisions':250,'completed_at':obs[-1]['committed_at']})
    top=sorted(obs,key=lambda o:(-o['validation_accuracy'],o['candidate_id']))[:10]
    top_manifest={'group':g,'ids':[o['candidate_id'] for o in top],'selection':'search measured validation only',
        'training_seeds':seeds['validation_training_seeds'],'new_node_split':False,'test_reads':0}
    seal(folder/'top10.json',top_manifest);validation=[];ready=True
    for cid in top_manifest['ids']:
        values=[];keys=[]
        for i,s in enumerate(seeds['validation_training_seeds']):
            key=enqueue(root,g,'validation',f'{cid}:{i}',index[cid],s);keys.append(key);r=result(root,key)
            if r is None:ready=False
            else:values.append(r['best_validation_accuracy'])
        if len(values)==30:
            validation.append({'candidate_id':cid,'seeds':seeds['validation_training_seeds'],'validation_scores':values,
                'validation_mean':float(np.mean(values)),'validation_sd':float(np.std(values,ddof=1)),'tasks':keys})
    if not ready:return False
    seal(folder/'validation_reevaluation.json',validation)
    chosen=min(validation,key=lambda r:(-r['validation_mean'],r['candidate_id']));cid=chosen['candidate_id']
    cpath=folder/'champion.json'
    if not cpath.exists():seal(cpath,{'group':g,'candidate_id':cid,'configuration':index[cid]['configuration'],
        'validation_mean':chosen['validation_mean'],'validation_sd':chosen['validation_sd'],'validation_tasks':chosen['tasks'],
        'selected_by':'30 new training seed validation mean, first candidate ID for tie','test_reads':0,'sealed_at':now()})
    elif read(cpath)['candidate_id']!=cid:raise ValueError('Frozen champion differs')
    final=[]
    for i,s in enumerate(seeds['final_training_seeds']):
        key=enqueue(root,g,'final',i,index[cid],s);r=result(root,key)
        if r is None:continue
        final.append({'seed':s,'task_id':key,'checkpoint':str(root/'training'/key/'best_model.pt'),
            'checkpoint_sha256':sha(root/'training'/key/'best_model.pt'),'validation_accuracy':r['best_validation_accuracy']})
    if len(final)!=30:return False
    fpath=folder/'final_models.json'
    if not fpath.exists():seal(fpath,{'group':g,'candidate_id':cid,'champion_sha256':sha(cpath),'models':final,
        'test_score_computations_before_seal':0,'sealed_at':now()})
    return True

def score_final(root):
    import torch
    import eval_utils as original
    from deterministic_runtime import configure_torch_determinism
    configure_torch_determinism(torch);torch.set_num_threads(1)
    path=root/'all_final_models_sealed.json'
    if not path.exists():
        groups=[]
        for g in range(5):
            folder=root/'groups'/str(g);manifest=read(folder/'final_models.json')
            if len(manifest['models'])!=30:raise ValueError('Cannot release test without all final models')
            for m in manifest['models']:
                if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('Final checkpoint changed before test')
            groups.append({'group':g,'champion_sha256':sha(folder/'champion.json'),'models_manifest_sha256':sha(folder/'final_models.json')})
        seal(path,{'groups':groups,'model_count':150,'sealed_at':now(),'test_computations_before_barrier':0})
    barrier=read(path);bundle,context=bundle_data();device=torch.device('cuda:0');data=bundle.data.to(device)
    scores=[];cache={};score_root=root/'test_scores';score_root.mkdir(exist_ok=True)
    for g in range(5):
        folder=root/'groups'/str(g);champion=read(folder/'champion.json');models=read(folder/'final_models.json')
        expected=barrier['groups'][g]
        if sha(folder/'champion.json')!=expected['champion_sha256'] or sha(folder/'final_models.json')!=expected['models_manifest_sha256']:raise ValueError('Test barrier artifact changed')
        config=champion['configuration']
        for m in models['models']:
            h=m['checkpoint_sha256'];sp=score_root/(h+'.json')
            if sha(m['checkpoint'])!=h:raise ValueError('Final checkpoint changed')
            if sp.exists():r=read(sp)
            elif h in cache:r=cache[h]
            else:
                started=time.perf_counter()
                with original.isolated_rng(m['seed'],device):
                    model=original.DynamicGNN(config,bundle.num_features,bundle.num_classes,dropout=config['dropout'],hidden_dim=config['hidden_dim'],
                        gcnii_alpha=config['gcnii_alpha'],gcnii_theta=config['gcnii_theta'],gat_heads=config['gat_heads'],
                        sage_aggr=config['sage_aggr'],gin_eps=config['gin_eps'],component_variant='baseline',activation_variant='relu').to(device)
                checkpoint=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)
                if checkpoint['identity']!=m['task_id']:raise ValueError('Final model identity mismatch')
                model.load_state_dict(checkpoint['model']);model.eval()
                with torch.no_grad():
                    predicted=model(data.x,data.edge_index).argmax(dim=1)
                    value=float(predicted[data.test_mask].eq(data.y[data.test_mask]).sum().item()/int(data.test_mask.sum()))
                r={'checkpoint_sha256':h,'test_accuracy':value,'barrier_sha256':sha(path),'computed_at':now(),
                    'elapsed_seconds':time.perf_counter()-started,'scoring_device':str(device),'training_updates':0}
                seal(sp,r);cache[h]=r;del model;torch.cuda.empty_cache()
            if r['barrier_sha256']!=sha(path) or not np.isfinite(r['test_accuracy']):raise ValueError('Invalid test score commit')
            scores.append({'group':g,'candidate_id':champion['candidate_id'],**m,**r})
    seal(root/'final_scores.json',scores);return scores

def controller(root,recover=False):
    verify(root);lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'COMPLETE.json').exists():return read(root/'COMPLETE.json')
    old=read(root/'status.json') if (root/'status.json').exists() else {}
    for i in range(2):
        wp=root/f'worker{i}.json'
        if wp.exists() and pid_alive(read(wp)['pid']):raise RuntimeError('Existing worker is alive; do not double launch')
    with connect(root) as db:
        unfinished=db.execute("SELECT id,status FROM tasks WHERE status IN ('running','failed')").fetchall()
        if unfinished and not recover:raise RuntimeError('Use --recover to resume interrupted/failed exact tasks')
        if unfinished:
            append(root/'recovery.jsonl',{'at':now(),'tasks':[dict(r) for r in unfinished],'rule':'same task ID/seed; exact saved epoch; attempts retained'})
            db.execute("UPDATE tasks SET status='queued' WHERE status IN ('running','failed')")
    if (root/'STOP').exists():(root/'STOP').unlink()
    gpus=[]
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in out.splitlines():
        i,u,n,m,util=[v.strip() for v in line.split(',')];gpus.append({'index':int(i),'uuid':u,'name':n,'memory_used_MiB':int(m),'utilization_percent':int(util)})
    if len(gpus)<2 or any(g['memory_used_MiB']>2000 for g in gpus[:2]):raise RuntimeError('Two GPU resources not currently available')
    resources={'gpus':gpus,'controller_pid':os.getpid(),'python':sys.executable,'checked_at':now(),'gp_device':'cpu','gp_threads':1,'workers':2}
    write(root/'resources.json',resources)
    for g in range(5):
        rows={r['candidate_id']:r for r in read(root/'groups'/str(g)/'candidates.json')};seeds=read(root/'groups'/str(g)/'seeds.json')
        for slot,cid in enumerate(read(root/'groups'/str(g)/'initialization.json')['ids']):enqueue(root,g,'search',slot,rows[cid],seeds['search_training_seed'])
    processes=[];logs=[];started=old.get('started_unix',time.time());last_eta=0;last_signature=None
    write(root/'status.json',{'state':'RUNNING','pid':os.getpid(),'started_unix':started,'updated_at':now(),**counts(root)})
    try:
        for i in range(2):
            log=(root/f'worker{i}.log').open('a');logs.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            p=subprocess.Popen([sys.executable,'-u','-B','-m','analyse.wi_schur_B_pool300','worker','--run-dir',str(root),'--index',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append(p)
        while True:
            for i,p in enumerate(processes):
                if p.poll() is not None:raise RuntimeError(f'GPU worker {i} exited unexpectedly with code {p.returncode}')
            completed=[]
            for g in range(5):
                if progress_group(root,g):completed.append(g)
            cc=counts(root);groups={}
            for g in range(5):
                f=root/'groups'/str(g);n=len(read(f/'observations.json')) if (f/'observations.json').exists() else 0
                phase='final_sealed' if (f/'final_models.json').exists() else 'final_training' if (f/'champion.json').exists() else 'validation_reevaluation' if (f/'top10.json').exists() else 'online_search' if n>=50 else 'initialization'
                groups[str(g)]={'observations':n,'phase':phase}
            status={'state':'RUNNING','pid':os.getpid(),'started_unix':started,'updated_at':now(),'heartbeat':time.time(),'groups':groups,**cc}
            write(root/'status.json',status);write(root/'controller_activity.json',{'state':'queue_and_progress','pid':os.getpid(),'started':time.time()})
            signature=digest({'groups':groups,'counts':cc})
            if signature!=last_signature:
                print(json.dumps(status),flush=True);last_signature=signature
            if cc['actual'].get('succeeded',0)>=10 and time.time()-last_eta>60:
                from .report import update_eta
                update_eta(root);last_eta=time.time()
            if len(completed)==5:break
            time.sleep(1)
        (root/'STOP').write_text(now())
        exits=[p.wait() for p in processes];write(root/'worker_exit_codes.json',exits)
        if exits!=[0,0]:raise RuntimeError('Worker exit codes failed')
        write(root/'status.json',dict(status,state='FINAL_TEST_SCORING',updated_at=now(),heartbeat=time.time()))
        score_final(root)
        from .report import final_report,accept_complete
        final_report(root);accepted=accept_complete(root)
        seal(root/'COMPLETE.json',{'state':'COMPLETE','accepted':accepted,'completed_at':now(),'only_B':True})
        write(root/'status.json',dict(status,state='COMPLETE',updated_at=now(),heartbeat=time.time(),**counts(root)))
    except BaseException:
        error=traceback.format_exc();append(root/'controller_failures.jsonl',{'at':now(),'error':error})
        (root/'STOP').write_text(now())
        write(root/'status.json',{'state':'FAILED','pid':os.getpid(),'updated_at':now(),'error':error,'started_unix':started,**counts(root)})
        exits=[p.wait() for p in processes];write(root/'worker_exit_codes.json',exits)
        raise
    finally:
        for log in logs:log.close()
        lock.close()
