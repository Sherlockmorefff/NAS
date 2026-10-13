"""Persistent server controller, exclusive queue writer and scheduled acceptance."""
import signal
import subprocess
import time
from fractions import Fraction
from .common import *
from .scheduler import Queue,pilot_tasks
from .monitor import estimate,due,recheck

def add_tasks(root, queue, tasks):
    for task in tasks:
        seal(root/'task_definitions'/(task['task_id']+'.json'),task)
        queue.add(task)

def refresh_manifest(root, queue):
    groups = {str(g):sha(root/'groups'/f'g{g:02d}.json') for g in range(20) if (root/'groups'/f'g{g:02d}.json').exists()}
    write(root/'TASK_MANIFEST.json',{'tasks':[json.loads(r['payload']) for r in queue.rows()],
        'immutable_group_manifests':groups,'pending_dependencies':20-len(groups),
        'planned_roles':{'rerank_seed':600,'rerank_mask':600,'evaluation_logical':600,'evaluation_unique':[200,600]},
        'logical_budget':13380,'query_cache_tasks_verified':528,
        'physical_task_count':len(queue.rows()),'time':now()})

def discover(root, queue):
    seeds = frozen_read(root/'SEEDS.json')['groups']; configs = frozen_read(root/'CONFIGURATIONS.json')['by_id']
    changed = False
    known_ids = {r['id'] for r in queue.rows()}
    for group in range(20):
        gp = root/'groups'/f'g{group:02d}.json'; ready = root/'search_ready'/f'g{group:02d}_SM.json'
        if ready.exists() and not gp.exists():
            r = read(ready); obs = r['observations']; top = [o['candidate_id'] for o in ordered_observations(obs,configs)[:5]]
            assert len(obs)==61 and len(set(o['candidate_id'] for o in obs))==61 and top==r['top5']
            tasks=[]
            for cid in top:
                for seed in seeds[str(group)]['rerank']:
                    tasks.append(physical_task(root,group,'rerank_seed',cid,seed,'original'))
                for split in ['split1','split2']:
                    for seed in seeds[str(group)]['M_each_split']:
                        tasks.append(physical_task(root,group,'rerank_mask',cid,seed,split))
            assert len(tasks)==60 and len({t['task_id'] for t in tasks})==60
            seal(gp,{'group':group,'top5':top,'tasks':tasks,'search_ready_sha256':sha(ready),
                'own_observations':61,'S_M_same_trajectory_and_top5':True})
            add_tasks(root,queue,tasks); changed=True
        if gp.exists():
            missing = [t for t in read(gp)['tasks'] if t['task_id'] not in known_ids]
            if missing:
                add_tasks(root,queue,missing); changed=True
        up = root/'search_ready'/f'g{group:02d}_U.json'
        ur = root/'returns'/f'g{group:02d}_U.json'
        if up.exists() and not ur.exists():
            r=read(up); cid=r['candidate_id']
            seal(ur,{'group':group,'rule':'U','candidate_id':cid,**configs[cid],
                'score_exact':{'numerator':r['query_correct_sum'],'denominator':r['query_denominator']},
                'source_sha256':sha(up),'evaluation_seeds':seeds[str(group)]['evaluation'],
                'new_evaluation_results_used_for_selection':False,'sealed_at':now()})
            changed=True
        if gp.exists():
            tasks=read(gp)['tasks']; states={r['id']:r['state'] for r in queue.rows()}
            for rule,role in [('S','rerank_seed'),('M','rerank_mask')]:
                rp=root/'returns'/f'g{group:02d}_{rule}.json'; relevant=[t for t in tasks if t['role']==role]
                if rp.exists() or not all(states.get(t['task_id'])=='COMPLETE' for t in relevant):
                    continue
                scores={}; evidence={}
                for cid in read(gp)['top5']:
                    ct=[t for t in relevant if t['candidate_id']==cid]
                    rows=[]
                    for task in ct:
                        binding=read(root/'training'/task['task_id']/'BINDING.json')
                        rows.append({'correct':binding['correct'],'denominator':binding['denominator'],'split':task['split']})
                    if rule=='S':
                        value=exact_mean(rows)
                    else:
                        value=(exact_mean([x for x in rows if x['split']=='split1'])+exact_mean([x for x in rows if x['split']=='split2']))/2
                    scores[cid]=value
                    evidence[cid]={'score_exact':{'numerator':value.numerator,'denominator':value.denominator},
                        'tasks':[t['task_id'] for t in ct],
                        'binding_hashes':{t['task_id']:sha(root/'training'/t['task_id']/'BINDING.json') for t in ct}}
                cid=min(scores,key=lambda c:(-scores[c],configs[c]['normalized_config_id']))
                seal(rp,{'group':group,'rule':rule,'candidate_id':cid,**configs[cid],
                    'score_exact':evidence[cid]['score_exact'],'all_top5_rerank':evidence,
                    'group_manifest_sha256':sha(gp),'evaluation_seeds':seeds[str(group)]['evaluation'],
                    'new_evaluation_results_used_for_selection':False,'original_query_scores_blended':False,'sealed_at':now()})
                changed=True
        # Preserve sealing order for every rule even when evaluation is deduplicated.
        # All three group returns precede that group's first evaluation initialization.
        group_returns_ready = all((root/'returns'/f'g{group:02d}_{r}.json').exists() for r in RULES)
        for rule in RULES if group_returns_ready else ():
            rp=root/'returns'/f'g{group:02d}_{rule}.json'
            if rp.exists():
                r=read(rp)
                for seed in seeds[str(group)]['evaluation']:
                    task=physical_task(root,group,'final_validation',r['candidate_id'],seed,'original')
                    if not (root/'task_definitions'/(task['task_id']+'.json')).exists():
                        changed=True
                    if task['task_id'] not in known_ids:
                        add_tasks(root,queue,[task]); known_ids.add(task['task_id'])
    if changed:
        refresh_manifest(root,queue)
    return changed

def associated_rules(root, task):
    if task['role']!='final_validation':
        return []
    return [r for r in RULES if (root/'returns'/f'g{task["group"]:02d}_{r}.json').exists()
        and read(root/'returns'/f'g{task["group"]:02d}_{r}.json')['candidate_id']==task['candidate_id']]

def complete_binding(root, row):
    folder=root/'training'/row['id']; task=json.loads(row['payload'])
    binding=read(folder/'BINDING.json'); result=read(folder/'result.json')
    assert binding['task']==task and binding['identity']==result['identity']
    assert binding['identity']==execution_identity(root,task,read(root/'GPUS.json')['workers'][str(row['worker'])])
    assert binding['result_sha256']==sha(folder/'result.json')
    verify(folder,result['files'])
    assert binding['denominator']==500 and binding['correct']/500==result['selection']
    assert result['test_forwards']==0 and not binding['old_test_or_H_labels_in_view']
    if (folder/'planned_pause.json').exists():
        pause=read(folder/'planned_pause.json')
        assert digest(result['trajectory'][:pause['epoch']])==pause['trajectory_hash']
        events=[json.loads(x) for x in (folder/'epoch_events.jsonl').read_text().splitlines()]
        assert [e['epoch'] for e in events]==list(range(1,result['stopped_epoch']+1))
    return result,binding

def ingest(root, queue):
    for row in queue.rows('RUNNING'):
        event_path=root/'worker_events'/(row['token']+'.json')
        if event_path.exists():
            event=read(event_path)
            if event.get('completed'):
                complete_binding(root,row)
            queue.finish(event)
            if not event.get('completed'):
                raise RuntimeError('Formal training failed; no free retry: '+row['id'])

def recover(root, queue):
    import torch
    for row in queue.rows('RUNNING'):
        event_path=root/'worker_events'/(row['token']+'.json')
        if event_path.exists():
            event=read(event_path)
            if event.get('completed'):
                complete_binding(root,row); queue.finish(event); continue
            if not event.get('stopped'):
                raise RuntimeError('Recorded failed grant; automatic reinitialization forbidden')
            # An explicit stop is resumable only through the same complete state
            # validation below. Retain the stopped grant as historical evidence.
        folder=root/'training'/row['id']
        if (folder/'result.json').exists() and (folder/'BINDING.json').exists():
            complete_binding(root,row)
            queue.finish({'token':row['token'],'task_id':row['id'],'worker':row['worker'],
                'completed':True,'exit_code':0,'recovered_completion':True,'wall_seconds':read(folder/'result.json')['worker_seconds']})
        elif (folder/'latest.pt').exists():
            saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
            expected=execution_identity(root,json.loads(row['payload']),read(root/'GPUS.json')['workers'][str(row['worker'])])
            assert saved['identity']==expected and all(k in saved for k in ['model','optimizer','scheduler','rng','epoch','best_state','best_val','no_improve'])
            queue.requeue_resumable(row['id'])
            log(root/'RECOVERY.jsonl',{'time':now(),'task_id':row['id'],'same_gpu_uuid':row['gpu'],
                'resumed_epoch':saved['epoch'],'new_initializations':0})
        else:
            raise RuntimeError('Interrupted grant lacks complete latest state; cannot restart within frozen budget: '+row['id'])

def progress(root, queue, phase):
    rows=queue.rows(); counts={s:sum(r['state']==s for r in rows) for s in ['READY','RUNNING','COMPLETE','FAILED']}
    completed=[r for r in rows if r['state']=='COMPLETE']
    initializations=sum((root/'training'/r['id']/'INITIALIZATION.json').exists() for r in rows)
    timings=[read(root/'training'/r['id']/'result.json') for r in completed]
    value={'state':'RUNNING','phase':phase,'time':now(),'counts':counts,'physical_tasks_materialized':len(rows),
        'formal_initializations':initializations,'initialization_reservations':sum(r['reserved'] for r in rows),
        'worker_seconds_completed':sum(r['worker_seconds'] for r in timings),
        'gpu_event_seconds_completed':sum(r['gpu_event_seconds'] for r in timings),
        'groups_materialized':len(list((root/'groups').glob('g*.json'))),
        'returns_sealed':len(list((root/'returns').glob('g*.json'))),'old_test_access':0,'H_access':0,
        'search_progress':read(root/'SEARCH_PROGRESS.json') if (root/'SEARCH_PROGRESS.json').exists() else {},
        'ETA_file':str(root/'ETA.json') if (root/'ETA.json').exists() else None}
    write(root/'STATUS.json',value)
    ledger=[]
    states={r['id']:dict(r) for r in rows}
    for group in range(20):
        for rule in RULES:
            obs_path=root/'trajectories'/f'g{group:02d}'/('U_observations.json' if rule=='U' else 'SM_observations.json')
            if obs_path.exists():
                queries=len(read(obs_path))
            else:
                up=root/'trajectories'/f'g{group:02d}'/'U_observations.json'
                queries=min(60,len(read(up))) if rule!='U' and up.exists() else 0
            rerank=[]
            gp=root/'groups'/f'g{group:02d}.json'
            if gp.exists() and rule!='U':
                role='rerank_seed' if rule=='S' else 'rerank_mask'
                rerank=[t for t in read(gp)['tasks'] if t['role']==role]
            ev=[]; rp=root/'returns'/f'g{group:02d}_{rule}.json'
            if rp.exists():
                ret=read(rp)
                ev=[physical_task(root,group,'final_validation',ret['candidate_id'],s,'original') for s in ret['evaluation_seeds']]
            counted=lambda tasks:sum((root/'training'/t['task_id']/'INITIALIZATION.json').exists() for t in tasks)
            done=lambda tasks:sum(states.get(t['task_id'],{}).get('state')=='COMPLETE' for t in tasks)
            consumed=queries*3+counted(rerank)+counted(ev)
            assert consumed<=223
            ledger.append({'group':group,'rule':rule,'logical_limit':223,'query_calls_completed':queries,
                'query_train_cache_hits':queries*3,'rerank_initializations':counted(rerank),'rerank_completed':done(rerank),
                'evaluation_initializations':counted(ev),'evaluation_completed':done(ev),'logical_consumed':consumed,
                'logical_complete':consumed==223 and done(rerank)==(0 if rule=='U' else 30) and done(ev)==10,
                'physical_costs_shared':'see physical_task_ledger.csv; duplicate return evaluations charged logically to each rule'})
    csvwrite(root/'BUDGET_LEDGER.csv',ledger)
    csvwrite(root/'physical_task_ledger.csv',[{'task_id':r['id'],**json.loads(r['payload']),'state':r['state'],
        'worker':r['worker'],'GPU_UUID':r['gpu'],'grants':r['grants'],'initialization_reserved':r['reserved'],
        'initialized':(root/'training'/r['id']/'INITIALIZATION.json').exists(),
        'worker_wall_seconds':json.loads(r['event']).get('wall_seconds') if r['event'] else None} for r in rows])
    return value

def run(root, repo, resume=False):
    lock=exclusive(root/'controller.lock'); verify_run(root,repo)
    queue=Queue(root/'task_queue.sqlite')
    if read(root/'STATUS.json')['state']=='COMPLETE':
        from .analysis import verify_complete
        verify_complete(root,repo); return
    if resume:
        recover(root,queue)
    elif queue.rows('RUNNING'):
        raise RuntimeError('Existing running grants require explicit resume')
    search_proc=None; search_output=None; started=time.time(); last_status=0; last_estimate=0
    write(root/'CONTROLLER_PROCESS.json',{'pid':os.getpid(),'started':times(started),'run_dir':str(root),'resume':resume})
    if not (root/'RUN_STARTED.json').exists():
        write(root/'RUN_STARTED.json',{'unix':started,**times(started)})
    write(root/'STATUS.json',{'state':'RUNNING','phase':'SEARCH_AND_DEPENDENCIES','time':now()})
    try:
        if not (root/'SEARCH_COMPLETE.json').exists():
            search_output=(root/'SEARCH.log').open('a')
            search_proc=subprocess.Popen([sys.executable,'-u','-B','-m',MODULE,'search','--run-dir',str(root)],
                cwd=repo,stdout=search_output,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'SEARCH_PROCESS.json',{'pid':search_proc.pid,'run_dir':str(root),'started':now()})
        while True:
            if (root/'STOP_REQUEST.json').exists():
                write(root/'STATUS.json',{'state':'STOPPED','time':now(),'reason':'user stop; complete latest states retained'})
                return
            if search_proc and search_proc.poll() is not None and search_proc.returncode!=0:
                raise RuntimeError('Sequential GP/search process failed: '+str(search_proc.returncode))
            ingest(root,queue); discover(root,queue)
            if not (root/'PILOT_MANIFEST.json').exists() and len(queue.rows())>=6:
                tasks=pilot_tasks([json.loads(r['payload']) for r in queue.rows()])
                if tasks:
                    seal(root/'PILOT_MANIFEST.json',{'task_ids':[t['task_id'] for t in tasks],
                        'tasks':tasks,'choice':'three lowest parameter-count legal S tasks and three largest legal M tasks; config ID/seed/mask ties; before any new results',
                        'worker_task_ids':{'1':[t['task_id'] for t in tasks[::2]],'2':[t['task_id'] for t in tasks[1::2]]},
                        'distinct_parameter_counts':sorted({t['parameter_count'] for t in tasks}),
                        'frozen_at':now(),'contains_no_extra_training':True})
            pilot=read(root/'PILOT_MANIFEST.json') if (root/'PILOT_MANIFEST.json').exists() else None
            rowmap={r['id']:r for r in queue.rows()}
            pilot_done=bool(pilot) and all(rowmap[t]['state']=='COMPLETE' for t in pilot['task_ids'])
            if pilot_done and not (root/'ETA.json').exists():
                eta=estimate(root,queue)
                print(json.dumps({'pilot_complete':True,'ETA':eta['estimated_completion_range'],'first_check':eta['first_check']}),flush=True)
                last_estimate=time.time()
            if pilot:
                for number in (1,2):
                    pulse=root/'workers'/f'worker-{number}.json'
                    if not pulse.exists():
                        continue
                    worker=read(pulse)
                    if worker['state']!='IDLE' or time.time()-worker.get('unix',0)>20:
                        continue
                    allowed=None if pilot_done else set(pilot['worker_task_ids'][str(number)])
                    gpu=frozen_read(root/'GPUS.json')['workers'][str(number)]
                    assignment=queue.claim(number,gpu,allowed)
                    if assignment:
                        task=assignment['task']; assignment['rules']=associated_rules(root,task)
                        assignment['exercise_interruption']=task['task_id']==pilot['task_ids'][0]
                        write(root/'assignments'/f'worker-{number}.json',assignment)
                        log(root/'GRANTS.jsonl',{'task_id':task['task_id'],'worker':number,'GPU_UUID':gpu['uuid'],
                            'token':assignment['token'],'time':now(),'physical_initialization_reservation':1})
            stamp=time.time()
            if stamp-last_status>=30:
                p=progress(root,queue,'FORMAL_TRAINING' if pilot_done else 'SIX_TASK_TIMING')
                write(root/'HEARTBEAT.json',{'pid':os.getpid(),'unix':stamp,'time':now(),'counts':p['counts']})
                print(json.dumps({'time':now(),'counts':p['counts'],'groups':p['groups_materialized'],'returns':p['returns_sealed']}),flush=True)
                for row in queue.rows('RUNNING'):
                    pulse=root/'workers'/f'worker-{row["worker"]}.json'
                    if not pulse.exists() or stamp-read(pulse).get('unix',0)>60:
                        raise RuntimeError('Worker heartbeat disappeared; explicit complete-state resume required: '+row['id'])
                    folder=root/'training'/row['id']; latest=folder/'latest.pt'
                    if latest.exists() and stamp-latest.stat().st_mtime>900:
                        child=read(pulse).get('child_pid')
                        gpu_status=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
                        child_status=subprocess.run(['ps','-p',str(child),'-o','pid,ppid,stat,etime,time,args'],
                            capture_output=True,text=True,timeout=15)
                        log(root/'STALL_DIAGNOSTICS.jsonl',{'time':now(),'task':row['id'],'worker':row['worker'],
                            'latest_age_seconds':stamp-latest.stat().st_mtime,'GPU_status':gpu_status.stdout,
                            'child_status':child_status.stdout,'action':'explicit failure; complete state retained; no blind reinitialization'})
                        raise RuntimeError('No epoch checkpoint for 900 seconds; diagnosed stalled task: '+row['id'])
                last_status=stamp
            if pilot_done and stamp-last_estimate>=900:
                estimate(root,queue); last_estimate=stamp
            if (root/'ETA.json').exists():
                eta=read(root/'ETA.json')
                if due(eta,stamp):
                    rows=queue.rows()
                    ready_all=(root/'SEARCH_COMPLETE.json').exists() and len(list((root/'returns').glob('g*.json')))==60
                    complete_all=ready_all and all(r['state']=='COMPLETE' for r in rows)
                    log(root/'ACCEPTANCE_CHECKS.jsonl',{'time':now(),'unix':stamp,'scheduled':times(eta['next_check_unix']),
                        'all_dependencies_complete':ready_all,'all_tasks_complete':complete_all,
                        'remaining_materialized':sum(r['state']!='COMPLETE' for r in rows),
                        'unmaterialized_groups':20-len(list((root/'groups').glob('g*.json')))})
                    if complete_all:
                        progress(root,queue,'FINAL_ACCEPTANCE')
                        from .analysis import finalize
                        finalize(root,repo,queue)
                        print(json.dumps({'state':'COMPLETE','report':str(root/'DEVELOPMENT_RESULTS.md')}),flush=True)
                        return
                    estimate(root,queue); eta=recheck(read(root/'ETA.json'),stamp); write(root/'ETA.json',eta)
            time.sleep(2)
    except BaseException as exc:
        progress(root,queue,'FAILED_ACCEPTANCE_OR_TASK')
        write(root/'FAILURE.json',{'type':type(exc).__name__,'reason':str(exc),'time':now(),
            'automatic_from_initialization_retries':0,'old_test_or_H_opened':False})
        write(root/'STATUS.json',{'state':'FAILED','time':now(),'reason':str(exc)})
        from .analysis import failed_report
        failed_report(root,queue,str(exc))
        raise
    finally:
        if search_proc and search_proc.poll() is None:
            os.killpg(search_proc.pid,signal.SIGTERM); search_proc.wait(timeout=30)
        if search_output:
            search_output.close()
