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


def check_task(root,task):
    folder=root/'fresh_training'/task['task_id'];result=read_json(folder/'result.json')
    boundary=read_json(folder/'data_access_boundary.json')
    expected=digest({'run':read_json(root/'run_identity.json')['identity'],'task':task,
        'dataset_context':boundary['dataset_context'],'device':boundary['device']})
    if result['identity']!=expected or result['state']!='COMPLETE':raise ValueError('Completed task identity/state mismatch')
    if set(result['checkpoint_hashes'])!={'resume.pt','best_model.pt'}:raise ValueError('Missing task checkpoints')
    for name,h in result['checkpoint_hashes'].items():
        if sha256(folder/name)!=h:raise ValueError('Changed task checkpoint: '+task['task_id'])
    if result['test_metric_computations'] or result['test_score_reads']:raise ValueError('Test boundary violated')
    return result


def paired_results(mapping,measurements):
    """Lookup by (configuration, training seed); repeated returns share values."""
    rows=[]
    for s in SEEDS:
        wi=mapping[f'{s}_W-I'];raw=mapping[f'{s}_RAW16']
        for r in (3,4,5):
            a=measurements[wi,r];b=measurements[raw,r]
            rows.append({'search_seed':s,'training_seed':r,'WI_candidate_id':wi,'RAW16_candidate_id':raw,
                'WI_validation_percent':100*a,'RAW16_validation_percent':100*b,'WI_minus_RAW16_pp':100*(a-b),
                'shared_measurement':wi==raw})
    return rows


def train_all(root,repo):
    set_status(root,'M3','RUNNING');tasks=read_json(root/'fresh_training_manifest.json')['tasks']
    results=[];errors=[];started=time.perf_counter()
    for task in tasks:
        folder=root/'fresh_training'/task['task_id'];folder.mkdir(parents=True,exist_ok=True)
        try:
            if not (folder/'result.json').exists():
                args=[sys.executable,'-B','-m','analyse.structured_kernel_followup','train-task','--run-dir',str(root),'--task-id',task['task_id']]
                with (folder/'worker.log').open('a') as log:
                    proc=subprocess.run(args,cwd=repo,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                if proc.returncode:raise RuntimeError(f'training worker exit {proc.returncode}; see {folder}/worker.log')
            result=check_task(root,task);results.append((task,result))
        except Exception as exc:
            message=(folder/'worker.log').read_text()[-6000:] if (folder/'worker.log').exists() else str(exc)
            state='RESOURCE_BLOCKED' if any(t in message.lower() for t in ('out of memory','cuda driver','no space left')) else 'INCOMPLETE'
            error={'task_id':task['task_id'],'state':state,'error':str(exc),'log_tail':message,'time':now()}
            append_jsonl(root/'errors.jsonl',error);errors.append(error)
        remaining=len(tasks)-len(results)-len(errors)
        timings=[r['elapsed_seconds'] for _,r in results]
        progress={'completed_tasks':len(results),'failed_tasks':len(errors),'pending_this_pass':remaining,
            'elapsed_coordinator_seconds':time.perf_counter()-started,
            'estimated_remaining_seconds':float(np.mean(timings)*remaining) if timings else None,
            'estimate_basis':'mean measured completed formal tasks; architecture variation remains',
            'last_task':task['task_id'],'time':now()}
        write_json(root/'M3_progress.json',progress);print(json.dumps(progress),flush=True)
        if results and not (root/'first_formal_task_benchmark.json').exists():
            t,r=results[0];write_json(root/'first_formal_task_benchmark.json',{'task_id':t['task_id'],
                'elapsed_seconds':r['elapsed_seconds'],'peak_gpu_memory_allocated_bytes':r['peak_gpu_memory_allocated_bytes'],
                'device':r['device'],'stopped_epoch':r['stopped_epoch'],'counted_in_15_formal_tasks':True,
                'initial_remaining14_estimate_seconds':14*r['elapsed_seconds']})
    rows=[];curves=[]
    for t,r in results:
        rows.append({'candidate_id':t['candidate_id'],'training_seed':t['seed'],'task_id':t['task_id'],
            **{k:r[k] for k in ('best_validation_accuracy','best_epoch','stopped_epoch','stop_reason','elapsed_seconds','device','peak_gpu_memory_allocated_bytes','identity')},
            'test_metric_computations':0,'test_score_reads':0})
        curves.extend({'candidate_id':t['candidate_id'],'training_seed':t['seed'],**e} for e in r['trajectory'])
    write_csv(root/'fresh_training_results.csv',rows);write_csv(root/'fresh_training_curves.csv',curves)
    attempts=[]
    for t in tasks:
        p=root/'fresh_training'/t['task_id']/'attempts.jsonl'
        if p.exists():attempts.extend({'task_id':t['task_id'],**json.loads(l)} for l in p.read_text().splitlines())
    write_json(root/'training_recovery_records.json',attempts)
    summary={'state':'COMPLETE' if len(results)==15 and not errors else ('RESOURCE_BLOCKED' if any(e['state']=='RESOURCE_BLOCKED' for e in errors) else 'INCOMPLETE'),
        'valid_tasks':len(results),'prescribed_tasks':15,'unique_configurations':len({t['candidate_id'] for t,r in results}),
        'training_attempts':len(attempts),'same_task_resumed_attempts':sum(a['same_task_retry'] for a in attempts),
        'total_measured_training_seconds':sum(r['elapsed_seconds'] for t,r in results),'errors':errors,
        'remaining_task_ids':[t['task_id'] for t in tasks if t['task_id'] not in {a['task_id'] for a,r in results}],
        'test_metric_computations':0,'test_score_reads':0,'independent_confirmation':False}
    if len(results)==15 and not errors:
        values={(t['candidate_id'],t['seed']):r['best_validation_accuracy'] for t,r in results}
        pairs=paired_results(read_json(root/'selected_mapping.json')['mapping'],values)
        write_csv(root/'fresh_return_pair_results.csv',pairs)
        configs=[]
        for cid in CONFIG_IDS:
            ys=np.array([values[cid,s] for s in (3,4,5)])
            configs.append({'candidate_id':cid,**{f'seed{s}_percent':100*values[cid,s] for s in (3,4,5)},
                'mean_percent':100*float(ys.mean()),'SE_pp':100*float(ys.std(ddof=1)/np.sqrt(3))})
        write_csv(root/'fresh_configuration_summary.csv',configs)
        ds=[{'training_seed':s,'D_pp':float(np.mean([p['WI_minus_RAW16_pp'] for p in pairs if p['training_seed']==s]))} for s in (3,4,5)]
        write_csv(root/'fresh_training_seed_summary.csv',ds)
        summary.update({'D_by_training_seed':ds,'D_mean_pp':float(np.mean([d['D_pp'] for d in ds])),
            'D_range_pp':[min(d['D_pp'] for d in ds),max(d['D_pp'] for d in ds)],'old_D_pp':.6,
            'unique_config_seed_measurements':15,'mapped_rows_are_not_independent_measurements':True})
    write_json(root/'M3_summary.json',summary);set_status(root,'M3',summary['state'],valid_tasks=len(results),errors=errors)
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,5,figsize=(16,3),sharey=True)
        for ax,cid in zip(axes,CONFIG_IDS):
            for s in (3,4,5):
                cr=[r for r in curves if r['candidate_id']==cid and r['training_seed']==s]
                if cr:ax.plot([r['epoch_1based'] for r in cr],[100*r['validation_accuracy'] for r in cr],label=f'seed{s}')
            ax.set(title=cid[:8],xlabel='Epoch');ax.legend(fontsize=7)
        axes[0].set_ylabel('Validation accuracy (%)');fig.tight_layout();fig.savefig(root/'M3_validation_curves.png',dpi=160);plt.close(fig)
    return summary


def module_files(root,name):
    names={'M1':['M1_summary.json','champion_trace.csv','kernel_mean_contributions.csv','training_point_contributions.csv','posterior_reconstruction_validation.json','M1_champion_diagnosis.png'],
        'M2':['M2_summary.json','random_trajectories.csv','random_paired_comparisons.csv','random_curves.csv','random_oracle_access_log.jsonl','M2_random_reference.png'],
        'M3':['M3_summary.json','fresh_training_results.csv','fresh_training_curves.csv','fresh_return_pair_results.csv','fresh_configuration_summary.csv','fresh_training_seed_summary.csv','training_recovery_records.json','M3_validation_curves.png'],
        'M4':['M4_summary.json','C_eligibility_manifest.json','historical_use_registry.json','historical_same_seed_audit.json']}[name]
    paths=[root/n for n in names]
    if name=='M2':paths+=sorted((root/'random_observations').glob('*.json'))
    if name=='M3':paths += [p for p in (root/'fresh_training').rglob('*') if p.is_file() and p.name in ('result.json','best_model.pt','resume.pt','data_access_boundary.json')]
    return {str(p.relative_to(root)):sha256(p) for p in paths}


def module_complete(root,name):
    p=root/f'{name}_completion.json'
    if not p.exists():return False
    seal=read_json(p)
    if seal['run_identity']!=read_json(root/'run_identity.json')['identity']:raise ValueError('Module completion identity changed')
    for f,h in seal['files'].items():
        if sha256(root/f)!=h:raise ValueError('Module result changed: '+f)
    return True


def work_counters(root):
    db=root/'random_oracle.sqlite';queries=calls=0
    if db.exists():
        con=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
        queries=con.execute('SELECT COUNT(*) FROM observations WHERE slot>=50').fetchone()[0]
        calls=con.execute('SELECT COUNT(*) FROM calls').fetchone()[0];con.close()
    attempts=sum(len(p.read_text().splitlines()) for p in (root/'fresh_training').glob('*/attempts.jsonl'))
    return {'GP_optimizations':0,'random_queries':queries,'oracle_calls':calls,'GNN_training_attempts':attempts}


def _run_locked(root,repo,resume=False):
    verify(root,repo);before=work_counters(root)
    summaries={p.name:sha256(p) for p in root.glob('M?_summary.json')}
    for name in ('M1','M2','M4','M3'):
        if module_complete(root,name):
            print(json.dumps({'module':name,'action':'verified completed result; skipped'}),flush=True);continue
        args=[sys.executable,'-B','-m','analyse.structured_kernel_followup','module','--run-dir',str(root),'--name',name]
        log=root/f'{name}_worker.log'
        with log.open('a') as f:result=subprocess.run(args,cwd=repo,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            set_status(root,name,'INCOMPLETE',worker_exit=result.returncode,log=str(log))
            append_jsonl(root/'errors.jsonl',{'module':name,'worker_exit':result.returncode,'log_tail':log.read_text()[-6000:],'time':now()})
        elif read_json(root/f'{name}_summary.json')['state']=='COMPLETE':
            write_json(root/f'{name}_completion.json',{'run_identity':read_json(root/'run_identity.json')['identity'],'files':module_files(root,name),'sealed_at':now()})
        print(json.dumps({'module':name,'state':read_json(root/'module_status.json')['modules'][name]['state']}),flush=True)
    after=work_counters(root)
    if resume:
        evidence={'before':before,'after':after,'additional':{k:after[k]-before[k] for k in before},
            'summaries_before':summaries,'summaries_after':{p.name:sha256(p) for p in root.glob('M?_summary.json')},'time':now()}
        evidence['pass']=all(v==0 for v in evidence['additional'].values()) and evidence['summaries_before']==evidence['summaries_after'] and len(summaries)==4
        write_json(root/'resume_verification.json',evidence)


def run(root,repo,resume=False):
    import fcntl
    with (root/'controller.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another followup controller is active; do not launch duplicate tasks')
        return _run_locked(root,repo,resume)
