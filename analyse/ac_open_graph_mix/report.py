import json
import math
import os
import statistics
import subprocess
import tarfile
import time
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
from .common import *

def stamp(t):
    dt=datetime.fromtimestamp(t,timezone.utc)
    return {'UTC':dt.isoformat(),'Asia_Shanghai':dt.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %z')}
def decisions(root):
    return [read(p) for m,g in TRAJECTORIES for p in sorted((folder(root,m,g)/'decisions').glob('*.json'))]
def update_eta(root):
    status=read(root/'STATUS.json');ds=decisions(root)
    samples={m:[x for x in ds if x['method']==m] for m in METHODS}
    formal=all(sum(x['method']==m and x['group']==0 for x in ds)>=10 for m in METHODS)
    if not ds:return
    with connect(root) as db:
        attempts=[dict(x) for x in db.execute('SELECT * FROM attempts WHERE exit_code=0')]
        physical=[dict(x) for x in db.execute('SELECT * FROM tasks')]
        logical=[dict(x) for x in db.execute('SELECT * FROM logical')]
    actual=[read(root/'training'/x['task']/'execution.json') for x in attempts if (root/'training'/x['task']/'execution.json').exists()]
    search=[x['task_wall_seconds'] for x in actual if x['stage']=='search'];sm=float(np.mean(search)) if search else None
    if sm is None:return
    capacity=read(root/'resources.json')['training_workers'];online=0.;by_method={};max_chain=0.
    for m in METHODS:
        measured=samples[m]
        fit_mean=float(np.mean([x['fit_seconds'] for x in measured])) if measured else 0.
        acq_mean=float(np.mean([x['acquisition_seconds'] for x in measured])) if measured else 0.
        measured_n=float(np.mean([x['observed_n'] for x in measured])) if measured else 50.
        per_group=[];work=0.
        for g in range(5):
            obs=read(folder(root,m,g)/'observations.json') if (folder(root,m,g)/'observations.json').exists() else []
            remaining=list(range(max(50,len(obs)),300));times=[]
            for n in remaining:
                growth=max(1.,n/measured_n);times.append(fit_mean*growth+min(300.,acq_mean*math.sqrt(growth)))
            work+=sum(times);max_chain=max(max_chain,sum(times)+len(remaining)*sm)
            per_group.append({'group':g,'observations':len(obs),'remaining_rounds':len(remaining),'projected_CPU_seconds':sum(times)})
        by_method[m]={'formal_rounds':len(measured),'group0_rounds':sum(x['group']==0 for x in measured),'fit_seconds_mean':fit_mean,'acquisition_seconds_mean':acq_mean,'measured_n_mean':measured_n,'remaining':per_group};online+=work
    phase=[];gpu_search=0.;gpu_final=0.
    for p,total in [('search',3000),('validation',3000),('final',300)]:
        done=sum(x['phase']==p for x in logical);known={x['task'] for x in logical if x['phase']==p};pending=[x for x in physical if x['id'] in known and x['source']=='OPEN-new' and x['status']!='succeeded']
        left=max(0,total-done)+len(pending)
        vals=[x['task_wall_seconds'] for x in actual if x['stage']==p]
        mean=float(np.mean(vals)) if vals else sm*(1. if p=='search' else 2.)
        seconds=left*mean
        if p=='search':gpu_search=seconds/capacity
        else:gpu_final+=seconds/capacity
        phase.append({'phase':p,'unrequested_plus_pending_unique_upper':left,'measured_samples':len(vals),'seconds_per_new_task':mean,'source':'current run measured stage' if vals else 'current search measured time times epoch ratio2; not old fixed-pool wallclock','projected_GPU_seconds':seconds})
    search_wall=max(online/2,gpu_search,max_chain);remaining=search_wall+gpu_final+300;at=time.time()
    eta={'at':now(),'formal_A0_C0_10_each':formal,'training_samples':len(actual),'search_GNN_seconds_mean':sm,'by_method':by_method,'stage_training':phase,
      'remaining_wall_seconds':remaining,'estimated_completion':stamp(at+remaining),'range_completion':[stamp(at+remaining*.65),stamp(at+remaining*1.65)],
      'assumptions':['current open-run timings only','GP fit growth linear in n, acquisition sqrt(n); early extrapolation descriptive, not confidence interval',
       'two numerical processes; actual admitted GPU count','future unrequested cache hits not deducted','search critical path +later GPU-stage load; overlap not credited','300s scoring/report allowance'],
      'measured_rounds':len(ds),'first_20_required_rounds_count_in_250':True}
    write(root/'ETA.json',eta)
    if formal:
        old=read(root/'acceptance_schedule.json') if (root/'acceptance_schedule.json').exists() else None
        if old is None or not old.get('first_triggered',False):
            schedule={'created_at':old['created_at'] if old else now(),'updated_at':now(),'due_unix':at+remaining+900,'due':stamp(at+remaining+900),
               'buffer_seconds':900,'first_triggered':False,'update_until_trigger_authorized':True,'healthy_repeat_seconds':300,'original_due':old.get('original_due',old.get('due')) if old else stamp(at+remaining+900)}
            write(root/'acceptance_schedule.json',schedule);append(root/'ETA_history.jsonl',{'ETA':eta,'schedule':schedule})

def final_report(root):
    scores=read(root/'AC_final_scores.json');rows=[];curves=[];top=[];rounds=decisions(root);acqrows=[]
    for method,g in TRAJECTORIES:
        f=folder(root,method,g);obs=read(f/'observations.json');champ=read(f/'champion.json');s=[x for x in scores if x['method']==method and x['group']==g]
        pool=read(inputs(root,g)/'candidates.json');cids={x['candidate_id'] for x in pool};aids={x['architecture_id'] for x in pool}
        search_best=max(o['validation_accuracy'] for o in obs);search_cid=min(obs,key=lambda o:(-o['validation_accuracy'],o['candidate_id']))['candidate_id']
        row={'method':method,'group':g,'candidate_id':champ['candidate_id'],'configuration':champ['configuration'],
          'final_validation_mean':float(np.mean([x['validation_accuracy'] for x in s])),'final_validation_SD':float(np.std([x['validation_accuracy'] for x in s],ddof=1)),
          'test_mean':float(np.mean([x['test_accuracy'] for x in s])),'test_SD':float(np.std([x['test_accuracy'] for x in s],ddof=1)),
          'reevaluation_validation_mean':champ['validation_mean'],'reevaluation_validation_SD':champ['validation_sd'],'reevaluation_changed_champion':search_cid!=champ['candidate_id'],
          'initial50_best':max(o['validation_accuracy'] for o in obs[:50]),'search300_best':search_best,'search_improvement_pp':100*(search_best-max(o['validation_accuracy'] for o in obs[:50])),
          'champion_discovery_query':next(i+1 for i,o in enumerate(obs) if o['candidate_id']==champ['candidate_id']),
          'online_outside_pool_complete_configurations':sum(o['candidate_id'] not in cids for o in obs[50:]),
          'online_new_graphs_relative_to_original_pool':sum(o['architecture_id'] not in aids for o in obs[50:]),
          'distinct_graphs_300':len({o['architecture_id'] for o in obs}),'missing_incoming_fallback_selected':sum(any(i not in {e[1] for e in o['configuration']['edges']} for i in range(1,len(o['configuration']['operations'])+1)) for o in obs)}
        rows.append(row)
        for i,o in enumerate(obs):curves.append({'method':method,'group':g,'query':i+1,**o})
        for k in [50,100,150,200,250,300]:row[f'best_at_{k}']=max(o['validation_accuracy'] for o in obs[:k])
        for x in read(f/'validation_reevaluation.json'):top.append({'method':method,'group':g,**x})
    diffs=np.array([next(x['test_mean'] for x in rows if x['method']=='C' and x['group']==g)-next(x['test_mean'] for x in rows if x['method']=='A' and x['group']==g) for g in range(5)])
    valdiff=np.array([next(x['final_validation_mean'] for x in rows if x['method']=='C' and x['group']==g)-next(x['final_validation_mean'] for x in rows if x['method']=='A' and x['group']==g) for g in range(5)])
    rng=np.random.default_rng(proposal_seed(0,31,0));idx=rng.integers(0,5,size=(10000,5));test_interval=np.percentile(diffs[idx].mean(1),[2.5,97.5])
    paired={'C_minus_A_test_pp':(diffs*100).tolist(),'mean_test_difference_pp':float(diffs.mean()*100),'SD_test_difference_pp':float(diffs.std(ddof=1)*100),
            'description95_interval_pp':(test_interval*100).tolist(),'C_minus_A_final_validation_pp':(valdiff*100).tolist(),'bootstrap_seed':proposal_seed(0,31,0)}
    for d in rounds:
        diag=d['diagnostics'];acqrows.append({'method':d['method'],'group':d['group'],'step':d['step'],'observed_n':d['observed_n'],'fit_seconds':d['fit_seconds'],
          'transform_seconds':d['transform_seconds'],'acquisition_seconds':d['acquisition_seconds'],'requests':d['requests'],'post_requests':d['post_requests'],
          'fallback_requests':d['fallback_requests'],'decoder_calls':diag['decoder_calls'],'encoder_calls':diag['encoder_calls'],
          'posterior_points':diag.get('posterior_points',diag.get('posterior_points_lower_bound')),'gradient_requests':diag.get('gradient_requests'),
          'forward_seconds':diag.get('forward_seconds'),'backward_seconds':diag.get('backward_seconds'),'WL_seconds':diag.get('WL_seconds'),
          'times':diag.get('times'),'events':diag['events'],'parameters':d['parameters'],'selected_CID':d['candidate_id'],'selected_graph':d['selected_row']['architecture_id'],
          'relaxed_realized_gap':d['selected_row']['relaxed_minus_realized'],'timeout_partial':diag.get('timeout_partial',False)})
    with connect(root) as db:
        physical=[dict(x) for x in db.execute('SELECT * FROM tasks')];logical=[dict(x) for x in db.execute('SELECT * FROM logical')];attempts=[dict(x) for x in db.execute('SELECT * FROM attempts')]
    method_cost={}
    for m in METHODS:
        rr=[x for x in rounds if x['method']==m];mapping=[x for x in logical if x['method']==m];phys={x['id']:x for x in physical}
        method_cost[m]={'GP_fit_seconds':sum(x['fit_seconds'] for x in rr),'GP_scale_distance_seconds':sum(x['transform_seconds'] for x in rr),
         'selection_seconds':sum(x['acquisition_seconds'] for x in rr),'decoder_calls':sum(x['diagnostics']['decoder_calls'] for x in rr),
         'logical_training_work_seconds':sum(phys[x['task']]['elapsed'] for x in mapping),
         'legacy_cache_uses':sum(x['cache_source'].startswith('legacy') for x in mapping),'cross_method_shared_uses':sum(x['cache_source']=='OPEN-shared' for x in mapping),
         'requests':sum(x['requests'] for x in rr),'backward_seconds':sum(x['diagnostics'].get('backward_seconds',0) for x in rr),
         'forward_seconds':sum(x['diagnostics'].get('forward_seconds',0) for x in rr),'WL_seconds':sum(x['diagnostics'].get('WL_seconds',0) for x in rr)}
    costs={'logical_total':6300,'new_unique_GNN_training':sum(x['source']=='OPEN-new' for x in physical),'counts':counts(root),'per_method':method_cost,
      'GPU_attempt_work_hours':sum((x['finished']-x['started'])/3600 for x in attempts),'training_stage_task_seconds':{p:sum(x['elapsed'] for x in physical if x['source']=='OPEN-new' and x['stage']==p) for p in ['search','validation','final']},
      'cache_lookup_wall_seconds':sum(json.loads(line)['seconds'] for line in (root/'cache_lookup_times.jsonl').read_text().splitlines()),
      'wall_seconds_to_report':time.time()-read(root/'STATUS.json')['started_unix'],'preparation_seconds':read(root/'freeze.json')['preparation_seconds'],
      'isolated_test_seconds':sum(read(p)['elapsed_seconds'] for p in (root/'new_test_scores').glob('*.json')),
      'timing_note':'phase totals and component timings are separate; cumulative work is not wallclock; candidate I/O not charged twice',
      'B_new_trajectories':0,'C_online_decoder_calls':method_cost['C']['decoder_calls'],'C_online_encoder_calls':0,'H_accessed':False}
    summary={'benchmark':'historical development benchmark','groups':rows,'paired_C_minus_A':paired,'costs':costs,'aggregates':{m:{k:{'mean':float(np.mean([x[k] for x in rows if x['method']==m])),
          'group_SD':float(np.std([x[k] for x in rows if x['method']==m],ddof=1))} for k in ['test_mean','final_validation_mean']} for m in METHODS}}
    write(root/'summary.json',summary);write(root/'COSTS.json',costs)
    csv_write(root/'FINAL_SEED_RESULTS.csv',scores);csv_write(root/'GROUP_COMPARISON.csv',rows);csv_write(root/'PAIRED_DIFFERENCES.csv',[{'group':g,'C_minus_A_test_pp':diffs[g]*100,'C_minus_A_validation_pp':valdiff[g]*100} for g in range(5)])
    csv_write(root/'SEARCH_CURVES.csv',curves);csv_write(root/'TOP10_REEVALUATION.csv',top);csv_write(root/'ACQUISITION_ROUNDS.csv',acqrows);csv_write(root/'TASK_MAPPING.csv',logical)
    text='# A/C 开放搜索结果\n\n'
    text+='A：RAW16梯度＋冻结VAE落地；C：完整稳定token W-I＋随机图/变异/混合HP优化。各5组，每组50初始化＋250在线；逻辑搜索3000、复评3000、final300。B未运行。\n\n'
    text+='|方法|5组test均值|组均值SD|final validation均值|\n|---|---:|---:|---:|\n'
    for m in METHODS:
        a=summary['aggregates'][m];text+=f"|{m}|{100*a['test_mean']['mean']:.4f}%|{100*a['test_mean']['group_SD']:.4f}pp|{100*a['final_validation_mean']['mean']:.4f}%|\n"
    text+=f"\nC−A五组test差(pp)：{paired['C_minus_A_test_pp']}；均差{paired['mean_test_difference_pp']:.4f}pp；描述95%区间{paired['description95_interval_pp']}。不把150个训练seed当150次独立NAS。\n\n"
    text+=f"C在线decoder调用{costs['C_online_decoder_calls']}、encoder调用0；新图与完整配置池外比例见GROUP_COMPARISON。新增唯一训练{costs['new_unique_GNN_training']}，精确缓存和共享分别对账。各方法GP/选点/训练累计耗时见COSTS，不因后跑缓存更多声称算法加速。\n\n"
    text+='本轮比较包含初始化、代理、候选生成和优化全流程差异，不能单归因于W-I核，也不能把采集值当准确率。A冻结VAE的可达图分布与C直接图族生成分布不同。保留无入边DynamicGNN fallback与记录图表示限制。旧test与复评/final种子已有历史开发暴露；本轮不是独立确认实验。所有10冠军与300最终逻辑模型先封存，再统一独立float32评分；H未访问。\n'
    (root/'RESULTS.md').write_text(text);(root/'GPT_ANALYSIS_REQUEST.md').write_text('请复核PROTOCOL/SANITY、10条300观测轨迹、GP与采集档案、Top10与最终逐seed成绩及C−A五组配对描述bootstrap。分析质量、GP/选点/GNN成本、C是否绕过decoder、完整配置池外与新图比例；保留缓存/旧test/旧训练seed暴露、图族可达分布差异、原WL/fallback限制。禁止把C优势单归因核，禁止将采集值当精度或追加实验。\n')

def accept_complete(root):
    from .prepare import verify
    verify(root);ds=decisions(root);scores=read(root/'AC_final_scores.json');barrier=read(root/'AC_all_final_models_sealed.json')
    if len(ds)!=2500 or len(scores)!=300 or barrier['logical_models']!=300:raise ValueError('Open trajectory/score counts incomplete')
    with connect(root) as db:
        if db.execute('SELECT COUNT(*) FROM logical').fetchone()[0]!=6300:raise ValueError('Logical training budget incomplete')
        if db.execute('SELECT COUNT(*) FROM tasks WHERE status!=?',('succeeded',)).fetchone()[0]:raise ValueError('Unfinished training')
        for x in db.execute('SELECT id FROM tasks'):result(root,x['id'])
    for m,g in TRAJECTORIES:
        f=folder(root,m,g);obs=read(f/'observations.json')
        if len(obs)!=300 or len({o['candidate_id'] for o in obs})!=300:raise ValueError('300 unique configurations missing')
        if [o['candidate_id'] for o in obs[:50]]!=read(f/'initialization.json')['ids']:raise ValueError('Initialization order changed')
        if m=='C' and any(o['latent'] is not None for o in obs):raise ValueError('C has invented latent')
        ree=read(f/'validation_reevaluation.json');chosen=min(ree,key=lambda r:(-r['validation_mean'],r['candidate_id']))
        if len(ree)!=10 or chosen['candidate_id']!=read(f/'champion.json')['candidate_id']:raise ValueError('Validation champion wrong')
    for d in ds:
        if not 0<d['requests']<=2048 or d['post_requests']>64 or d['fallback_requests']>(128 if d['method']=='A' else 64) or d['acquisition_seconds']>300:raise ValueError('Round hard budget violated')
        normal=d['diagnostics']['normal_requests'];starts=d['diagnostics']['starts']
        if normal>(1856 if d['method']=='A' else 1920) or any(c>(200 if d['method']=='A' else 30) for c in starts.values()):raise ValueError('Optimizer detail cap violated')
        if d['method']=='C' and (d['diagnostics']['decoder_calls'] or d['diagnostics']['encoder_calls']):raise ValueError('C called VAE')
    for s in scores:
        if s['computed_at']<barrier['sealed_at'] or s['training_updates'] or s['dtype']!='float32' or s['B_score_value_copied']:raise ValueError('Test gate/scoring isolation violated')
        if sha(s['checkpoint'])!=s['checkpoint_sha256']:raise ValueError('Final checkpoint changed')
    if any(read(root/'worker_exit_codes.json')) or read(root/'scoring_exit_code.json')!=0:raise ValueError('Unclean worker/scoring exit')
    receipt={'checks':'PASS','at':now(),'logical_training':6300,'trajectories':10,'search_configs_each':300,'online_rounds':2500,'final_logical_models':300,
      'all_round_caps_verified':True,'C_online_VAE_calls':0,'test_barrier_verified':True,'H_accessed':False,'B_new_trajectories':0,'historical_sources_unchanged':True}
    write(root/'ACCEPTANCE.json',receipt);return receipt

def package(root):
    path=root/'EVIDENCE.tar.gz';tmp=root/'EVIDENCE.tar.gz.tmp'
    with tarfile.open(tmp,'w:gz') as t:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p in [path,tmp] or 'sanity' in p.parts:continue
            if any(part in p.parts for part in ['training','new_test_scores','code']):continue
            if p.suffix in ['.pt','.pth','.sqlite','.log','.lock','.tmp'] or p.name.endswith(('-wal','-shm')):continue
            if p.name in ['archive.jsonl','training.json','package_manifest.json'] or p.name.startswith('start') and p.suffix=='.jsonl':continue
            t.add(p,arcname=str(p.relative_to(root)),recursive=False)
        for p in sorted(Path('analyse/ac_open_graph_mix').glob('*')):
            if p.suffix in ['.py','.sh']:t.add(p,arcname='code/'+p.name,recursive=False)
    tmp.replace(path);write(root/'package_manifest.json',{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'full_archives_and_weights_retained_on_server':True,'at':now()})

def pass_report(root,accepted,trigger):
    write(root/'ACCEPTANCE.json',{**accepted,'trigger':trigger});write(root/'acceptance_status.json',{'state':'PASS','at':now()});package(root)

def timer(root):
    write(root/'acceptance_status.json',{'state':'WAITING_FIRST_20_FORMAL_ROUNDS','pid':os.getpid(),'at':now()});next_due=None;first_triggered=False;recoveries=0
    while True:
        if (root/'COMPLETE.json').exists():pass_report(root,read(root/'COMPLETE.json')['accepted'],'pipeline completion event');return
        status=read(root/'STATUS.json')
        if status['state']=='FAILED':
            error=status.get('error','');eligible=any(k in error for k in ['out of memory','temporarily unavailable','InterruptedError','ConnectionResetError'])
            if eligible and recoveries<3 and all(not pid_alive(read(p).get('pid')) for p in list(root.glob('worker?.json'))+list(root.glob('gp_worker?.json'))):
                append(root/'ERRORS.jsonl',{'at':now(),'phase':'automatic_exact_checkpoint_recovery','attempt':recoveries+1,'error':error});recoveries+=1
                with (root/'controller.log').open('a') as f:
                    subprocess.Popen([read(root/'resources.json')['python'],'-u','-B','-m','analyse.ac_open_graph_mix','run','--recover','--run-dir',str(root)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(10);continue
            write(root/'acceptance_status.json',{'state':'FAILED_OR_PAUSED','at':now(),'error':error,'automatic_recovery_attempts':recoveries});return
        if (root/'acceptance_schedule.json').exists():
            schedule=read(root/'acceptance_schedule.json')
            if not first_triggered:next_due=schedule['due_unix']
            if next_due is not None and time.time()>=next_due:
                first_triggered=True;schedule['first_triggered']=True;write(root/'acceptance_schedule.json',schedule)
                write(root/'acceptance_status.json',{'state':'HEALTHY_INCOMPLETE','at':now(),'next_check':stamp(time.time()+300),'status':status});next_due=time.time()+300
        time.sleep(30)
