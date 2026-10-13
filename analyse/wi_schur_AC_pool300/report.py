import json
import os
import tarfile
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from .common import *

def format_time(ts):
    dt=datetime.fromtimestamp(ts,timezone.utc)
    return {'UTC':dt.isoformat(),'Asia_Shanghai':dt.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')}

def timing_reference(root):
    p=root/'B_timing_reference.json'
    if p.exists():return read(p)
    b=b_root(root)
    with sqlite3.connect((b/'tasks.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('SELECT stage,elapsed FROM tasks WHERE status=?',('succeeded',)).fetchall()
    stages={s:[float(v) for ss,v in rows if ss==s] for s in ('search','validation','final')}
    profile={}
    # Timing only, after AC protocol freeze. These fields never enter C geometry/GP.
    for g in range(5):
        for n in range(50,300):
            f=read(b/'groups'/str(g)/'fits'/f'n{n:03d}'/'fit.json')
            d=read(b/'groups'/str(g)/'decisions'/f'{n-49:03d}.json')
            profile.setdefault(str(n),[]).append([f['fit_seconds'],d['acquisition_seconds']])
    reference={'stage_seconds':stages,'WI_n_profile':{n:np.mean(v,axis=0).tolist() for n,v in profile.items()},
        'source':'B elapsed time only, never B scores or learned parameters','created_at':now()}
    write(p,reference);return reference

def update_eta(root):
    reference=timing_reference(root)
    with connect(root) as db:
        tasks=[dict(r) for r in db.execute('SELECT * FROM tasks')]
        logic=[dict(r) for r in db.execute('SELECT * FROM logical')]
        jobs=[dict(r) for r in db.execute('SELECT * FROM jobs')]
    samples=[]
    task_index={r['id']:r for r in tasks}
    for r in tasks:
        p=root/'training'/r['id']/'execution.json'
        if r['source']=='AC-new' and r['status']=='succeeded' and p.exists():samples.append(read(p))
    timing={};training_work={};ranges={};remaining={}
    for stage,planned in [('search',3000),('validation',3000),('final',300)]:
        selected=[s['task_wall_seconds'] for s in samples if s['stage']==stage]
        measured=bool(selected);values=selected if measured else reference['stage_seconds'][stage]
        mean=float(np.mean(values));lo,hi=np.quantile(values,[.25,.90]).tolist()
        records=[l for l in logic if l['phase']==stage];requested=len(records)
        pending={r['id'] for r in tasks if r['source']=='AC-new' and r['status'] in ('queued','running') and r['stage']==stage}
        unknown=planned-requested;physical_remaining=unknown+len(pending)
        training_work[stage]=physical_remaining*mean;ranges[stage]=[physical_remaining*lo,physical_remaining*hi]
        remaining[stage]={'logical_remaining':planned-sum(task_index[l['task']]['status']=='succeeded' for l in records),
            'unrequested_logical':unknown,'confirmed_pending_unique_new':len(pending),'future_cache_hits_deducted':0,
            'remaining_new_training_upper_count':physical_remaining,'seconds_per_training':mean,'seconds_range':[lo,hi],
            'source':'formal AC new physical tasks' if measured else 'B same150/40 or300/80 recipe actual elapsed times',
            'AC_measured_samples':len(selected),'remaining_GPU_hours':training_work[stage]/3600}
    decisions={m:[read(p) for p in (root/'trajectories'/m).glob('*/decisions/*.json')] for m in METHODS}
    cpu={};future_total=0.;cpu_low=0.;cpu_high=0.;chains=[]
    for method in METHODS:
        fits=decisions[method];ratios=[]
        for d in fits:
            base=reference['WI_n_profile'][str(d['observed_n'])]
            ratios.append([(d['fit_seconds']/base[0]),(d['acquisition_seconds']/base[1])])
        ratio=np.median(ratios,axis=0) if ratios else np.ones(2)
        qlo=np.quantile(ratios,.25,axis=0) if ratios else ratio
        qhi=np.quantile(ratios,.90,axis=0) if ratios else ratio
        fitwork=acqwork=low=high=0.;nremaining=0
        for m,g in [(m,g) for m,g in TRAJECTORIES if m==method]:
            f=folder(root,m,g);nobs=len(read(f/'observations.json')) if (f/'observations.json').exists() else 0
            done={j['n'] for j in jobs if j['method']==m and j['g']==g and j['status']=='succeeded'}
            chain=0.
            for n in range(max(50,nobs),300):
                if n in done:continue
                baseline=np.asarray(reference['WI_n_profile'][str(n)]);work=baseline*ratio
                fitwork+=float(work[0]);acqwork+=float(work[1]);low+=float((baseline*qlo).sum());high+=float((baseline*qhi).sum());chain+=float(work.sum());nremaining+=1
            chains.append(chain+max(0,300-nobs)*remaining['search']['seconds_per_training'])
        cpu[method]={'formal_decisions_measured':len(fits),'remaining_fits':nremaining,'fit_seconds':fitwork,'acquisition_seconds':acqwork,
            'measured_ratio_to_same_n_reference':ratio.tolist(),'source':('formal '+KERNELS[method]+' measured timings, projected on B per-n growth curve' if fits else ('B same W-I per-n actual timing' if method=='C' else 'RAW16 not yet timed: provisional WI-time upper reference, not RAW16 measurement'))}
        future_total+=fitwork+acqwork;cpu_low+=low;cpu_high+=high
    # Two independent GP processes and two GPU workers; use resource/trajectory critical paths.
    # No deduction for cross-phase overlap. No arbitrary safety multiplier.
    searchwall=max(training_work['search']/2,future_total/2,max(chains,default=0.))
    tail=(training_work['validation']+training_work['final'])/2
    export=300.;eta=searchwall+tail+export
    lo=max(ranges['search'][0]/2,cpu_low/2)+(ranges['validation'][0]+ranges['final'][0])/2+export
    hi=max(ranges['search'][1]/2,cpu_high/2,max(chains,default=0.))+(ranges['validation'][1]+ranges['final'][1])/2+export
    ts=time.time();worker_samples={str(i):[s for s in samples if s['worker']==i] for i in range(2)}
    value={'updated_at':now(),'new_training_samples':len(samples),'first_new_training_samples':samples[:12],
        'per_worker_new_samples':{i:len(v) for i,v in worker_samples.items()},'phase_estimates':remaining,'GP_and_acquisition':cpu,
        'GP_capacity_wall_seconds':future_total/2,'longest_online_chain_seconds':max(chains,default=0.),
        'search_stage_critical_path_seconds':searchwall,'final_stages_GPU_wall_seconds':tail,'scoring_report_seconds':export,
        'remaining_wall_seconds':eta,'remaining_wall_range_seconds':[lo,hi],'estimated_completion':format_time(ts+eta),
        'range_completion':[format_time(ts+lo),format_time(ts+hi)],'counts':counts(root),
        'assumptions':['future unrequested cache hits not deducted','RAW16 and WI measured separately; per-n growth projected from B',
          'two GP CPU processes, two GPU workers','search critical path plus final-stage load; cross-phase overlap not deducted',
          'timing quantiles are descriptive ranges, not confidence bounds','5-minute report/scoring allowance; no multiplicative safety factor']}
    write(root/'ETA.json',value)
    if not (root/'first_acceptance_schedule.json').exists():
        # Prefer8..12 new formal samples covering both GPUs. If only caches so far, establish a referenced ETA without blocking execution.
        if len(samples)>=10 and all(worker_samples.values()) or (ts-read(root/'status.json')['started_unix']>120):
            seal(root/'first_acceptance_schedule.json',{'created_at':now(),'due_unix':ts+eta+900,'due':format_time(ts+eta+900),
               'remaining_wall_seconds_at_creation':eta,'buffer_seconds':900,'repeat_if_healthy_incomplete_seconds':300,'not_silently_moved':True})
    states=read(root/'status.json').get('trajectories',{});landmark=digest({k:(v['observations']//50,v['phase']) for k,v in states.items()})
    old=root/'ETA_landmark.json'
    if not old.exists() or read(old)['identity']!=landmark:
        append(root/'ETA_history.jsonl',value);write(old,{'identity':landmark,'at':now()})
    return value

def accept_complete(root):
    from .prepare import verify
    from .initialization import select
    verify(root)
    with connect(root) as db:
        logic=[dict(r) for r in db.execute('SELECT * FROM logical')];physical=[dict(r) for r in db.execute('SELECT * FROM tasks')]
    if any(r['status']!='succeeded' for r in physical):raise ValueError('Unresolved physical training tasks')
    if read(root/'worker_exit_codes.json')!=[0,0,0,0] or read(root/'scoring_exit_code.json')!=0:raise ValueError('Worker/scoring exits not clean')
    for method in METHODS:
        for phase,n in [('search',1500),('validation',1500),('final',150)]:
            if len([l for l in logic if l['method']==method and l['phase']==phase])!=n:raise ValueError('Logical budget missing')
    for r in physical:
        value=result(root,r['id'])
        if task_key(read(root/'AC_PROTOCOL.json'),json.loads(r['config']),r['seed'],r['stage'])!=r['id'] or value['test_metric_computations']!=0:raise ValueError('Physical cache/evaluator key mismatch')
    barrier=read(root/'AC_all_final_models_sealed.json');scores=read(root/'AC_final_scores.json')
    if len(scores)!=300 or barrier['logical_models']!=300:raise ValueError('AC test mappings incomplete')
    for method,g in TRAJECTORIES:
        f=folder(root,method,g);obs=read(f/'observations.json');initial=read(f/'initialization.json');public=read(inputs(root,g)/'public_features.json')
        binit=read(inputs(root,g)/'initialization.json')
        if method=='A' and initial['ids']!=binit['ids']:raise ValueError('A initialization differs from B')
        if method=='C':
            expected=select(public,binit['first_point'])
            if initial['ids']!=expected['ids'] or initial['transform']!=expected['transform']:raise ValueError('C geometry/order changed')
        if initial['first_point']!=binit['first_point'] or len(set(initial['ids']))!=50:raise ValueError('Wrong shared first point/initial count')
        if len(obs)!=300 or len({o['candidate_id'] for o in obs})!=300 or [o['candidate_id'] for o in obs[:50]]!=initial['ids']:raise ValueError('Invalid search trajectory')
        for i,o in enumerate(obs):
            r=result(root,o['task_id'])
            if r['best_validation_accuracy']!=o['validation_accuracy']:raise ValueError('Wrong requested observation')
            if i>=50:
                d=read(f/'decisions'/f'{i-49:03d}.json');fit=read(f/'fits'/f'n{i:03d}'/'fit.json')
                if d['method']!=method or d['step']!=i-49 or d['observed_ids']!=[x['candidate_id'] for x in obs[:i]] or d['candidate_id']!=o['candidate_id'] or d['mc_seed']!=seed(g,'mc',i-49):raise ValueError('Wrong sequential acquisition history')
                if not np.isfinite([d['qlogei'],d['posterior_mean_accuracy'],d['posterior_latent_std_accuracy']]).all():raise ValueError('Invalid posterior/score values')
                if len(fit['starts'])!=3 or any(s['adam_updates']!=300 or not s['success'] or not np.isfinite(s['final_nll']) for s in fit['starts']):raise ValueError('Incomplete GP recipe')
        v=read(f/'validation_reevaluation.json');champ=read(f/'champion.json');models=read(f/'final_models.json')
        if len(v)!=10 or any(len(x['validation_scores'])!=30 or not np.isfinite(x['validation_scores']).all() for x in v):raise ValueError('Reevaluation incomplete')
        chosen=min(v,key=lambda x:(-x['validation_mean'],x['candidate_id']))
        if chosen['candidate_id']!=champ['candidate_id'] or champ['test_used_for_selection']:raise ValueError('Champion not validation selected')
        if len(models['models'])!=30:raise ValueError('Final training/mappings incomplete')
        for m in models['models']:
            if sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('Final checkpoint changed')
        for s in [x for x in scores if x['method']==method and x['group']==g]:
            if s['barrier_sha256']!=sha(root/'AC_all_final_models_sealed.json') or s['computed_at']<barrier['sealed_at'] or s['training_updates']!=0 or s['B_score_value_copied'] or s['dtype']!='float32':raise ValueError('Invalid isolated new-batch test scoring')
    for name in ('RESULTS.md','ABC_COMPARISON.md','GPT_ANALYSIS_REQUEST.md','costs.json','final_scores.csv','ABC_group_results.csv','analysis_package.tar.gz'):
        if not (root/name).is_file():raise ValueError('Missing delivery '+name)
    return {'checks':'PASS','AC_trajectories':10,'search_queries_each':300,'AC_logical_training':6300,'ABC_logical_training':9450,
            'test_barrier_verified':True,'B_unchanged':True,'H_accessed':False,'counts':counts(root),'at':now()}

def final_report(root):
    acscores=read(root/'AC_final_scores.json');bscores=read(b_root(root)/'final_scores.json')
    scores=acscores+[dict(s,method='B') for s in bscores];groups=[];curves=[];top=[];fallback=read(root/'AC_PROTOCOL.json')['representation_limitation']['missing_incoming_edge_fallback_ids']
    for method in ('A','B','C'):
        for g in range(5):
            f=(b_root(root)/'groups'/str(g)) if method=='B' else folder(root,method,g)
            champion=read(f/'champion.json');obs=read(f/'observations.json');rows=[s for s in scores if s['method']==method and s['group']==g]
            best_search=max(o['validation_accuracy'] for o in obs);search_champ=min(obs,key=lambda o:(-o['validation_accuracy'],o['candidate_id']))['candidate_id']
            groups.append({'method':method,'group':g,'candidate_id':champion['candidate_id'],'configuration':champion['configuration'],
              'final_validation_mean':float(np.mean([s['validation_accuracy'] for s in rows])),'final_validation_sd':float(np.std([s['validation_accuracy'] for s in rows],ddof=1)),
              'test_mean':float(np.mean([s['test_accuracy'] for s in rows])),'test_sd':float(np.std([s['test_accuracy'] for s in rows],ddof=1)),
              'reevaluation_validation_mean':champion['validation_mean'],'reevaluation_validation_sd':champion['validation_sd'],
              'initial50_best_validation':max(o['validation_accuracy'] for o in obs[:50]),'search300_best_validation':best_search,
              'search_improvement':best_search-max(o['validation_accuracy'] for o in obs[:50]),
              'returned_config_first_discovery_query':next(i+1 for i,o in enumerate(obs) if o['candidate_id']==champion['candidate_id']),
              'reevaluation_changed_search_champion':search_champ!=champion['candidate_id'],
              'fallback_configurations_selected':len({o['candidate_id'] for o in obs}&set(fallback[str(g)])),
              'initial50_fallback_selected':len({o['candidate_id'] for o in obs[:50]}&set(fallback[str(g)])),
              'top10_fallback_selected':len(set(read(f/'top10.json')['ids'])&set(fallback[str(g)])),
              'champion_has_recorded_WL_execution_fallback_mismatch':champion['candidate_id'] in fallback[str(g)]})
            curves.extend(dict(o,method=method,group=g) for o in obs)
            top.extend(dict(x,method=method,group=g) for x in read(f/'validation_reevaluation.json'))
    rng=np.random.default_rng(2026101102);indices=rng.integers(0,5,(10000,5));comparisons=[];aggregates={}
    for method in ('A','B','C'):
        rows=[r for r in groups if r['method']==method]
        aggregates[method]={metric:{'mean':float(np.mean([r[metric] for r in rows])),'group_sd':float(np.std([r[metric] for r in rows],ddof=1))} for metric in ('test_mean','final_validation_mean')}
    for title,left,right,role in [('B-A','B','A','primary'),('C-B','C','B','primary'),('C-A','C','A','auxiliary')]:
        for metric in ('test_mean','final_validation_mean'):
            l=sorted([r for r in groups if r['method']==left],key=lambda r:r['group']);rr=sorted([r for r in groups if r['method']==right],key=lambda r:r['group'])
            dif=np.asarray([a[metric]-b[metric] for a,b in zip(l,rr)])
            comparisons.append({'comparison':title,'role':role,'metric':metric,'paired_group_differences':dif.tolist(),
                'mean_difference':float(dif.mean()),'difference_sd':float(dif.std(ddof=1)),
                'descriptive_bootstrap95':np.quantile(dif[indices].mean(1),[.025,.975]).tolist(),'multiplicity_adjusted':False})
    with connect(root) as db:
        physical=[dict(r) for r in db.execute('SELECT * FROM tasks')];logical=[dict(r) for r in db.execute('SELECT * FROM logical')];attempts=[dict(r) for r in db.execute('SELECT * FROM attempts')]
    lookup={r['id']:r for r in physical};new=[r for r in physical if r['source']=='AC-new']
    per_method={m:{'logical_tasks':len([l for l in logical if l['method']==m]),
        'B_cache_uses':sum(l['method']==m and l['cache_source']=='B' for l in logical),
        'AC_shared_uses':sum(l['method']==m and l['cache_source']=='AC-shared' for l in logical),
        'logical_training_work_hours':sum(lookup[l['task']]['elapsed'] for l in logical if l['method']==m)/3600} for m in METHODS}
    costs={'counts':counts(root),'AC_new_unique_training':len(new),'B_unique_tasks_referenced':sum(r['source']=='B' for r in physical),
        'ABC_deduplicated_actual_training':3150+len(new),'AC_attempts':len(attempts),'AC_failed_attempts':sum(a['exit_code']!=0 for a in attempts),
        'AC_GPU_task_hours':sum(a['finished']-a['started'] for a in attempts)/3600,'B_GPU_task_hours_reference':read(b_root(root)/'costs.json')['GPU_task_hours'],
        'per_method_logical_work':per_method,'wall_hours':(time.time()-read(root/'status.json')['started_unix'])/3600,
        'GP':{m:{'fit_seconds':sum(read(p)['fit_seconds'] for p in (root/'trajectories'/m).glob('*/fits/*/fit.json')),
                 'acquisition_seconds':sum(read(p)['acquisition_seconds'] for p in (root/'trajectories'/m).glob('*/decisions/*.json'))} for m in METHODS},
        'new_test_scoring_seconds':sum(read(p)['elapsed_seconds'] for p in (root/'new_test_scores').glob('*.json')),
        'note':'B cached work not counted as new AC training; shared physical work counted once; method logical work separately estimated'}
    summary={'extension_after_Btest':True,'benchmark':'historical development benchmark','aggregates':aggregates,'groups':groups,
       'paired_comparisons':comparisons,'bootstrap_seed':2026101102,'bootstrap_replicates':10000,'unit':'5 paired search groups, not450 independent searches',
       'common_resampling_group_indices':True,'multiple_comparisons_adjusted':False,'H_accessed':False,'no_test_selected_overall_champion':True}
    write(root/'summary.json',summary);write(root/'costs.json',costs)
    csv_write(root/'ABC_group_results.csv',groups);csv_write(root/'paired_comparisons.csv',comparisons);csv_write(root/'final_scores.csv',scores)
    csv_write(root/'search_curves.csv',curves);csv_write(root/'top10_reevaluation.csv',top);csv_write(root/'logical_task_mapping.csv',logical)
    csv_write(root/'physical_tasks.csv',physical);csv_write(root/'attempts.csv',attempts)
    results='# 匹配 A/C 追加实验结果\n\nA/C各5组，均完成50初始化+250在线搜索、Top10×30validation复评和冠军×30final。A/C逻辑6300，连同B逻辑9450。B文件/数据库/模型只读，未重跑B，H未读取/评分。\n\n'
    results+='新增唯一训练 '+str(len(new))+'；包含B去重实际训练 '+str(3150+len(new))+'。缓存、尝试、GPU/GP/采集/评分/墙钟和按方法逻辑工作成本见costs.json和映射表。后运行方法命中更多缓存不等于其算法更快。\n\n'
    results+='A与B复用同50初始配置和精确搜索观测；A仅单seed RAW16各向同性Matérn5/2代理。C同首点，以已冻结的无标签W-I signal Schur和pool尺度选49点，后续WI性能代理全部尺度仅由C已见观测拟合。完整核/初值/300Adam+LBFGS/512MC/q250配方见AC_PROTOCOL.json。\n\n'
    results+='全部10个A/C冠军和300个final逻辑模型先封存，新独立float32评分进程在新门槛后对每个唯一checkpoint重新评分；prior_test_exposure逐模型记录，未抄B旧test值。训练/复评期间test计算为0。\n\n'
    results+='这是B test已查看后追加的历史开发基准匹配比较，不是三方法在历史首次test前共同预注册，也不恢复旧test独立性。新训练种子继续采用同一节点validation划分。\n\n'
    results+='保留B WL记录图未补入边而DynamicGNN执行存在前节点fallback的表示限制，未删除候选或补边。每组受影响池ID及各方法选择数量见AC_PROTOCOL.json、ABC_group_results.csv。10个完整冠军operations/edges/HP、原模型hash和真实权重路径见summary.json、final_scores.csv。\n\n'
    results+='实际结束原因、NLL、参数、梯度、jitter及警告保存在每个method/group/fits，不将警告直接视作失败。任何异常/恢复见日志；完整成功以COMPLETE.json和acceptance_report.json为准。完成后不追加新实验。\n'
    (root/'RESULTS.md').write_text(results)
    comparison='# A/B/C 配对比较\n\n|方法|5组test均值|组均值SD|5组final-seed validation均值|\n|---|---|---|---|\n'
    for m in ('A','B','C'):comparison+=f"|{m}|{100*aggregates[m]['test_mean']['mean']:.4f}%|{100*aggregates[m]['test_mean']['group_sd']:.4f}pp|{100*aggregates[m]['final_validation_mean']['mean']:.4f}%|\n"
    comparison+='\n|比较(test)|均差pp|组差SD pp|组级描述95%区间pp|\n|---|---|---|---|\n'
    for c in comparisons:
        if c['metric']=='test_mean':comparison+=f"|{c['comparison']}|{100*c['mean_difference']:.4f}|{100*c['difference_sd']:.4f}|{[100*v for v in c['descriptive_bootstrap95']]}|\n"
    comparison+='\n两个主比较为B−A、C−B，C−A辅助；先形成各组配对差，再以同一组号重采样10000次、seed2026101102取双侧95%分位描述区间，未进行多重比较校正，不作确认性显著性声明。只有5个完整搜索组，不把450次final训练当独立搜索。\n\n'
    comparison+='B−A是整套代理方案差异；C−B是整套初始化规则差异，涉及表示、几何尺度与核，不单独归因于核家族。每组差、初始50至300改善、发现步骤、复评换冠军、结构/HP和各训练seed分布保留，不按test挑一个方法/seed包装预选总冠军。科研解释仍需结合GPT_ANALYSIS_REQUEST审阅，不能只据单一总均值得出普遍结论。\n'
    (root/'ABC_COMPARISON.md').write_text(comparison)
    (root/'GPT_ANALYSIS_REQUEST.md').write_text('# A/B/C匹配比较分析请求\n\n请核查AC_PROTOCOL、summary、ABC_group_results、paired_comparisons、search_curves、top10、final_scores和costs。分析5组配对B−A和C−B（主比较）、C−A（辅助），使用已冻结的组级bootstrap口径，区分组间和30训练seed波动。解释搜索改善、冠军发现与复评切换、配置差异与逻辑/实际成本，不把缓存先后优势当算法加速。\n\n保留B test后追加身份、historical development benchmark与WL/执行fallback限制；未校正多重比较，不能声称独立测试或确认性显著性。区分事实、解释和假说，禁止据test重选总冠军/删除组；未知问题另列为后续研究，不自动运行。\n')
    package(root)

def package(root):
    path=root/'analysis_package.tar.gz'
    with tarfile.open(path,'w:gz',compresslevel=3) as archive:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p==path or p.name.endswith(('.pt','.pth','.sqlite','-wal','-shm','.lock','.tmp','.pyc')) or '__pycache__' in rel.parts:continue
            if 'training' in rel.parts and p.name!='execution.json':continue
            if 'fits' in rel.parts and (p.name=='training.json' or p.suffix=='.jsonl'):continue
            archive.add(p,arcname=str(rel))
    write(root/'package_manifest.json',{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'weights_included':False,
        'full_training_and_iteration_logs_retained_on_server':True,'at':now()})

def pass_report(root,checks,trigger):
    report={'state':'PASS','at':now(),'checks':checks,'trigger':trigger}
    write(root/'acceptance_report.json',report)
    (root/'ACCEPTANCE.md').write_text('# A/C 验收通过\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n')

def timed_acceptance(root):
    info={'pid':os.getpid(),'started_at':now(),'repeat_seconds_if_healthy_incomplete':300}
    due=None
    while True:
        status=read(root/'status.json') if (root/'status.json').exists() else {}
        if status.get('state')=='COMPLETE':
            checks=read(root/'COMPLETE.json')['accepted'];pass_report(root,checks,'pipeline completion event')
            write(root/'acceptance_status.json',dict(info,state='PASS',ended_at=now()));package(root);return
        if status.get('state')=='FAILED' or (status.get('state') in ('RUNNING','FINAL_TEST_SCORING','REPORT_AND_VERIFICATION') and not pid_alive(status.get('pid'))):
            report={'state':'FAILED_OR_BLOCKED','at':now(),'status':status};write(root/'acceptance_report.json',report)
            write(root/'acceptance_status.json',dict(info,state='FAILED_OR_BLOCKED',ended_at=now()));return
        if due is None and (root/'first_acceptance_schedule.json').exists():due=read(root/'first_acceptance_schedule.json')['due_unix']
        state='WAITING_FOR_ETA' if due is None else 'WAITING_UNTIL_DUE'
        if due is not None and time.time()>=due:
            with connect(root) as db:
                running=[dict(r) for r in db.execute('SELECT id,pid,started FROM tasks WHERE status=?',('running',))]
                jobs=[dict(r) for r in db.execute('SELECT * FROM jobs WHERE status=?',('running',))]
            error=None
            for j in jobs:
                if not pid_alive(j['pid']) or time.time()-j['started']>7200:error='GP process exited or unchanged2 hours'
            for r in running:
                p=root/'training'/r['id']/'epoch_events.jsonl';last=max(r['started'],p.stat().st_mtime if p.exists() else r['started'])
                if not pid_alive(r['pid']) or time.time()-last>3600:error='GNN process exited or epoch state unchanged1 hour'
            if error:
                write(root/'acceptance_report.json',{'state':'FAILED_OR_BLOCKED','at':now(),'error':error,'status':status})
                write(root/'acceptance_status.json',dict(info,state='FAILED_OR_BLOCKED',ended_at=now()));return
            due=time.time()+300;report={'state':'HEALTHY_INCOMPLETE','at':now(),'progress':status,'next_due_unix':due,'next_due':format_time(due)}
            append(root/'acceptance_history.jsonl',report);write(root/'acceptance_report.json',report);state='WAITING_RECHECK'
        write(root/'acceptance_status.json',dict(info,state=state,next_due_unix=due,next_due=format_time(due) if due else None,
            health_check_interval_seconds=30,completion_event_enabled=True))
        time.sleep(30 if due is None else min(30,max(.1,due-time.time())))
