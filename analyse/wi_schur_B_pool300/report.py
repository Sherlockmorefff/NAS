"""Measured ETA, full-budget checks, and B-only final analysis artifacts."""
import json
import os
import tarfile
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from .common import *

def format_time(epoch):
    dt=datetime.fromtimestamp(epoch,timezone.utc)
    return {'UTC':dt.isoformat(),'Asia_Shanghai':dt.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')}

def update_eta(root):
    with connect(root) as db:
        tasks=[dict(r) for r in db.execute('SELECT * FROM tasks')]
    samples=[]
    for t in tasks:
        p=root/'training'/t['id']/'execution.json'
        if t['status']=='succeeded' and p.exists():samples.append(read(p))
    searched=[r for r in samples if r['stage']=='search']
    if len(searched)<10:return None
    means={};perworker={}
    for stage,total in [('search',1500),('validation',1500),('final',150)]:
        selected=[r for r in samples if r['stage']==stage]
        if selected:
            arr=np.asarray([r['task_wall_seconds'] for r in selected]);mean=float(arr.mean())
            source='measured formal '+stage+' tasks';lo=float(np.quantile(arr,.25));hi=float(np.quantile(arr,.90))
        else:
            # Explicit epoch/patience extrapolation, not a 3150*search_mean/2 estimate.
            baseline=float(np.mean([r['task_wall_seconds'] for r in searched]));mean=baseline*2
            lo=baseline*1.3;hi=baseline*3.;source='unmeasured final300/80; search150/40 elapsed scaled 1.3..3, midpoint2'
        done=sum(1 for t in tasks if t['stage']==stage and t['status']=='succeeded');remaining=total-done
        means[stage]={'measured_tasks':len(selected),'planned_logical':total,'actual_completed':done,'remaining_unconfirmed_cache_tasks':remaining,
          'seconds_per_task':mean,'seconds_range':[lo,hi],'estimate_source':source,
          'estimated_remaining_GPU_hours':remaining*mean/3600,'remaining_GPU_hours_range':[remaining*lo/3600,remaining*hi/3600]}
        for w in range(2):
            ss=[r for r in selected if r['worker']==w]
            perworker[f'{stage}_gpu{w}']={'samples':len(ss),'seconds_mean':float(np.mean([r['task_wall_seconds'] for r in ss])) if ss else None}
    decisions=[read(p) for p in root.glob('groups/*/decisions/*.json')]
    stages=read(root/'status.json').get('groups',{});nfits=len(decisions);remaining_fits=1250-nfits
    if decisions:
        coeff=np.median([d['fit_seconds']/max(1.,(d['observed_n']/50.)**2) for d in decisions])
        fit_remaining=sum(float(coeff)*max(1.,(n/50.)**2) for g in range(5) for n in range(max(50,stages.get(str(g),{}).get('observations',0)),300))
        acq_per=float(np.mean([d['acquisition_seconds'] for d in decisions]));gp_range=[fit_remaining*.7,fit_remaining*2.]
        gp_source='formal W-I fits, quadratic extrapolation in observed n; range0.7..2; updated from actual fits'
    else:
        # Conservative provisional bound, replaced as soon as official 50-point fits exist.
        growth=sum((n/50.)**1.5 for g in range(5) for n in range(50,300))
        fit_remaining=growth*5.;gp_range=[growth*2.,growth*20.];acq_per=1.
        gp_source='not yet measured at n50..299; provisional full three-start fit at n50=5 seconds, range2..20; growth(n/50)^1.5; replaced by formal timing'
    acq_remaining=remaining_fits*acq_per
    training_total=sum(v['estimated_remaining_GPU_hours']*3600 for v in means.values());gpu_wall=training_total/2
    # Sum rather than max until real overlap is established. This includes serial BO waits and CPU fitting;
    # 10% training margin covers unequal worker load and online queue idle time.
    dependence_margin=gpu_wall*.1;scoring_export=600.
    estimate=gpu_wall+fit_remaining+acq_remaining+dependence_margin+scoring_export
    low=sum(v['remaining_GPU_hours_range'][0]*3600 for v in means.values())/2+gp_range[0]+acq_remaining+scoring_export
    high=sum(v['remaining_GPU_hours_range'][1]*3600 for v in means.values())/2*1.15+gp_range[1]+acq_remaining+1200
    ts=time.time();signature=digest({g:(s.get('observations',0)//50,s.get('phase')) for g,s in stages.items()})
    eta={'updated_at':now(),'training_samples':len(samples),'first_timing_samples':samples[:12],'per_stage':means,'per_worker':perworker,
      'GP':{'measured_full_fits':nfits,'remaining_fits':remaining_fits,'remaining_seconds':fit_remaining,'range_seconds':gp_range,'source':gp_source},
      'acquisition_remaining_seconds':acq_remaining,'two_worker_training_wall_seconds':gpu_wall,'serial_dependency_and_load_margin_seconds':dependence_margin,
      'test_and_report_export_seconds':scoring_export,'remaining_wall_seconds':estimate,'wall_range_seconds':[low,high],
      'estimated_completion':format_time(ts+estimate),'range_completion':[format_time(ts+low),format_time(ts+high)],
      'checkpoint_signature':signature,'unconfirmed_cache_hits_deducted':0,'overlap_deducted':False}
    write(root/'ETA.json',eta)
    if not (root/'first_acceptance_schedule.json').exists():
        seal(root/'first_acceptance_schedule.json',{'created_at':now(),'estimated_remaining_wall_seconds_at_creation':estimate,
           'buffer_seconds':900,'due_unix':ts+estimate+900,'due':format_time(ts+estimate+900),
           'repeat_if_healthy_incomplete_seconds':300,'do_not_silently_postpone':True})
    old=root/'eta_last_landmark.json'
    if not old.exists() or read(old)['signature']!=signature:
        append(root/'eta_history.jsonl',eta);write(old,{'signature':signature,'at':now()})
    return eta

def accept_complete(root):
    from .prepare import verify
    verify(root)
    c=counts(root)
    if c['logical_enqueued']!={'search':1500,'validation':1500,'final':150}:raise ValueError('Logical budgets incomplete')
    if c['actual'].get('queued',0) or c['actual'].get('running',0) or c['actual'].get('failed',0):raise ValueError('Unresolved training tasks')
    if read(root/'worker_exit_codes.json')!=[0,0]:raise ValueError('Workers have not exited successfully')
    barrier=read(root/'all_final_models_sealed.json');scores=read(root/'final_scores.json')
    if len(scores)!=150:raise ValueError('Missing final test scores')
    for g in range(5):
        folder=root/'groups'/str(g);rows=read(folder/'candidates.json');obs=read(folder/'observations.json')
        if len(rows)!=768 or len(set(r['candidate_id'] for r in rows))!=768:raise ValueError('Invalid complete configuration pool')
        if any(digest(r['configuration'])!=r['candidate_id'] for r in rows):raise ValueError('Pool configuration identity changed')
        if len(obs)!=300 or len(set(o['candidate_id'] for o in obs))!=300:raise ValueError('Search incomplete')
        initial=read(folder/'initialization.json')
        if [o['candidate_id'] for o in obs[:50]]!=initial['ids'] or initial['kernel']!='original_global_RBF_Schur':raise ValueError('Wrong B initialization')
        seen=[]
        for i,o in enumerate(obs):
            r=result(root,o['task_id'])
            if r is None or not np.isfinite(o['validation_accuracy']) or r['best_validation_accuracy']!=o['validation_accuracy']:raise ValueError('Uncommitted search observation')
            if i>=50:
                d=read(folder/'decisions'/f'{i-49:03d}.json')
                if d['kind']!='qlogei' or d['step']!=i-49 or d['observed_ids']!=seen or d['candidate_id']!=o['candidate_id']:raise ValueError('Invalid online dependency')
                remaining=set(r['candidate_id'] for r in rows)-set(seen)
                if set(d['remaining_ids'])!=remaining or not np.isfinite([d['qlogei'],d['posterior_mean_accuracy'],d['posterior_latent_std_accuracy']]).all():raise ValueError('Invalid decision/prediction record')
                f=read(folder/'fits'/f'n{i:03d}'/'fit.json')
                if len(f['starts'])!=3 or any(s['adam_updates']!=300 or not np.isfinite(s['final_nll']) for s in f['starts']):raise ValueError('Incomplete GP fitting recipe')
            seen.append(o['candidate_id'])
        top=read(folder/'top10.json');v=read(folder/'validation_reevaluation.json');champ=read(folder/'champion.json')
        if len(top['ids'])!=10 or len(v)!=10 or any(len(r['validation_scores'])!=30 or not np.isfinite(r['validation_scores']).all() for r in v):raise ValueError('Top10 reevaluation incomplete')
        chosen=min(v,key=lambda r:(-r['validation_mean'],r['candidate_id']))
        if champ['candidate_id']!=chosen['candidate_id'] or champ['test_reads']!=0:raise ValueError('Wrong champion selection')
        final=read(folder/'final_models.json')
        if len(final['models'])!=30 or final['champion_sha256']!=sha(folder/'champion.json'):raise ValueError('Final model seal incomplete')
        for m in final['models']:
            r=result(root,m['task_id'])
            if r is None or sha(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('Final checkpoint missing')
        for s in [r for r in scores if r['group']==g]:
            if s['barrier_sha256']!=sha(root/'all_final_models_sealed.json') or s['computed_at']<barrier['sealed_at']:raise ValueError('Test before all-model barrier')
            if not np.isfinite([s['test_accuracy'],s['validation_accuracy']]).all():raise ValueError('Nonfinite final score')
    for name in ('RESULTS.md','GPT_ANALYSIS_REQUEST.md','costs.json','final_scores.csv','group_results.csv','search_curves.csv','analysis_package.tar.gz'):
        if not (root/name).is_file():raise ValueError('Missing delivery '+name)
    return {'checks':'PASS','groups':5,'queries_per_group':300,'logical_training':3150,'actual_counts':c['actual'],
       'only_B':True,'A_executed':False,'C_executed':False,'H_accessed':False,'test_barrier_verified':True,'at':now()}

def final_report(root):
    scores=read(root/'final_scores.json');groups=[];curves=[];toprows=[]
    for g in range(5):
        folder=root/'groups'/str(g);champ=read(folder/'champion.json');ss=[r for r in scores if r['group']==g]
        groups.append({'group':g,'candidate_id':champ['candidate_id'],'configuration':champ['configuration'],
          'reevaluation_validation_mean':champ['validation_mean'],'reevaluation_validation_sd':champ['validation_sd'],
          'final_training_validation_mean':float(np.mean([r['validation_accuracy'] for r in ss])),
          'final_training_validation_sd':float(np.std([r['validation_accuracy'] for r in ss],ddof=1)),
          'test_mean':float(np.mean([r['test_accuracy'] for r in ss])),'test_sd':float(np.std([r['test_accuracy'] for r in ss],ddof=1))})
        for o in read(folder/'observations.json'):curves.append({'group':g,**o})
        for r in read(folder/'validation_reevaluation.json'):toprows.append({'group':g,**r})
    test_means=np.asarray([r['test_mean'] for r in groups]);val_means=np.asarray([r['reevaluation_validation_mean'] for r in groups])
    rng=np.random.default_rng(2026101102);indices=rng.integers(0,5,(10000,5))
    summary={'only_B':True,'groups':groups,'test_group_mean':float(test_means.mean()),'test_group_sd':float(test_means.std(ddof=1)),
       'validation_group_mean':float(val_means.mean()),'validation_group_sd':float(val_means.std(ddof=1)),
       'test_group_bootstrap95':np.quantile(test_means[indices].mean(1),[.025,.975]).tolist(),
       'validation_group_bootstrap95':np.quantile(val_means[indices].mean(1),[.025,.975]).tolist(),
       'bootstrap_unit':'5 independent complete search groups; not150 independent searches','bootstrap_seed':2026101102,
       'test_role':'historical development benchmark','AC_comparisons':None}
    write(root/'summary.json',summary)
    with connect(root) as db:
        tasks=[dict(r) for r in db.execute('SELECT * FROM tasks')];attempts=[dict(r) for r in db.execute('SELECT * FROM attempts')]
        mapping=[dict(r) for r in db.execute('SELECT * FROM logical')]
    csv_write(root/'task_mapping.csv',mapping);csv_write(root/'tasks.csv',tasks);csv_write(root/'attempts.csv',attempts)
    csv_write(root/'final_scores.csv',scores);csv_write(root/'group_results.csv',groups);csv_write(root/'search_curves.csv',curves);csv_write(root/'top10_reevaluation.csv',toprows)
    costs={'counts':counts(root),'actual_unique_training_tasks':len(tasks),'attempts':len(attempts),'failed_attempts':sum(a['exit_code']!=0 for a in attempts),
       'GPU_task_hours':sum((a['finished']-a['started']) for a in attempts if a['finished'])/3600,
       'training_recorded_hours':sum(t['elapsed'] or 0 for t in tasks)/3600,
       'GP_fit_seconds':sum(read(p)['fit_seconds'] for p in root.glob('groups/*/fits/n*/fit.json')),
       'acquisition_seconds':sum(read(p)['acquisition_seconds'] for p in root.glob('groups/*/decisions/*.json')),
       'wall_seconds':time.time()-read(root/'status.json')['started_unix'],'resources':read(root/'resources.json'),
       'equal_training_counts_do_not_imply_equal_GPU_hours':True,'actual_training_restarts_visible_in_attempts':True}
    write(root/'costs.json',costs)
    text='# B 实验结果\n\n5 组均完成固定候选池内的完整 B 搜索：每组原 RBF-Schur 50 初始化 + 250 次在线 W-I GP/qLogEI。未执行 A/C，未读取或评分 H。\n\n'
    text+='搜索/复评/最终训练逻辑次数为1500/1500/150，总计3150；实际任务与尝试次数见 costs.json、tasks.csv、attempts.csv。\n\n'
    text+='|组|新训练种子复评 validation 均值/SD|最终 test 均值/SD|\n|---|---|---|\n'
    for r in groups:text+=f"|{r['group']}|{r['reevaluation_validation_mean']:.6f} / {r['reevaluation_validation_sd']:.6f}|{r['test_mean']:.6f} / {r['test_sd']:.6f}|\n"
    text+=f"\n5个组均值的 test 均值/SD：{summary['test_group_mean']:.6f}/{summary['test_group_sd']:.6f}；组级 bootstrap 描述95%区间 {summary['test_group_bootstrap95']}。5组不确定性估计有限，150次最终训练不是150次独立搜索。\n\n"
    text+='W-I使用单种子标签和 GaussianLikelihood 学习未知噪声；不提供已知SE。每组WL词表仅由冻结768池构建，尺度仅据该轨迹已见观测，3起点各300 Adam+原LBFGS，完整警告/NLL/参数见 fits。与原连续BO和原三种子观测协议存在已声明差异。\n\n'
    text+='冠军及150个最终模型均在本轮首次 test 前封存，见 all_final_models_sealed.json。test仅是历史开发基准；复评采用新训练种子及同一节点validation划分。\n\n'
    text+='现有证据描述B性能、组间及训练种子波动和成本；不能证明W-I优于RAW16或W-I初始化优于RBF，A/C问题未回答。科研解释仍需审阅 GPT_ANALYSIS_REQUEST.md。\n\n'
    text+='未来匹配A/C所需候选实际配置、16维代表、RBF顺序、无标签WL词表、全部种子、配方、决策、精确任务映射及代码身份保存在 groups/、protocol.json、freeze.json、code/、task_mapping.csv。大权重/最终checkpoint保留在服务器真实路径，不纳入分析包。\n\n'
    text+='各冠军完整 operations/edges/HP 见 summary.json。成本、失败及精确续跑记录见 costs.json、attempts.csv 和失败日志。全部训练已结束；本轮不自动启动A/C。\n'
    (root/'RESULTS.md').write_text(text)
    (root/'GPT_ANALYSIS_REQUEST.md').write_text('# B 结果分析请求\n\n请结合 RESULTS.md、summary.json、group_results.csv、search_curves.csv、top10_reevaluation.csv、final_scores.csv、costs.json 和完整配方分析B的组间与训练种子波动、搜索轨迹及计算成本。核查 fits 中数值/收敛警告和工程故障是否影响结论。\n\n只分析B，不造A/C基线，不推断W-I比RAW16更好。区分事实、解释和假说；同一Cora节点validation复用且旧test为历史开发基准。bootstrap以5组为单位。说明未来严格匹配A/C尚需哪些执行与明确协议。\n')
    package=root/'analysis_package.tar.gz'
    with tarfile.open(package,'w:gz') as archive:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or p==package or 'training' in rel.parts or p.suffix in ('.pt','.pth','.sqlite','.lock') or p.name.endswith(('.tmp','-wal','-shm')):continue
            archive.add(p,arcname=str(rel))
    write(root/'package_manifest.json',{'path':str(package),'sha256':sha(package),'weights_included':False,'created_at':now()})

def timed_acceptance(root):
    info={'pid':os.getpid(),'started_at':now(),'interval_after_healthy_incomplete_seconds':300}
    write(root/'acceptance_status.json',dict(info,state='WAITING_FOR_INITIAL_ETA'))
    while not (root/'first_acceptance_schedule.json').exists():
        status=read(root/'status.json') if (root/'status.json').exists() else {}
        if status.get('state')=='FAILED' or (status.get('state')=='RUNNING' and not pid_alive(status.get('pid'))):
            report={'state':'FAILED_BEFORE_ETA','at':now(),'status':status};write(root/'acceptance_report.json',report)
            write(root/'acceptance_status.json',dict(info,state='FAILED',ended_at=now()));return
        time.sleep(10)
    schedule=read(root/'first_acceptance_schedule.json');due=schedule['due_unix']
    write(root/'acceptance_status.json',dict(info,state='WAITING_UNTIL_DUE',next_due_unix=due,next_due=format_time(due)))
    while True:
        while time.time()<due:time.sleep(min(60,max(.1,due-time.time())))
        status=read(root/'status.json')
        try:
            if status['state']=='COMPLETE':
                checks=accept_complete(root);report={'state':'PASS','at':now(),'checks':checks}
                write(root/'acceptance_report.json',report);write(root/'acceptance_status.json',dict(info,state='PASS',ended_at=now()))
                (root/'ACCEPTANCE.md').write_text('# B 定时验收通过\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n');return
            if status['state']=='FAILED' or not pid_alive(status.get('pid')):raise RuntimeError('Controller failed/exited before complete: '+str(status))
            activity=read(root/'controller_activity.json') if (root/'controller_activity.json').exists() else {}
            with connect(root) as db:
                failed=db.execute("SELECT id,error FROM tasks WHERE status='failed'").fetchall()
                running=db.execute("SELECT id,started,pid FROM tasks WHERE status='running'").fetchall()
            if failed:raise RuntimeError('Failed training tasks: '+str([dict(r) for r in failed]))
            if any(not pid_alive(r['pid']) for r in running):raise RuntimeError('Running task worker exited')
            # GP can leave GPUs idle. Declare blocked only on a long stale activity/epoch record, not GPU utilization.
            if activity.get('state')=='GP_and_qLogEI' and time.time()-activity['started']>7200:raise RuntimeError('GP stage unchanged for more than2 hours')
            for r in running:
                p=root/'training'/r['id']/'epoch_events.jsonl'
                if time.time()-max(r['started'],p.stat().st_mtime if p.exists() else r['started'])>3600:raise RuntimeError('Training epoch state unchanged for more than1 hour')
            due=time.time()+300
            report={'state':'HEALTHY_INCOMPLETE','at':now(),'progress':status,'next_due_unix':due,'next_due':format_time(due)}
            append(root/'acceptance_history.jsonl',report);write(root/'acceptance_report.json',report)
            write(root/'acceptance_status.json',dict(info,state='WAITING_RECHECK',next_due_unix=due,next_due=format_time(due)))
        except Exception:
            import traceback
            report={'state':'FAILED_OR_BLOCKED','at':now(),'error':traceback.format_exc(),'status':status}
            write(root/'acceptance_report.json',report);write(root/'acceptance_status.json',dict(info,state='FAILED_OR_BLOCKED',ended_at=now()))
            (root/'ACCEPTANCE.md').write_text('# B 定时验收失败/阻塞\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n');return
