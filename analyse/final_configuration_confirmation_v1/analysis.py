"""Sealed validation-only endpoint, frozen paired inference and final artifacts."""
import tarfile
import time
from .common import *

def selected_rule(comparisons, formal_M=True):
    s_pass= comparisons['S-U']['selection_CI'][0] > 0
    if not formal_M:
        return 'S' if s_pass else 'U'
    m_pass= comparisons['M-U']['selection_CI'][0] > 0
    if s_pass:
        return 'M' if m_pass and comparisons['M-S']['selection_CI'][0]>0 else 'S'
    return 'M' if m_pass else 'U'

def bootstrap(matrix, seed=2026100899, repetitions=10000, formal_M=True):
    import numpy as np
    matrix=np.asarray(matrix,dtype=np.int64)
    if matrix.shape!=(20,10,3):
        raise ValueError('Formal analysis requires complete 20x10x3 paired matrix')
    rng=np.random.default_rng(seed)
    groups=rng.integers(0,20,size=(repetitions,20))
    positions=rng.integers(0,10,size=(repetitions,10))
    samples=matrix[groups[:,:,None],positions[:,None,:],:].mean(axis=(1,2))/5
    means=matrix.mean(axis=(0,1))/5
    observed_groups=matrix.sum(axis=1)
    comparisons={}
    confidence=1-.05/3 if formal_M else .95
    tail=(1-confidence)/2
    for label,a,b in [('S-U',1,0),('M-U',2,0),('M-S',2,1)]:
        samples_diff=samples[:,a]-samples[:,b]
        group_diff=(observed_groups[:,a]-observed_groups[:,b])/50
        interval=np.quantile(samples_diff,[tail,1-tail]).tolist()
        comparisons[label]={'mean_pp':float(means[a]-means[b]),'selection_CI':interval,
            'selection_confidence':confidence,'CI95_pp':np.quantile(samples_diff,[.025,.975]).tolist(),
            'group_differences_pp':group_diff.tolist(),'wins':int((group_diff>0).sum()),
            'ties':int((group_diff==0).sum()),'losses':int((group_diff<0).sum()),
            'formal_for_selection':formal_M or label=='S-U'}
    return {'rule_means_percent':dict(zip(RULES,means.tolist())),'comparisons':comparisons,
        'selected_rule':selected_rule(comparisons,formal_M),'bootstrap_repetitions':repetitions,'statistics_seed':seed,
        'group_draws_sha256':digest(groups.tolist()),'replicate_draws_sha256':digest(positions.tolist()),
        'sampling':'shared crossed initialization x paired evaluation-replicate positions; same draws across three comparisons',
        'actual_evaluation_seeds':'disjoint between groups; replicate positions shared within groups',
        'conditional_on':'fixed C pool/query cache/rerank results and masks; no node bootstrap; not 200 independent searches'}

def score_gate(root):
    if not (root/'RETURN_CHECKPOINT_SEAL.json').exists():
        raise PermissionError('Validation endpoint analysis blocked before complete return/checkpoint seal')
    sealed=read(root/'RETURN_CHECKPOINT_SEAL.json')
    if sealed['returns']!=60 or sealed['evaluation_logical_tasks']!=600 or sealed['old_test_or_H_access']!=0:
        raise ValueError('Invalid endpoint seal')
    verify(root,sealed['files'])
    return sealed

def verify_complete(root,repo):
    verify_run(root,repo)
    done=read(root/'COMPLETE.json')
    verify(root,done['files'])
    score_gate(root)
    return done

def failed_report(root,queue,reason):
    tasks=queue.rows(); completed=sum(r['state']=='COMPLETE' for r in tasks)
    failed=[{'task_id':r['id'],'task':json.loads(r['payload']),'event':json.loads(r['event']) if r['event'] else None}
        for r in tasks if r['state']=='FAILED']
    write(root/'FAILED_TASKS.json',{'tasks':failed,'reason':reason,'completed_physical_tasks':completed,
        'formal_20_group_analysis':False,'free_retries':0,'time':now()})
    write(root/'SELECTED_RETURN_RULE.json',{'status':'INCOMPLETE','selected_rule':None,'default_rule':'U',
        'reason':'Complete frozen matrix unavailable; no formal rule selection','failure':reason})
    (root/'DEVELOPMENT_RESULTS.md').write_text('# 旧 C 终局返回规则开发：未完整完成\n\n'
        f'明确失败/阻塞：{reason}\n\n已完成物理任务 {completed}，记录失败 {len(failed)}。'
        '不伪造20组结果，不删除失败，不换seed补跑，不进行正式规则选择。\n\n'
        '实际费用见 BUDGET_LEDGER.csv / physical_task_ledger.csv；完整失败见 FAILED_TASKS.json。'
        '旧test和H均未开放；H资格只阻塞后续最终确认。\n')

def finalize(root,repo,queue):
    verify_run(root,repo)
    from .controller import complete_binding,progress
    manifest=read(root/'TASK_MANIFEST.json'); rows=queue.rows()
    assert len(rows)==len(manifest['tasks']) and 1400<=len(rows)<=1800
    assert all(r['state']=='COMPLETE' for r in rows)
    assert len(manifest['immutable_group_manifests'])==20
    all_tasks={r['id']:json.loads(r['payload']) for r in rows}; all_bindings={}; costs=[]
    for row in rows:
        result,binding=complete_binding(root,row); all_bindings[row['id']]=binding
        event=json.loads(row['event'])
        costs.append({'task_id':row['id'],'role':all_tasks[row['id']]['role'],'worker':row['worker'],
            'GPU_UUID':row['gpu'],'wall_seconds':event.get('wall_seconds'),
            'worker_seconds':result['worker_seconds'],'GPU_event_seconds':result['gpu_event_seconds'],
            'epochs':result['stopped_epoch'],'peak_GPU_bytes':result['peak_gpu_bytes'],
            'initializations':1,'strict_resumes':event.get('strict_resumes',0)})
    assert sum(t['role']=='rerank_seed' for t in all_tasks.values())==600
    assert sum(t['role']=='rerank_mask' for t in all_tasks.values())==600
    files={}; matrix=[]; endpoint=[]; group_table=[]; exploration=[]
    configurations=frozen_read(root/'CONFIGURATIONS.json')['by_id']
    for group in range(20):
        gm=root/'groups'/f'g{group:02d}.json'; assert sha(gm)==manifest['immutable_group_manifests'][str(group)]
        gp=read(gm); sm=read(root/'search_ready'/f'g{group:02d}_SM.json')
        obs_u=read(root/'trajectories'/f'g{group:02d}'/'U_observations.json')
        obs_sm=sm['observations']
        assert len(obs_u)==71 and len(obs_sm)==61 and obs_u[:60]==obs_sm[:60]
        assert gp['top5']==[o['candidate_id'] for o in ordered_observations(obs_sm,configurations)[:5]]
        assert read(root/'trajectories'/f'g{group:02d}'/'decisions/SM_terminal_n60.json')['kind']=='terminal_mean'
        assert read(root/'trajectories'/f'g{group:02d}'/'decisions/U_terminal_n70.json')['kind']=='terminal_mean'
        assert read(root/'trajectories'/f'g{group:02d}'/'decisions/U_q11.json')['kind']=='qlogei'
        returns={rule:read(root/'returns'/f'g{group:02d}_{rule}.json') for rule in RULES}
        latest_seal=max(r['sealed_at'] for r in returns.values())
        for rule,r in returns.items():
            p=root/'returns'/f'g{group:02d}_{rule}.json'; files[str(p.relative_to(root))]=sha(p)
            assert r['configuration_hash']==configurations[r['candidate_id']]['configuration_hash']
        seeds=frozen_read(root/'SEEDS.json')['groups'][str(group)]['evaluation']
        group_matrix=[]
        for replicate,seed in enumerate(seeds):
            values=[]
            for rule in RULES:
                r=returns[rule]; task=physical_task(root,group,'final_validation',r['candidate_id'],seed,'original')
                assert all_tasks[task['task_id']]==task
                folder=root/'training'/task['task_id']; binding=all_bindings[task['task_id']]
                init=read(folder/'INITIALIZATION.json')
                assert init['time']>=latest_seal
                assert binding['return_seals_before_evaluation'][rule]==sha(root/'returns'/f'g{group:02d}_{rule}.json')
                for name in ['best_model.pt','BINDING.json','result.json','selection.npz','INITIALIZATION.json']:
                    files[str((folder/name).relative_to(root))]=sha(folder/name)
                values.append(binding['correct'])
                endpoint.append({'group':group,'rule':rule,'replicate_id':replicate,'seed':seed,
                    'candidate_id':r['candidate_id'],'normalized_config_id':r['normalized_config_id'],
                    'task_id':task['task_id'],'correct':binding['correct'],'denominator':500,
                    'validation_percent':binding['correct']/5,'checkpoint_sha256':sha(folder/'best_model.pt')})
            group_matrix.append(values)
        matrix.append(group_matrix)
        for i,rule in enumerate(RULES):
            total=sum(v[i] for v in group_matrix)
            group_table.append({'group':group,'rule':rule,'candidate_id':returns[rule]['candidate_id'],
                'sum_correct':total,'denominator':5000,'validation_mean_percent':total/50,
                'configuration':returns[rule]['effective_configuration']})
        ub=ordered_observations(obs_u,configurations)[0]; sb=ordered_observations(obs_sm,configurations)[0]
        exploration.append({'group':group,'U_n71_best_query_percent':100*ub['query_correct_sum']/1500,
            'SM_n61_best_query_percent':100*sb['query_correct_sum']/1500,
            'U_minus_SM_query_best_pp':100*(ub['query_correct_sum']-sb['query_correct_sum'])/1500,
            'U_terminal':obs_u[-1]['candidate_id'],'SM_terminal':obs_sm[-1]['candidate_id'],
            'U_terminal_equals_SM_terminal':obs_u[-1]['candidate_id']==obs_sm[-1]['candidate_id']})
    csvwrite(root/'VALIDATION_RESULTS.csv',endpoint); csvwrite(root/'GROUP_RESULTS.csv',group_table)
    csvwrite(root/'EXPLORATION_COST.csv',exploration); csvwrite(root/'TRAINING_COSTS.csv',costs)
    seal(root/'RETURN_CHECKPOINT_SEAL.json',{'returns':60,'evaluation_logical_tasks':600,
        'evaluation_unique_tasks':sum(t['role']=='final_validation' for t in all_tasks.values()),
        'old_test_or_H_access':0,'all_tasks_complete':True,'files':files,'time':now()})
    score_gate(root)
    formal_M=read(root/'SPLITS.json')['formal_M_allowed']
    stats=bootstrap(matrix,formal_M=formal_M)
    write(root/'STATISTICS.json',stats)
    write(root/'SELECTED_RETURN_RULE.json',{'status':'COMPLETE','selected_rule':stats['selected_rule'],
        'default_rule':'U','M_formal_allowed':formal_M,'comparisons':stats['comparisons'],
        'protocol_sha256':sha(root/'PROTOCOL.json'),'endpoint':'mean ten new best-public-validation scores',
        'scope':'old C development, fixed query/rerank data; no independent generalization confirmation',
        'next_stage_started':False,'H_qualification':'NOT_CERTIFIED'})
    ledger=list(csv.DictReader((root/'BUDGET_LEDGER.csv').open()))
    assert len(ledger)==60 and all(int(r['logical_consumed'])==223 and r['logical_complete']=='True' for r in ledger)
    assert sum(int(r['logical_consumed']) for r in ledger)==13380
    source_code=frozen_read(root/'CODE_MANIFEST.json')['files']; verify(repo,source_code)
    write(root/'ACCEPTANCE.json',{'state':'PASS','groups':20,'rules':3,'matrix_shape':[20,10,3],
        'logical_training_cost':13380,'physical_initializations':len(rows),'query_cache_hits_logical':11580,
        'new_query_training':0,'rerank_seed_tasks':600,'rerank_mask_tasks':600,
        'evaluation_unique_tasks':len(rows)-1200,'failures':0,'all_masks_and_configs_bound':True,
        'own_top5_and_distinct_terminals_verified':True,'return_seals_precede_every_evaluation':True,
        'completed_tasks_reentered_without_new_training':True,'epoch5_actual_interruption_verified':True,
        'old_test_label_decodes':0,'H_label_decodes':0,'time':now()})
    elapsed=time.time()-read(root/'RUN_STARTED.json')['unix']
    fit_records=[read(p) for p in (root/'trajectories').glob('*/fits/n*/fit.json')]
    assert len(fit_records)==420 and sum(len(f['starts']) for f in fit_records)==1260
    write(root/'COSTS.json',{'logical':13380,'physical_initializations':len(rows),'query_new_training':0,
        'failures':0,'strict_resumes':sum(c['strict_resumes'] for c in costs),
        'worker_seconds':sum(c['worker_seconds'] for c in costs),'GPU_epoch_event_seconds':sum(c['GPU_event_seconds'] for c in costs),
        'formal_wall_seconds_including_scheduled_acceptance_wait':elapsed,
        'GP_fits':420,'GP_starts':1260,'GP_start_seconds':sum(f['fit_seconds'] for f in fit_records),
        'test_or_H_forwards':0,'interpretation':'GPU events are epoch-stream timing, not exclusive GPU utilization; wall includes waiting for appointment'})
    lines=['# 旧 C 终局返回规则开发结果','',f'20组×U/S/M已完整完成。最终选择 **{stats["selected_rule"]}**，按训练前冻结的区间规则决定。',
        '', '主指标是封存返回配置后，10个新种子各自最佳原validation成绩的平均；旧test和H均未开放。',
        '', '| 比较 | 平均差pp | 调整后98.333…%双侧区间 | 95%区间 | 胜/平/负 |',
        '|---|---:|---|---|---|']
    for label,c in stats['comparisons'].items():
        ci=c['selection_CI']; ci95=c['CI95_pp']
        lines.append(f'| {label} | {c["mean_pp"]:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | [{ci95[0]:.4f}, {ci95[1]:.4f}] | {c["wins"]}/{c["ties"]}/{c["losses"]} |')
    s=stats['comparisons']['S-U']; m=stats['comparisons']['M-S']
    interpret=lambda c: '确认本冻结开发任务上的平均收益' if c['selection_CI'][0]>0 else ('有改善方向，尚未确认收益' if c['mean_pp']>0 else '未确认平均收益；不能据此认定没有差异')
    import numpy as np
    lines.extend(['',f'复评是否值得：S−U {interpret(s)}；这是少探索10个配置、将30次训练转给复评后的完整预算比较。',
        f'增加划分是否优于只增加种子：M−S {interpret(m)}。M使用640开发标签，改变复评训练节点身份。',
        f'少探索10个配置的查询端代价：U的n71最佳查询均值相对S/M的n61平均差为 {np.mean([x["U_minus_SM_query_best_pp"] for x in exploration]):.4f}pp；终端查询在各自截止点重新执行，详见EXPLORATION_COST.csv。',
        '',f'逻辑费用13,380；实际新增完整训练{len(rows)}，其中复评1200、评估{len(rows)-1200}，新增查询训练0。GP 420次拟合/1260起点。',
        f'worker累计{sum(c["worker_seconds"] for c in costs)/3600:.3f}小时，CUDA epoch事件累计{sum(c["GPU_event_seconds"] for c in costs)/3600:.3f}小时；含预约等待的墙钟{elapsed/3600:.3f}小时。',
        '', '重采样共同抽取初始化组和配对评估重复位置，保留三规则共享关系。组间实际训练seed不重复；不是把200个分数当200次搜索，也不对图节点bootstrap。',
        '结论条件于旧C、固定查询成绩、复评seed/masks及已训练评估表。原validation同时用于早停和成绩，结果说明该固定validation上的返回质量，不能宣称独立泛化优势。',
        '', 'H尚未认证，后续独立最终确认受资格问题阻塞；本阶段不启动Prompt 3。历史结果保持原样。',
        '', '依据：PROTOCOL.json、VALIDATION_RESULTS.csv、GROUP_RESULTS.csv、STATISTICS.json、SELECTED_RETURN_RULE.json、BUDGET_LEDGER.csv、ACCEPTANCE.json。'])
    (root/'DEVELOPMENT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    report_files=['PROTOCOL.json','CODE_MANIFEST.json','TASK_MANIFEST.json','BUDGET_LEDGER.csv','physical_task_ledger.csv',
        'DEVELOPMENT_RESULTS.md','SELECTED_RETURN_RULE.json','H_QUALIFICATION.json','STATISTICS.json','COSTS.json',
        'VALIDATION_RESULTS.csv','GROUP_RESULTS.csv','EXPLORATION_COST.csv','TRAINING_COSTS.csv','ACCEPTANCE.json',
        'RETURN_CHECKPOINT_SEAL.json','TIMING_PILOT.csv','ETA.json','FREEZE.json','CACHE_ACCEPTANCE.json']
    checkpoint_manifest={str((root/'training'/r['id']/'best_model.pt').relative_to(root)):
        {'sha256':sha(root/'training'/r['id']/'best_model.pt'),'bytes':(root/'training'/r['id']/'best_model.pt').stat().st_size} for r in rows}
    write(root/'CHECKPOINT_MANIFEST.json',checkpoint_manifest); report_files.append('CHECKPOINT_MANIFEST.json')
    pack=root/'EVIDENCE.tar.gz'
    with tarfile.open(pack,'w:gz') as tar:
        for name in report_files:
            tar.add(root/name,arcname=name)
        for directory in ['frozen_code','returns','groups','search_ready','task_definitions','worker_events','trajectories']:
            tar.add(root/directory,arcname=directory)
        for row in rows:
            for name in ['result.json','BINDING.json','INITIALIZATION.json','selection.npz','attempts.jsonl']:
                path=root/'training'/row['id']/name
                tar.add(path,arcname=str(path.relative_to(root)))
        for name in ['SEEDS.json','SPLITS.json','CONFIGURATIONS.json','PILOT_MANIFEST.json','ACCEPTANCE_CHECKS.jsonl','GRANTS.jsonl','ENVIRONMENT_VERIFICATION.json','OPERATIONS.md','FINAL_EVAL_AUDIT.json']:
            if (root/name).exists():
                tar.add(root/name,arcname=name)
    complete_files={n:sha(root/n) for n in report_files}
    complete_files['EVIDENCE.tar.gz']=sha(pack)
    write(root/'COMPLETE.json',{'state':'COMPLETE','files':complete_files,'time':now(),
        'new_training_or_scoring_on_resume':0})
    write(root/'STATUS.json',{'state':'COMPLETE','time':now(),'selected_rule':stats['selected_rule'],
        'formal_initializations':len(rows),'logical':13380,'report':str(root/'DEVELOPMENT_RESULTS.md')})
