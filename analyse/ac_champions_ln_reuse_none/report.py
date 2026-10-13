import statistics
import tarfile
import time
import numpy as np
from .common import *

def direction(values):
    return {'positive':sum(v>1e-12 for v in values),'equal':sum(abs(v)<=1e-12 for v in values),'negative':sum(v< -1e-12 for v in values)}

def mean_sd(values):
    return {'mean':statistics.mean(values),'sample_SD':statistics.stdev(values)}

def aggregate(configs,metric):
    result={};rng=np.random.default_rng(2026101309);idx=rng.integers(0,5,size=(10000,5))
    arrays={m:np.array([next(x['delta_'+metric+'_pp_mean'] for x in configs if x['method']==m and x['group']==g) for g in range(5)]) for m in ('A','C')}
    for m in ['A','C','overall']:
        chosen=[x for x in configs if m=='overall' or x['method']==m]
        ds=[x['delta_'+metric+'_pp_mean'] for x in chosen]
        boot=(arrays['A'][idx].mean(1)+arrays['C'][idx].mean(1))/2 if m=='overall' else arrays[m][idx].mean(1)
        result[m]={'None_mean':statistics.mean(x['None_'+metric+'_mean'] for x in chosen),
                   'LN_mean':statistics.mean(x['LN_'+metric+'_mean'] for x in chosen),'delta_pp_mean':statistics.mean(ds),
                   'delta_pp_configuration_sample_SD':statistics.stdev(ds),'configuration_directions':direction(ds),
                   'descriptive_bootstrap95_pp':np.percentile(boot,[2.5,97.5]).tolist(),
                   'bootstrap_blocks':5,'bootstrap_replicates':10000,'bootstrap_seed':2026101309}
    return result

def validation_seal(root):
    verify(root)
    if (root/'LN_TRAINING_SEAL.json').exists():return
    if any((root/'test_scores').glob('*.json')) or (root/'LN_SEED_RESULTS.json').exists():
        raise ValueError('LN test exists before training/validation seal')
    baseline={(x['method'],int(x['group']),int(x['seed'])):float(x['validation_accuracy']) for x in csv_read(root/'BASELINE_VALIDATION.csv')}
    configs=[];models=[]
    for m,g in TRAJECTORIES:
        ts=[x for x in tasks(root) if x['method']==m and x['group']==g];none=[];ln=[]
        if len(ts)!=30:raise ValueError('Configuration LN budget mismatch')
        for t in ts:
            v=result(root,t);none.append(baseline[(m,g,t['seed'])]);ln.append(v['best_validation_accuracy'])
            models.append({'task_id':t['task_id'],'method':m,'group':g,'seed':t['seed'],'configuration_sha256':digest(t['configuration']),
                           'checkpoint':str(root/'training'/t['task_id']/'best_model.pt'),'checkpoint_sha256':v['checkpoint_hashes']['best_model.pt'],
                           'validation_accuracy':v['best_validation_accuracy'],'result_sha256':sha256(root/'training'/t['task_id']/'result.json')})
        ds=[100*(b-a) for a,b in zip(none,ln)]
        configs.append({'method':m,'group':g,'None_validation_mean':statistics.mean(none),'LN_validation_mean':statistics.mean(ln),
                        'delta_validation_pp_mean':statistics.mean(ds),'paired_direction_counts':direction(ds)})
    agg=aggregate(configs,'validation')
    positive=all(agg[m]['delta_pp_mean']>1e-12 for m in ('A','C','overall'))
    decision={'sealed_at':now(),'criterion':'overall/A/C validation means each >1e-12 pp','aggregates':agg,'configurations':configs,
              'recommendation':'值得进一步验证固定LN搜索' if positive else '暂不支持直接全面采用，需看配置依赖性',
              'damaged_configurations':[{k:x[k] for k in ('method','group','delta_validation_pp_mean')} for x in configs if x['delta_validation_pp_mean']< -1e-12],
              'LN_test_available_at_decision':False,'None_validation_source':'BASELINE_VALIDATION.csv projected from source final CSV',
              'automatic_fixed_LN_NAS':False}
    seal(root/'VALIDATION_DECISION.json',decision)
    seal(root/'LN_TRAINING_SEAL.json',{'sealed_at':now(),'new_LN_models':300,'None_new_models':0,'models':models,
         'validation_decision_sha256':sha256(root/'VALIDATION_DECISION.json'),'task_manifest_sha256':sha256(root/'LN_TASK_MANIFEST.json')})

def report(root,start):
    before=time.perf_counter()
    baseline=read_json(root/'BASELINE_MANIFEST.json')['rows'];ln=read_json(root/'LN_SEED_RESULTS.json')
    key=lambda x:(x['method'],int(x['group']),x['source_candidate_id'],int(x['seed']))
    old={key(x):x for x in baseline};new={key(x):x for x in ln}
    if len(old)!=300 or set(old)!=set(new):raise ValueError('None/LN strict keyed pairing failed')
    paired=[];combined=[]
    for task in tasks(root):
        a=old[key(task)];b=new[key(task)]
        row={k:task[k] for k in ('method','group','source_candidate_id','seed','seed_index')}
        for metric in ('validation','test'):
            row['None_'+metric+'_accuracy']=a[metric+'_accuracy'];row['LN_'+metric+'_accuracy']=b[metric+'_accuracy']
            row['delta_'+metric+'_pp']=100*(b[metric+'_accuracy']-a[metric+'_accuracy'])
        for label,item in [('None',a),('LN',b)]:
            for field in ['task_id','checkpoint','checkpoint_sha256','best_epoch','stopped_epoch','parameter_count','peak_gpu_memory_allocated_bytes']:
                row[label+'_'+field]=item.get(field)
            row[label+'_training_elapsed_seconds']=item.get('elapsed_seconds') if label=='None' else item['elapsed_seconds_training']
        paired.append(row)
        combined.append({k:a[k] for k in ['method','group','source_candidate_id','seed','task_id','checkpoint','checkpoint_sha256','best_epoch','stopped_epoch','parameter_count','validation_accuracy','test_accuracy','elapsed_seconds','peak_gpu_memory_allocated_bytes']} | {'condition':'none','provenance':'historical_reuse','new_training':False})
        combined.append({k:b[k] for k in ['method','group','source_candidate_id','seed','task_id','checkpoint','checkpoint_sha256','best_epoch','stopped_epoch','parameter_count','validation_accuracy','test_accuracy','peak_gpu_memory_allocated_bytes']} | {'condition':'layer_norm','provenance':'new_this_run','new_training':True,'elapsed_seconds':b['elapsed_seconds_training']})
    configs=[]
    for m,g in TRAJECTORIES:
        rows=[x for x in paired if x['method']==m and x['group']==g]
        if len(rows)!=30:raise ValueError('Paired configuration not 30 seeds')
        c={'method':m,'group':g,'source_candidate_id':rows[0]['source_candidate_id'],'pairs':30}
        for metric in ('validation','test'):
            for condition in ('None','LN'):
                v=mean_sd([x[condition+'_'+metric+'_accuracy'] for x in rows]);c[condition+'_'+metric+'_mean']=v['mean'];c[condition+'_'+metric+'_sample_SD']=v['sample_SD']
            ds=[x['delta_'+metric+'_pp'] for x in rows];v=mean_sd(ds)
            c['delta_'+metric+'_pp_mean']=v['mean'];c['delta_'+metric+'_pp_sample_SD']=v['sample_SD']
            c[metric+'_pair_directions']=direction(ds)
        c['validation_test_mean_same_direction']=(np.sign(c['delta_validation_pp_mean'])==np.sign(c['delta_test_pp_mean']))
        configs.append(c)
    aggregates={metric:aggregate(configs,metric) for metric in ('validation','test')}
    csv_write(root/'PAIRED_SEED_RESULTS.csv',paired);csv_write(root/'COMBINED_CONDITION_RESULTS.csv',combined);csv_write(root/'CONFIG_SUMMARY.csv',configs)
    csv_write(root/'METHOD_SUMMARY.csv',[{'method':m,**{metric:aggregates[metric][m] for metric in aggregates}} for m in ('A','C','overall')])
    with connect(root) as db:attempts=[dict(x) for x in db.execute('SELECT * FROM attempts')]
    execution=[read_json(root/'training'/t['task_id']/'execution.json') for t in tasks(root)]
    results=[result(root,t) for t in tasks(root)]
    gpu_cost={str(i):sum(x['finished']-x['started'] for x in attempts if x['worker']==i) for i in range(len(read_json(root/'resources.json')['GPUs']))}
    idle={}
    for i in gpu_cost:
        spans=[x for x in attempts if x['worker']==int(i)]
        elapsed=max(x['finished'] for x in spans)-min(x['started'] for x in spans)
        idle[i]={'observed_worker_span_seconds':elapsed,'attempt_active_seconds':gpu_cost[i],'between_task_idle_seconds':elapsed-gpu_cost[i]}
    timing={'at':now(),'new_LN_tasks':300,'new_None_tasks':0,'historical_None_results_reused':300,
            'preparation_seconds':read_json(root/'freeze.json')['preparation_seconds'],
            'new_LN_training_elapsed_seconds_sum':sum(x['elapsed_seconds'] for x in results),
            'new_LN_task_wall_seconds_sum':sum(x['task_wall_seconds'] for x in execution),
            'new_LN_GPU_attempt_active_seconds':gpu_cost,'worker_span_and_idle':idle,
            'new_LN_stage_seconds_sum':{k:sum(x['stage_timing'][k] for x in results) for k in ['forward_loss_seconds','backward_Adam_scheduler_seconds','validation_seconds','epoch_wall_seconds']},
            'checkpoint_IO_seconds_sum':sum(x['checkpoint_io_seconds'] for x in results),
            'new_LN_test_seconds_sum':sum(x['elapsed_seconds'] for x in ln),
            'retry_attempts':sum(x['exit_code']!=0 for x in attempts),'retry_attempt_seconds':sum(x['finished']-x['started'] for x in attempts if x['exit_code']!=0),
            'old_None_elapsed_seconds_sum':sum(x['elapsed_seconds'] for x in baseline if x['elapsed_seconds'] is not None),
            'old_None_elapsed_missing':sum(x['elapsed_seconds'] is None for x in baseline),
            'old_None_segment_timing_missing':sum(x['stage_timing'] is None for x in baseline),
            'old_None_peak_memory_missing':sum(x['peak_gpu_memory_allocated_bytes'] is None for x in baseline),
            'historical_None_gpu_resources':read_json(root/'resources.json')['original'],
            'current_gpu_resources':read_json(root/'resources.json')['GPUs'],
            'wall_seconds_to_report':time.time()-start,'report_compute_seconds':time.perf_counter()-before,
            'initialization_check_and_task_overhead_seconds':sum(x['task_wall_seconds']-x['training_seconds'] for x in execution),
            'queueing_note':'all tasks enqueue after freeze; queue time included in wall; worker between-task idle shown separately',
            'queue_wait_seconds_sum_over_tasks':sum(x['queue_wait_seconds'] for x in execution),
            'queue_wait_seconds_max':max(x['queue_wait_seconds'] for x in execution),
            'cost_limitation':'historical None and current LN were not simultaneous controls; missing old timing is not imputed; sums are not wallclock'}
    write_json(root/'TIMING_REPORT.json',timing)
    summary={'protocol_version':VERSION,'configurations':configs,'aggregates':aggregates,'validation_decision':read_json(root/'VALIDATION_DECISION.json'),
             'C3':next(x for x in configs if x['method']=='C' and x['group']==3),'timing':timing,
             'historical_development_benchmark':True,'baseline_test_previously_observed':True,'None_new_training':0,'None_new_test_scoring':0}
    write_json(root/'summary.json',summary)
    text='# 十个冻结冠军：历史 None 与新训 LayerNorm 配对结果\n\n'
    text+='恰好 300 次新 LN 训练，复用 300 条原 final None；无新 None 训练/评分，无 NAS/GP/VAE。每配置原 30 个 final seeds，LN 从头训练，仅 validation 选择 checkpoint。\n\n'
    text+='|配置|None validation|LN validation|差(pp)|None test|LN test|差(pp)|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for c in configs:
        text+=f"|{c['method']}{c['group']}|{100*c['None_validation_mean']:.4f}%|{100*c['LN_validation_mean']:.4f}%|{c['delta_validation_pp_mean']:+.4f}|{100*c['None_test_mean']:.4f}%|{100*c['LN_test_mean']:.4f}%|{c['delta_test_pp_mean']:+.4f}|\n"
    text+='\n|汇总|None validation|LN validation|差(pp)|None test|LN test|差(pp)|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m in ('A','C','overall'):
        v=aggregates['validation'][m];t=aggregates['test'][m]
        text+=f"|{m}|{100*v['None_mean']:.4f}%|{100*v['LN_mean']:.4f}%|{v['delta_pp_mean']:+.4f}|{100*t['None_mean']:.4f}%|{100*t['LN_mean']:.4f}%|{t['delta_pp_mean']:+.4f}|\n"
    text+='\nvalidation 工程建议（LN test 前已封存）：'+summary['validation_decision']['recommendation']+'。受损配置：'+json.dumps(summary['validation_decision']['damaged_configurations'],ensure_ascii=False)+'。不自动启动固定 LN NAS。\n\n'
    for m in ('A','C','overall'):
        text+=f"{m}：validation 描述95%区间 {aggregates['validation'][m]['descriptive_bootstrap95_pp']} pp；test 区间 {aggregates['test'][m]['descriptive_bootstrap95_pp']} pp。\n\n"
    text+=f"新增 LN 训练累计 {timing['new_LN_training_elapsed_seconds_sum']/3600:.4f} 小时；本轮至报告墙钟 {timing['wall_seconds_to_report']/60:.2f} 分钟；重试 {timing['retry_attempts']} 次。旧 None 累计训练耗时仅历史参照，未计入新增成本，旧分阶段计时缺失 {timing['old_None_segment_timing_missing']} 项。成本详情、逐epoch分段、显存与各GPU活跃/空闲见 TIMING_REPORT 和任务记录。\n\n"
    text+='共同初始参数及训练前 CPU/CUDA/Python/NumPy RNG 从冻结 None 代码按原 seed 重建并逐项比较；旧任务没有保存真实初始化哈希，不能声称历史训练当时已逐对核验。无旧权重 warm start。\n\n'
    text+='None 是历史复用、LN 是本轮新训。旧 test 及训练 seeds 已有开发暴露，本轮不是独立盲测或同期随机对照。bootstrap 仅五 group 块，A/C 同块重采样，不把300对当300次NAS重复。结果只说明十个固定配置在原HP下的LN效果，不代表LN重新调HP或固定LN NAS收益；负结果也不排除重新调HP后的收益。旧图表示及无入边fallback保持。\n'
    (root/'RESULTS.md').write_text(text)
    (root/'GPT_ANALYSIS_REQUEST.txt').write_text('请核验原final CSV基线与300对严格键连接、原HP/recipe/数据、LN位置与初始化重建证据、300训练test=0及模型/validation封存后的strict layer_norm评分。分析validation主判断、test方向、A/C/整体、C3、配置依赖、五group块bootstrap和新增/历史成本，保留历史暴露与非同期限制，不自动追加NAS。\n')

def accept(root):
    verify(root,baseline_weights=True)
    ts=tasks(root);pairs=read_json(root/'PAIR_MANIFEST.json')['pairs'];scores=read_json(root/'LN_SEED_RESULTS.json')
    barrier=read_json(root/'LN_TRAINING_SEAL.json');decision=read_json(root/'VALIDATION_DECISION.json')
    if len(ts)!=300 or len(pairs)!=300 or len(scores)!=300 or len(barrier['models'])!=300:
        raise ValueError('Expected 300 LN/None pair records')
    with connect(root) as db:
        if db.execute("SELECT count(*) FROM tasks WHERE status='succeeded'").fetchone()[0]!=300 or db.execute('SELECT count(*) FROM tasks').fetchone()[0]!=300:
            raise ValueError('LN training budget not complete')
    original={c['source_candidate_id']:c['configuration'] for c in read_json(root/'INPUT_MANIFEST.json')['champions']}
    mapping={x['task_id']:x for x in barrier['models']}
    for t in ts:
        v=result(root,t);f=root/'training'/t['task_id'];init=read_json(f/'INITIALIZATION_CHECK.json')
        if {**t['configuration'],'normalization':'none'}!=original[t['source_candidate_id']] or t['component_variant']!='layer_norm' or v['warm_start']:
            raise ValueError('LN condition/HP/architecture/warm start mismatch')
        if not init['shared_parameters_equal'] or not init['pretraining_rng_equal'] or init['initialization_evidence']!='reconstructed_from_frozen_none_code':
            raise ValueError('Initialization evidence invalid')
        if sha256(f/'result.json')!=mapping[t['task_id']]['result_sha256']:
            raise ValueError('Sealed training result changed')
    if barrier['validation_decision_sha256']!=sha256(root/'VALIDATION_DECISION.json') or decision['LN_test_available_at_decision'] or read_json(root/'scoring_exit_code.json')!=0 or any(read_json(root/'worker_exit_codes.json')):
        raise ValueError('Invalid validation/scoring boundary or worker exit')
    for s in scores:
        if s['computed_at']<barrier['sealed_at'] or s['variant']!='layer_norm' or not s['strict_checkpoint_load'] or s['training_updates'] or s['None_score_computed'] or s['dtype']!='float32':
            raise ValueError('Invalid LN-only post-seal scoring')
        if s['checkpoint_sha256']!=mapping[s['task_id']]['checkpoint_sha256']:
            raise ValueError('Scoring/model hash mismatch')
    receipt={'checks':'PASS','at':now(),'historical_None_reused':300,'new_LN_training':300,'paired_results':300,'condition_rows':600,
             'None_new_training':0,'None_new_test_scoring':0,'LN_training_test_evaluations':0,'LN_all_models_sealed_before_test':True,
             'validation_recommendation_sealed_before_LN_test':True,'LN_variant_strict_scoring':True,'warm_start':False,'source_unchanged':True,
             'H_accessed':False,'NAS_started':False,'historical_development_benchmark':True,'initialization_evidence':'reconstructed_from_frozen_none_code'}
    write_json(root/'ACCEPTANCE.json',receipt)
    (root/'ACCEPTANCE_REPORT.md').write_text('# 验收：PASS\n\n'+json.dumps(receipt,ensure_ascii=False,indent=2)+'\n\n详见 RESULTS.md、BASELINE_MANIFEST、初始化记录、LN_TRAINING_SEAL、VALIDATION_DECISION 与所有配对CSV。None历史复用，LN新训；不能称同期随机对照或独立test确认。\n')
    return receipt

def package(root):
    archive=root/'EVIDENCE.tar.gz';tmp=root/('EVIDENCE.'+str(os.getpid())+'.tmp')
    with tarfile.open(tmp,'w:gz') as tar:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p==archive or p==tmp or p.name=='package_manifest.json' or p.suffix in ('.pt','.pth','.sqlite','.log','.lock','.tmp') or p.name.endswith(('-wal','-shm')):continue
            tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    tmp.replace(archive)
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if member.isfile():
                f=tar.extractfile(member)
                while f.read(1024*1024):continue
    write_json(root/'package_manifest.json',{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive),'at':now(),
               'archive_full_read':'PASS','checkpoints_retained_on_server':True})
