import math
import time
import tarfile
import statistics
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from .common import read,write,db,now,sha,csv_write,TERMINAL,ZERO,digest

def timestring(unix):
    dt=datetime.fromtimestamp(unix,timezone.utc)
    return {'UTC':dt.isoformat(),'Asia_Shanghai':dt.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %z')}
def results(root):
    with db(root) as c:rows=c.execute('SELECT * FROM runs ORDER BY ord').fetchall()
    return [(dict(row),read(root/'runs'/row['id']/'result.json') if (root/'runs'/row['id']/'result.json').exists() else None) for row in rows]

def update_eta(root):
    all_rows=results(root);done=[x for row,x in all_rows if x and row['state'] in TERMINAL]
    if len(done)<4:return
    estimates=[];seconds=0.;low=0.;high=0.
    for f in ['A','WI']:
        for n in [50,150,250]:
            left=sum(row['state'] not in TERMINAL and __import__('json').loads(row['spec'])['family']==f and __import__('json').loads(row['spec'])['n']==n for row,x in all_rows)
            measured=[x['elapsed_seconds'] for x in done if x['family']==f and x['n']==n]
            other=[x for x in done if x['family']==f]
            if measured:
                mean=statistics.mean(measured);lo=min(measured);hi=max(measured);source='same family and n formal run timings'
            elif other:
                ref=statistics.mean(x['elapsed_seconds'] for x in other);refn=statistics.mean(x['n'] for x in other)
                mean=min(300.,ref*max(1.,n/refn));lo=ref;hi=300.;source='unmeasured n: linear n growth assumption, 300s hard upper bound'
            else:mean=150.;lo=0.;hi=300.;source='no family timing yet; hard 300s ceiling'
            mean=min(300.,mean);seconds+=left*mean;low+=left*lo;high+=left*hi
            estimates.append({'family':f,'n':n,'remaining':left,'measured_runs':len(measured),'seconds_per_run':mean,'range_per_run':[lo,hi],'source':source})
    remaining=seconds/2+60;at=time.time();eta={'at':now(),'terminal':len(done),'remaining':36-len(done),'workers':2,'phase_estimates':estimates,
       'estimated_completion':timestring(at+remaining),'completion_range':[timestring(at+low/2+30),timestring(at+high/2+60)],
       'remaining_wall_seconds':remaining,'report_allowance_seconds':60,'assumptions':['two independent CPU numerical processes','prep excluded and reported separately',
       'future n timings projected until actually measured','300s ceiling is per-run algorithm time; queue/setup/cleanup may add overhead'],
       'first_four':[{'id':x['id'],'seconds':x['elapsed_seconds'],'state':x['state']} for x in done[:4]]}
    write(root/'ETA.json',eta)
    if not (root/'first_acceptance_schedule.json').exists():
        write(root/'first_acceptance_schedule.json',{'created_at':now(),'due':timestring(at+remaining+900),'due_unix':at+remaining+900,
                  'buffer_seconds':900,'original_eta':eta,'not_silently_moved':True,'healthy_repeat_seconds':300})

def final_report(root):
    rows=[x for row,x in results(root) if x];pairs=[];decision={};flat=[]
    for x in rows:
        r={k:v for k,v in x.items() if k not in ['post_candidates','returned','milestones','gradient_starts']}
        r['returned_candidate_id']=x['returned']['candidate_id'] if x.get('returned') else None
        r['returned_configuration']=x['returned']['configuration'] if x.get('returned') else None
        r['returned_u_float64']=x['returned']['u_float64'] if x.get('returned') else None
        r['returned_z_float32']=x['returned']['z_float32'] if x.get('returned') else None
        r['post_legal_rate']=x.get('post_valid',0)/x['post_requests'] if x.get('post_requests') else None
        r['inner_legal_rate']=x.get('inner_valid',0)/x['requests'] if x.get('requests') and x['family']=='WI' else None
        flat.append(r)
    for f,refmethod in [('A','GRADIENT'),('WI','RANDOM')]:
        family_pairs=[]
        for n in [50,150,250]:
            for r in range(3):
                de=next(x for x in rows if x['family']==f and x['n']==n and x['replicate']==r and x['method']=='DE')
                ref=next(x for x in rows if x['family']==f and x['n']==n and x['replicate']==r and x['method']==refmethod)
                valid=de['state']=='complete' and de['requests']==512 and de.get('returned') is not None and ref['state'] in ['complete','complete_early'] and ref.get('returned') is not None
                scale=de.get('frozen_y_scale',ref.get('frozen_y_scale'));eps=max(1e-12,1e-8*scale) if scale else None
                d=de.get('improvement_approx_validation');v=ref.get('improvement_approx_validation')
                pair={'family':f,'n':n,'replicate':r,'DE_run':de['id'],'reference_run':ref['id'],'reference_method':refmethod,
                 'DE_state':de['state'],'reference_state':ref['state'],'valid_pair':valid,'eps':eps,
                 'DE_realized_qLogEI':de.get('realized_best'),'reference_realized_qLogEI':ref.get('realized_best'),
                 'DE_improvement_approx':d,'reference_improvement_approx':v,'positive_beyond_eps':False,
                 'uninformative':False,'improvement_ratio':None,'elapsed_ratio':None}
                if valid:
                    if de['MC_base_tensor_hash']!=ref['MC_base_tensor_hash'] or de['frozen_y_scale']!=ref['frozen_y_scale']:raise ValueError('Paired frozen target differs')
                    pair.update(improvement_difference=d-v,positive_beyond_eps=d-v>eps,uninformative=d<=eps and v<=eps,
                      improvement_ratio=(d+eps)/(v+eps),elapsed_ratio=de['elapsed_seconds']/ref['elapsed_seconds'])
                family_pairs.append(pair);pairs.append(pair)
        valid=[x for x in family_pairs if x['valid_pair']];wins=sum(x['positive_beyond_eps'] for x in valid);uninformative=sum(x['uninformative'] for x in valid)
        ratio=statistics.median(x['improvement_ratio'] for x in valid) if valid else None
        tratio=statistics.median(x['elapsed_ratio'] for x in valid) if valid else None
        checks={'all_9_pairs_valid_complete':len(valid)==9,'positive_beyond_eps_at_least_6_of_9':wins>=6,
             'median_improvement_ratio_at_least_1_10':ratio is not None and ratio>=1.10,'median_elapsed_ratio_at_most_2_0':tratio is not None and tratio<=2.}
        state='FAIL_DE' if len(valid)!=9 else 'INCONCLUSIVE' if uninformative>=5 else 'PASS_DE' if all(checks.values()) else 'FAIL_DE'
        decision[f]={'decision':state,'checks':checks,'valid_pairs':len(valid),'positive_pairs':wins,'uninformative_pairs':uninformative,
                     'median_improvement_ratio':ratio,'median_elapsed_ratio':tratio}
    csv_write(root/'run_results.csv',flat);csv_write(root/'paired_results.csv',pairs);write(root/'decision.json',decision)
    totals={k:sum(x.get('times',{}).get(k,0.) for x in rows) for k in ['proposal','decoder','WL','input_transform','posterior_qLogEI','backward','postprocessing','file_IO']}
    costs={**ZERO,'runs':len(rows),'state_counts':{s:sum(x['state']==s for x in rows) for s in TERMINAL},
          'total_inner_requests':sum(x['requests'] for x in rows),'total_post_requests':sum(x['post_requests'] for x in rows),
          'total_numerical_run_seconds':sum(x['elapsed_seconds'] for x in rows),'stage_seconds_nonoverlapping':totals,
          'unattributed_algorithm_overhead_seconds':sum(max(0.,x['elapsed_seconds']-sum(x.get('times',{}).values())) for x in rows),
          'common_sanity_seconds':read(root/'sanity_report.json')['elapsed_seconds'],
          'worker_preparation_events':[__import__('json').loads(line) for line in (root/'preparation_events.jsonl').read_text().splitlines()],
          'zero_work_evidence':{'optimizer_steps_prohibited_at_runtime':True,'GNN_evaluators_prohibited_at_runtime':True,
          'no_dataset_loader_or_scoring_entry_called':True,'each_valid_run_GP_parameter_hash_unchanged':all(x.get('GP_parameter_hash_before')==x.get('GP_parameter_hash_after') for x in rows if x['state'] in ['complete','complete_early'])}}
    write(root/'COSTS.json',costs)
    text='# 冻结代理采集优化器离线比较\n\n'
    text+=f"36 run执行记录：{costs['state_counts']}。仅组0，三个相关GP阶段、三个优化随机重复；不是9次独立NAS。\n\n"
    text+='|family|结论|有效配对|DE正收益配对|改进量比中位数|耗时比中位数|\n|---|---|---:|---:|---:|---:|\n'
    for f,d in decision.items():text+=f"|{f}|{d['decision']}|{d['valid_pairs']}/9|{d['positive_pairs']}/9|{d['median_improvement_ratio']}|{d['median_elapsed_ratio']}|\n"
    text+='\nI=y_scale×exp(realized qLogEI)，是冻结代理标准化y单位下平滑MC改进量的近似，不是新增validation/test精度；不跨A/WI比较绝对qLogEI。判据及每个配对见decision.json与paired_results.csv。\n\n'
    text+='A内层优化共同连续RAW16松弛，最终输出使用CPU float32实际latent和原HP量化后的重评分数。松弛与落地差见run_results.csv。W-I使用真实解码配置和新旧全部WL符号的精确距离。\n\n'
    text+='W-I组0 n50的架构长度尺度约12985.9147，该阶段核对架构变化很不敏感，收益可能主要来自HP，不能全部归于架构优化。沿用原记录图与执行fallback表示限制。\n\n'
    text+=f"GNN训练、GP参数更新、新validation观测、新test/H计算均为0。共{costs['total_inner_requests']}次内层向量请求，{costs['total_post_requests']}次后处理请求；累计run耗时{costs['total_numerical_run_seconds']:.3f}s，准备/阶段耗时见COSTS.json。\n\n"
    text+='只有全部36个run完整有效才称完整成功。TIMEOUT/NO_VALID_RETURN/FAILED不补跑、不换种子。优化器筛选结论不是实测性能提升或小样本显著性结论；不自动启动开放ABC。\n'
    (root/'RESULTS.md').write_text(text)
    (root/'GPT_ANALYSIS_REQUEST.md').write_text('请结合PROTOCOL、六模型manifest、sanity_report、36run完整访问档案、18个配对、decision及COSTS审阅两family的工程判据。区分A松弛与实际量化落地，保留失败/超时与零训练边界；W-I n50架构长度尺度约1.30e4。不要横比两family的绝对采集值，不把平滑MC改进量当实测准确率，不宣称9次独立NAS或小样本显著性，不追加实验。\n')

def accept(root):
    from .checks import verify
    verify(root);rows=results(root);specs=read(root/'PROTOCOL.json')['order']
    if len(rows)!=36 or any(row['state'] not in TERMINAL or x is None for row,x in rows):raise ValueError('36 runs not fully accounted')
    manifests={(x['family'],x['n']):x for x in read(root/'model_manifest.json')}
    for row,x in rows:
        if x['id']!=row['id'] or x['requests']>512 or x['post_requests']>32 or any(x.get(k)!=0 for k in ZERO):raise ValueError('Budget/zero-work mismatch')
        if x['state'] in ['complete','complete_early']:
            if x['elapsed_seconds']>300 or not x.get('returned'):raise ValueError('Completion beyond deadline or missing return')
            if x['method'] in ['DE','RANDOM'] and x['requests']!=512:raise ValueError('Fixed optimizer budget incomplete')
            if x['GP_parameter_hash_before']!=x['GP_parameter_hash_after']:raise ValueError('GP changed during optimization')
            m=manifests[(x['family'],x['n'])];ret=x['returned']
            if ret['candidate_id'] in set(m['training_order']) or digest(ret['configuration'])!=ret['candidate_id']:raise ValueError('Returned seen/incorrect configuration')
            if ret['vector_key'] not in {v['vector_key'] for v in x['post_candidates']} or not math.isfinite(ret['score']):raise ValueError('Return not from own finite postprocessing')
            from analyse.structured_kernel_phase1.features import build_operator_graph
            build_operator_graph(ret['configuration'])
            if x['MC_base_tensor_hash']!=m['MC_base_tensor_hash']:raise ValueError('Fixed MC target changed')
        trace=root/'runs'/row['id']/'trace.jsonl'
        if x['requests'] and not trace.exists():raise ValueError('Missing partial/full trace')
    for worker in [0,1]:
        if read(root/f'worker{worker}_exit.json')['exit_code']!=0:raise ValueError('Supervisor exit failure')
    allvalid=all(x['state'] in ['complete','complete_early'] for row,x in rows)
    receipt={'at':now(),'state':'PASS' if allvalid else 'EXECUTION_COMPLETE_WITH_FAILURES','all_36_valid_complete':allvalid,'runs_accounted':36,'pairs':18,
        'state_counts':read(root/'COSTS.json')['state_counts'],'readonly_sources_unchanged':True,'common_sanity':'PASS','budget_and_provenance':'PASS',**ZERO}
    write(root/'acceptance_report.json',receipt);package(root);return receipt

def package(root):
    path=root/'analysis_package.tar.gz';tmp=root/'analysis_package.tar.gz.tmp'
    with tarfile.open(tmp,'w:gz') as t:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p==path or p==tmp or p.name=='package_manifest.json' or p.suffix in ['.lock','.sqlite','.log','.pt','.pth','.tmp'] or 'sanity_budget' in p.parts:continue
            if '-wal' in p.name or '-shm' in p.name:continue
            t.add(p,arcname=str(p.relative_to(root)),recursive=False)
        for p in sorted(Path('analyse/acq_optimizer_offline').glob('*')):
            if p.suffix in ['.py','.sh']:t.add(p,arcname='code/'+p.name,recursive=False)
    tmp.replace(path);write(root/'package_manifest.json',{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'weights_included':False,'at':now()})

def acceptance_loop(root):
    write(root/'acceptance_status.json',{'state':'WAITING','pid':__import__('os').getpid(),'at':now()});next_due=None
    while True:
        if (root/'FAILED.json').exists():
            write(root/'acceptance_status.json',{'state':'FAILED','at':now(),'failure':read(root/'FAILED.json')});return 1
        if (root/'COMPLETE.json').exists():
            receipt=read(root/'COMPLETE.json')['acceptance'];receipt['trigger']='pipeline completion event'
            write(root/'acceptance_report.json',receipt);write(root/'acceptance_status.json',{'state':receipt['state'],'at':now()});package(root);return 0
        p=root/'first_acceptance_schedule.json'
        if next_due is None and p.exists():
            next_due=read(p)['due_unix'];write(root/'acceptance_status.json',{'state':'WAITING_UNTIL_DUE','due':timestring(next_due),'pid':__import__('os').getpid(),'at':now()})
        if next_due is not None and time.time()>=next_due:
            status=read(root/'status.json')
            if status.get('state')=='RUNNING' and time.time()-status.get('heartbeat',0)>120:
                write(root/'FAILED.json',{'reason':'controller heartbeat stale','at':now()});continue
            write(root/'acceptance_status.json',{'state':'HEALTHY_INCOMPLETE','next_check':timestring(time.time()+300),'status':status,'at':now()});next_due=time.time()+300
        time.sleep(2)
