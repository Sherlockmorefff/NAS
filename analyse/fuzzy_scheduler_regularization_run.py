"""Sequential isolated inner validation, repaired fits, and historical oracle."""
from __future__ import annotations
import argparse,fcntl,os,platform,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_scheduler_regularization_core import *
from analyse.fuzzy_convergence_run import verify as verify_previous,analytic_gp,finish
from analyse.fuzzy_loss_interaction_run import oracle_request,copy_file
OLD=ROOT/'results/posthoc/fuzzy-convergence-v1-20260921T043054Z'
GRID=[0.,1e-5,1e-4,1e-3]
NEW=['analyse/fuzzy_scheduler_regularization_'+n+'.py' for n in ['core','run']]+['tests/test_fuzzy_scheduler_regularization.py','configs/fuzzy_scheduler_regularization_v1.json']
CASES=[('mapping','replay-start2026091903-W','mapping/known30'),('mapping','replay-start2026091900-W','mapping/known40'),('mapping','replay-start2026091901-C','mapping/known40'),('gp','offline-f2-s1-F','gp-FIXED'),('gp','offline-f0-s0-W','gp-ADAPTIVE'),('gp','offline-f3-s2-C','gp-ADAPTIVE')]


def split_inner(features,ids,seed):
    arch=dict(zip(features['ids'],features['architectures']));groups={a:sorted(c for c in ids if arch[c]==a) for a in sorted({arch[c] for c in ids})}
    order=np.random.default_rng(seed).permutation(sorted(groups)).tolist();folds=[[] for _ in range(3)]
    for a in order:folds[min(range(3),key=lambda i:(len(folds[i]),i))].extend(groups[a])
    rows=[dict(inner_fold=i,validation_ids=sorted(v),train_ids=sorted(set(ids)-set(v))) for i,v in enumerate(folds)]
    sufficient=all(len(r['validation_ids'])>=3 and len({arch[c] for c in r['validation_ids']})>=2 and len(r['train_ids'])>=8 and len({arch[c] for c in r['train_ids']})>=4 for r in rows)
    record=dict(seed=seed,known_ids=sorted(ids),architecture_order=order,folds=rows,sufficient=sufficient)
    record['identity']=digest(record);return record


def choose_strength(scores,sufficient=True):
    if not sufficient:return dict(lambda_w=0.,status='INNER_VALIDATION_INSUFFICIENT',deciding_metric='predeclared_fallback',aggregates=[])
    agg=[]
    for w in GRID:
        rows=[r for r in scores if r['lambda_w']==w]
        valid=len(rows)==3 and all(r['valid'] for r in rows)
        agg.append(dict(lambda_w=w,valid=valid,**({k:float(np.mean([r[k] for r in rows])) for k in ['r2','r1','mae']} if valid else {})))
    viable=[r for r in agg if r['valid']];trace=[]
    if not viable:return dict(lambda_w=0.,status='TUNING_FAILED_FALLBACK_ZERO',deciding_metric='predeclared_fallback',aggregates=agg)
    for key in ['r2','r1','mae']:
        best=min(r[key] for r in viable);survive=[r for r in viable if r[key]-best<=1e-12]
        trace.append(dict(metric=key,best=best,before=[r['lambda_w'] for r in viable],remaining=[r['lambda_w'] for r in survive]))
        viable=survive
        if len(viable)==1:return dict(lambda_w=viable[0]['lambda_w'],status='SELECTED',deciding_metric=key,aggregates=agg,decision_trace=trace)
    return dict(lambda_w=min(r['lambda_w'] for r in viable),status='SELECTED',deciding_metric='smaller_lambda_w',aggregates=agg,decision_trace=trace)


def verify(o,private=False):
    p=read(o/'protocol.json');assert digest({k:v for k,v in p.items() if k!='identity'})==p['identity']
    effective=dict(p['sources'])
    if (o/'instrumentation_amendment.json').exists():
        amendment=read(o/'instrumentation_amendment.json');assert amendment['original_protocol']==p['identity']
        assert digest({k:v for k,v in amendment.items() if k!='identity'})==amendment['identity']
        for n,change in amendment['sources'].items():
            assert n=='analyse/fuzzy_scheduler_regularization_run.py' and change['original']==p['sources'][n]
            effective[n]=change['current']
    for n,h in effective.items():assert file_hash(ROOT/n)==h,n
    for n,h in p['inputs'].items():assert file_hash(o/n)==h,n
    if private:
        for n,h in p['private_inputs'].items():assert file_hash(o/n)==h,n
    return p


def prepare(o,prompt):
    assert not (o/'protocol.json').exists();o.mkdir(parents=True,exist_ok=True)
    prev=verify_previous(OLD);workspace=dict(hostname=platform.node(),project=str(ROOT),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),dirty=subprocess.check_output(['git','status','--short'],text=True),python=sys.version,torch=torch.__version__,gpytorch=gpytorch.__version__,numpy=np.__version__,device='cpu',threads=1,time=utc())
    write(o/'workspace_before.json',workspace);(o/'git_diff_before.patch').write_text(subprocess.check_output(['git','diff','--binary'],text=True))
    for n in ['features.json','candidate_identity.json','folds.json','starts.json','data_checks.json','recipe_verified.json','protected_before.json','private/evaluation_A.json']:copy_file(OLD/n,o/n)
    assert file_hash(o/'private/evaluation_A.json')==prev['private_A_sha']
    for n in ['REQUEST.md','REPORT.md','protocol.json','analysis_amendment.json','source_audit.md','reuse_manifest.json','execution_manifest.json']:copy_file(OLD/n,o/'reference_snapshot'/n)
    for n in ['AGENTS.md','README.md','docs/repository_layout.md','docs/experiment_artifact_guide.md']:copy_file(ROOT/n,o/'reference_snapshot'/n)
    copy_file(prompt,o/'REQUEST.md');copy_file(ROOT/'configs/fuzzy_scheduler_regularization_v1.json',o/'config.json')
    features=read(o/'features.json');assert len(features['ids'])==179 and len(set(features['architectures']))==96
    oldtasks={t['name']:t for t in read(OLD/'tasks.json')};tasks=[];splits={};contexts=[]
    # Materialize only the task's legal observations, never a supervised warm start.
    for seed in range(3):
        name=f'initial_weights/seed{seed}.pt';copy_file(OLD/f'initial_weights/f0-s{seed}.pt',o/name)
        st=load(o/name);assert st['untrained'] and st['model_hash']==tree_hash(JointMetricModel(12,seed=seed).state_dict())
    for fold in read(o/'folds.json'):
        f=fold['fold'];context=f'offline-f{f}';lp=f'legal_labels/{context}.json';copy_file(OLD/f'train_labels/f{f}-s0.json',o/lp)
        contexts.append(dict(context=context,stage='offline',fold=f,seed=2026092100+f,label_file=lp,train_ids=fold['train_ids'],query_ids=fold['query_ids'],epsilon=fold['epsilon'],lambda_F=fold['lambda_F']))
    for seed,start in read(o/'starts.json').items():
        context=f'replay-start{seed}';lp=f'legal_labels/{context}.json';copy_file(OLD/f'initial_labels/start{seed}.json',o/lp)
        contexts.append(dict(context=context,stage='replay',search_seed=int(seed),seed=int(seed),label_file=lp,train_ids=sorted(start['initial_ids']),initial_ids=start['initial_ids'],epsilon=.01,lambda_F=.1))
    for ctx in contexts:
        split=split_inner(features,ctx['train_ids'],ctx['seed']);splits[ctx['context']]=split;labels=read(o/ctx['label_file'])
        for row in split['folds']:
            lp=f"inner_labels/{ctx['context']}-i{row['inner_fold']}.json";write(o/lp,{c:labels[c] for c in row['train_ids']})
            for group in ['C','F']:
                for wi,w in enumerate(GRID):
                    tasks.append(dict(name=f"inner-{ctx['context']}-{group}-i{row['inner_fold']}-w{wi}",stage='inner_cv',purpose='offline_inner' if ctx['stage']=='offline' else 'replay_inner',context=ctx['context'],group=group,inner_fold=row['inner_fold'],lambda_w=w,lambda_f=0. if group=='C' else ctx['lambda_F'],epsilon=ctx['epsilon'],model_seed=0,weight_file='initial_weights/seed0.pt',label_file=lp,train_ids=row['train_ids'],query_ids=row['validation_ids'],sufficient=split['sufficient']))
        for group in ['C','F']:
            for flow in ['REPAIRED','REPAIRED_SELECT_REG']:
                for seed in (range(3) if ctx['stage']=='offline' else [0]):
                    t={k:v for k,v in ctx.items() if k not in ['seed']};t.update(name=f"{ctx['context']}-{group}-{flow}-s{seed}",purpose='offline_formal' if ctx['stage']=='offline' else 'replay_formal',group=group,protocol_type=flow,model_seed=seed,weight_file=f'initial_weights/seed{seed}.pt',lambda_f=0. if group=='C' else ctx['lambda_F'])
                    tasks.append(t)
    cases=[]
    for i,(kind,name,rel) in enumerate(CASES):
        oldtask=oldtasks[name];src=OLD/'tasks'/name;origin=src/rel;initial=load(origin/'post000.pt');labels=read(OLD/oldtask['label_file'])
        if kind=='mapping':
            q=int(rel[-2:])
            for b in range(q//5):labels.update(read(src/'batches'/f'b{b:02}'/'reveal.json')['labels'])
        assert sorted(labels)==initial['train_ids'];lp=f'engineering_inputs/case{i}/labels.json';write(o/lp,labels)
        for f in ['post000.pt','latest.pt','summary.json','post100.pt' if kind=='mapping' else 'post020.pt']:copy_file(origin/f,o/f'engineering_inputs/case{i}'/f)
        t=dict(name=f'engineering-case{i}',stage='engineering',purpose='engineering',kind=kind,group=oldtask['group'],model_seed=oldtask['model_seed'],epsilon=oldtask['epsilon'],lambda_f=oldtask['lambda_f'],lambda_w=0.,label_file=lp,original_stage=oldtask['stage'],original_path=str(origin),input_directory=f'engineering_inputs/case{i}',train_ids=sorted(labels))
        cases.append(t)
    tasks=cases+tasks;write(o/'tasks.json',tasks);write(o/'contexts.json',contexts);write(o/'inner_splits.json',splits)
    # Only C/F old ADAPTIVE and original RANDOM are imported; no old W formal cells.
    reuse=[]
    for t in oldtasks.values():
        if t['group'] not in ['C','F']:continue
        src=OLD/'tasks'/t['name'];done=read(src/'completed.json');assert done['protocol']==prev['identity']
        for n,h in done['files'].items():assert file_hash(src/n)==h,(src,n)
        wanted=[]
        if t['stage']=='offline':wanted=['predictions-ADAPTIVE-ADAPTIVE.json','mapping/latest.pt','mapping/post100.pt','gp-ADAPTIVE/latest.pt']
        else:
            wanted=['state.pt','initial_access.json']+[f'batches/b{b:02}/{n}' for b in range(10) for n in ['selection.json','reveal.json','gp/latest.pt']]
        for n in wanted:copy_file(src/n,o/'imports'/t['name']/n)
        reuse.append(dict(task=t,source=str(src),protocol='OLD_ADAPTIVE',source_protocol=prev['identity'],source_manifest_sha=file_hash(src/'completed.json'),files={n:file_hash(src/n) for n in wanted},additional_updates=0))
    for r in read(OLD/'reuse_manifest.json')['trajectories']:
        if r['task']['group']!='RANDOM':continue
        t=r['task'];src=OLD/'imports'/t['name'];wanted=['initial_access.json']+[f'batches/b{b:02}/{n}' for b in range(10) for n in ['selection.json','reveal.json']]
        for n in wanted:assert file_hash(src/n)==r['files'][n];copy_file(src/n,o/'imports'/t['name']/n)
        reuse.append(dict(task=t,source=str(src),protocol='RANDOM',source_protocol=r['source_protocol'],files={n:file_hash(src/n) for n in wanted},additional_updates=0))
    write(o/'reuse_manifest.json',reuse)
    public=['features.json','candidate_identity.json','folds.json','starts.json','data_checks.json','recipe_verified.json','REQUEST.md','config.json','tasks.json','contexts.json','inner_splits.json']+[f'initial_weights/seed{s}.pt' for s in range(3)]
    private=[str(f.relative_to(o)) for d in ['legal_labels','inner_labels','engineering_inputs','private','imports'] for f in (o/d).rglob('*') if f.is_file()]
    sources=set(prev['sources'])|set(NEW)
    for n in sources:copy_file(ROOT/n,o/'source_snapshot'/n)
    p=dict(version='fuzzy-scheduler-regularization-v1',previous_protocol=prev['identity'],previous_source=str(OLD),sources={n:file_hash(ROOT/n) for n in sorted(sources)},inputs={n:file_hash(o/n) for n in public},private_inputs={n:file_hash(o/n) for n in private},created_at=utc(),formal_updates_before_seal=0,config=read(o/'config.json'))
    p['identity']=digest(p);write(o/'protocol.json',p,immutable=True)
    (o/'source_audit.md').write_text('服务器当前代码逐项通过 evidence9 来源清单及其两文件复算修订校验。沿用实际标准化：z、HP、y均由训练部分拟合，未新增变换。新隔离模块复制原fit循环，仅修改调度、显式参数项、检查项与记录。旧核心及结果不改。\n保护期120/60，末步不降；L0与原加权项使用自己的尺度，新总损失和参数加权项使用新尺度。2337个训练参数含bias，Adam weight_decay=0。inner_cv显式使用训练ID作停止集合。分组种子只含外折/起点，选择种子0；外层0/1/2。旧模型仅作工程样例和事后参照。\n工程起始状态精确来自post000，known30/40为新增计数；GP案例保持原固定表示。内选与新回放均从未训练初始化开始。独立子进程白名单实际拒绝全A/其他任务/验证标签。候选池A反复开发，不是独立确认。\n')
    print(json.dumps(dict(prepared=str(o),tasks=len(tasks),protocol=p['identity'],inner_splits={k:[len(r['validation_ids']) for r in v['folds']] for k,v in splits.items()})),flush=True)


def install_guard(o,folder,task,p):
    import tempfile
    runtime=folder/'runtime_tmp';runtime.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(runtime)
    allowed={str((o/n).resolve()) for n in p['inputs']}|{str((o/task['label_file']).resolve()),str((o/'protocol.json').resolve()),str((o/'instrumentation_amendment.json').resolve())}
    if task['stage']=='engineering':allowed|={str((o/task['input_directory']/n).resolve()) for n in ['post000.pt','latest.pt','summary.json','post100.pt','post020.pt']}
    accesses=set();denied=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        own=path.is_relative_to(folder);ledger=path.is_relative_to(o/'ledger')
        if writing and not (own or ledger):raise PermissionError('write outside own task denied '+str(path))
        protected=any(path.is_relative_to(ROOT/n) for n in ['results','checkpoints','data','Cora'])
        if protected and not (str(path) in allowed or own or ledger):denied.append(str(path));raise PermissionError('hidden labels/SE or other trajectory denied '+str(path))
        if protected and not writing:accesses.add(str(path))
    sys.addaudithook(hook);return accesses,denied


def case_compare(folder,origin,kind,new):
    old=load(origin/'latest.pt');prefix='post100.pt' if kind=='mapping' else 'post020.pt'
    a=load(folder/prefix);b=load(origin/prefix)
    checks={k:tree_hash(a[k])==tree_hash(b[k]) for k in ['model','optimizer','rng']};assert all(checks.values()),checks
    nlr={h['step']:h['lr_used_for_update'] for h in new['history'][1:]};olr={h['step']:h['lr'] for h in old['history'][1:]}
    divergence=next((t for t in sorted(set(nlr)&set(olr)) if nlr[t]!=olr[t]),None)
    stop=divergence if divergence else min(new['completed_updates'],old['completed_updates'])+1
    maximum=max(abs(new['history'][t]['total']-old['history'][t]['total']) for t in range(stop));assert maximum==0,maximum
    for chk in old['check_history']:
        if chk['step']<stop:
            nc=next(c for c in new['check_history'] if c['step']==chk['step'])
            assert nc['gradient']==chk['gradient']
    row=dict(kind=kind,prefix_exact=checks,first_different_update=divergence,pre_divergence_loss_max_error=maximum,old_status=old['status'],new_status=new['status'],old_steps=old['completed_updates'],new_steps=new['completed_updates'],old_lr_changes=old['scheduler']['lr_changes'],new_lr_changes=new['scheduler']['lr_changes'],final_reason_flags=new['reason_flags'],old_final_three_windows=[d['valid'] for d in old['decisions'][-3:]],new_final_three_windows=[d['valid'] for d in new['decisions'][-3:]])
    assert all(row['new_final_three_windows']);write(folder/'comparison.json',row)



def finish(folder,p,task,accesses,denied,started):
    # Use this isolated core's counters, not the predecessor module's globals.
    write(folder/'access_boundary.json',dict(protected_reads=sorted(accesses),denied=denied,full_A_received=False,B_test_reads=0))
    write(folder/'timing.json',dict(TIMES,**PERF,task_seconds=time.perf_counter()-started,phase_counter_source='fuzzy_scheduler_regularization_core'))
    files={str(f.relative_to(folder)):file_hash(f) for f in folder.rglob('*') if f.is_file() and f.name!='completed.json' and not f.name.endswith('.log') and not f.is_relative_to(folder/'runtime_tmp')}
    write(folder/'completed.json',dict(protocol=p['identity'],task=task,files=files,time=utc()),immutable=True)


def worker(o,name):
    started=time.perf_counter();p=read(o/'protocol.json');task=next(t for t in read(o/'tasks.json') if t['name']==name);folder=o/'tasks'/name;folder.mkdir(parents=True,exist_ok=True)
    accesses,denied=install_guard(o,folder,task,p);verify(o)
    if (folder/'completed.json').exists():return
    blocked=[o/'private/evaluation_A.json',OLD/'private/evaluation_A.json',ROOT/'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2/oracle_validation.json',ROOT/'checkpoints/joint_model_pipeline_global4_best.pth']
    blocked += [o/n for n in p['private_inputs'] if n.startswith('legal_labels/') and n!=task['label_file']][:2]
    for path in blocked:
        try:path.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('hidden read allowed '+str(path))
    assert file_hash(o/task['label_file'])==p['private_inputs'][task['label_file']]
    labels=read(o/task['label_file']);features=read(o/'features.json');budget=Budget(o/'ledger',p['identity'])
    torch.manual_seed(task['model_seed']);np.random.seed(task['model_seed']);random.seed(task['model_seed'])
    meta={k:v for k,v in task.items() if k in ['stage','purpose','group','model_seed','protocol_type','fold','search_seed','context','inner_fold','epsilon']};meta['task']=name
    if task['stage']=='engineering':
        origin=o/task['input_directory'];initial=load(origin/'post000.pt');scaler=initial['scaler'];data=observed_data(features,scaler,labels,task['epsilon']);meta['stage']=task['original_stage'];restore_rng(initial['rng'])
        if task['kind']=='mapping':st=fit_mapping(folder/'fit',initial['model'],data,task['lambda_f'],budget,meta,lambda_w=0.)
        else:st=fit_gp(folder/'fit',initial['cache'],data,budget,meta)
        case_compare(folder/'fit',origin,task['kind'],st);finish(folder,p,task,accesses,denied,started);return
    weights=load(o/task['weight_file']);assert weights['untrained'] and weights['model_hash']==tree_hash(JointMetricModel(12,seed=task['model_seed']).state_dict())
    w=task['lambda_w'] if task['stage']=='inner_cv' else read(folder/'assignment.json')['lambda_w']
    meta['lambda_w']=w;scaler=scaler_from_training(features,sorted(labels),labels)
    if task['stage'] in ['offline','inner_cv']:
        data=observed_data(features,scaler,labels,task['epsilon']);mapped=fit_mapping(folder/'mapping',weights['model'],data,task['lambda_f'],budget,dict(meta,representation_version=0),lambda_w=w)
        model=reload_model(mapped['model']);model.eval();before=tree_hash(model.state_dict());cache=embedding_cache(model,data)
        gp=fit_gp(folder/'gp',cache,data,budget,dict(meta,representation_version=0));assert before==tree_hash(model.state_dict())==gp['mapping_hash_after']
        rows=predict(cache,data['ids'],task['query_ids'],gp)
        train_mu,_=gp_prediction(gp,data['ids']);pos={c:i for i,c in enumerate(cache['ids'])}
        train_rows=[dict(candidate_id=c,gp_mean=float(v),head_mean=float(cache['head'][pos[c]])) for c,v in zip(data['ids'],train_mu)]
        write(folder/'predictions.json',dict(rows=rows,train_rows=train_rows,train_ids=data['ids'],query_ids=task['query_ids'],mapping_hash=before,gp_hash=gp['model_hash'],sealed_at=utc(),lambda_w=w),immutable=True)
        finish(folder,p,task,accesses,denied,started);return
    if (folder/'state.pt').exists():state=load(folder/'state.pt');assert state['scaler']==scaler and state['lambda_w']==w;restore_rng(state['rng'])
    else:
        state=dict(protocol=p['identity'],known=labels,query_order=[],model=weights['model'],scaler=scaler,version=-1,rng=rng_state(),lambda_w=w)
        save(folder/'state.pt',state);write(folder/'initial_access.json',dict(ids=task['initial_ids'],count=20,labels_hash=digest(labels),protocol=p['identity']),immutable=True)
    while len(state['query_order'])<50:
        q=len(state['query_order']);d=folder/'batches'/f'b{q//5:02}';unknown=sorted(set(features['ids'])-set(state['known']))
        if not (d/'selection.json').exists():
            data=observed_data(features,scaler,state['known'],task['epsilon']);version=q//10
            if state['version']<version:
                mapped=fit_mapping(folder/'mapping'/f'known{q:02}',state['model'],data,task['lambda_f'],budget,dict(meta,new_queries=q,representation_version=version),lambda_w=w)
                state.update(model=mapped['model'],mapping_status=mapped['status'],version=version,rng=rng_state());save(folder/'state.pt',state)
            model=reload_model(state['model']);model.eval();before=tree_hash(model.state_dict());cache=embedding_cache(model,data)
            gp=fit_gp(d/'gp',cache,data,budget,dict(meta,new_queries=q,representation_version=version));assert before==tree_hash(model.state_dict())==gp['mapping_hash_after']
            rows=predict(cache,data['ids'],unknown,gp);selected=[r['candidate_id'] for r in sorted(rows,key=lambda r:(-r['gp_mean'],r['candidate_id']))[:5]]
            write(d/'selection.json',dict(rows=rows,selected_ids=selected,known_ids=sorted(state['known']),known_labels_hash=digest(state['known']),query_count_before=q,remaining_budget=50-q,representation_version=version,scaler_hash=digest(scaler),model_hash=before,gp_hash=gp['model_hash'],mapping_status=state['mapping_status'],gp_status=gp['status'],fallback=state['mapping_status']!='STABLE_WITHIN_TOLERANCE' or gp['status']!='STABLE_WITHIN_TOLERANCE',lambda_w=w,protocol=p['identity'],saved_at=utc()),immutable=True)
        selection=read(d/'selection.json');assert selection['known_ids']==sorted(state['known']) and selection['known_labels_hash']==digest(state['known'])
        receipt=oracle_request(o,folder,task,p,q,selection)
        for cid in receipt['ids']:assert cid not in state['known'];state['known'][cid]=receipt['labels'][cid];state['query_order'].append(cid)
        state['rng']=rng_state();save(folder/'state.pt',state)
    assert len(state['known'])==70;finish(folder,p,task,accesses,denied,started)


def inner_score(o,t):
    folder=o/'tasks'/t['name'];dest=o/'inner_scores'/(t['name']+'.json')
    if dest.exists():return read(dest)
    if not (folder/'completed.json').exists():row=dict(task=t['name'],context=t['context'],group=t['group'],inner_fold=t['inner_fold'],lambda_w=t['lambda_w'],valid=False,status='NUMERICAL_FAILURE')
    else:
        completed=read(folder/'completed.json');assert file_hash(folder/'predictions.json')==completed['files']['predictions.json']
        pred=read(folder/'predictions.json');ctx=next(c for c in read(o/'contexts.json') if c['context']==t['context']);legal=read(o/ctx['label_file'])
        ids=t['query_ids'];assert set(ids)<=set(legal) and not set(ids)&set(t['train_ids']);rank=sorted(pred['rows'],key=lambda r:(-r['gp_mean'],r['candidate_id']))
        truth={c:legal[c]['validation_mean'] for c in ids};best=max(truth.values())
        row=dict(task=t['name'],context=t['context'],group=t['group'],inner_fold=t['inner_fold'],lambda_w=t['lambda_w'],valid=True,r2=best-max(truth[r['candidate_id']] for r in rank[:2]),r1=best-truth[rank[0]['candidate_id']],mae=float(np.mean([abs(r['gp_mean']-truth[r['candidate_id']]) for r in rank])),all_scores=[dict(r,true_validation=truth[r['candidate_id']]) for r in rank],sealed_sha=file_hash(folder/'completed.json'),prediction_sealed_at=pred['sealed_at'],evaluated_at=utc(),existing_label_partition_only=True,mapping_status=read(folder/'mapping/summary.json')['status'],gp_status=read(folder/'gp/summary.json')['status'])
    write(dest,row,immutable=True);return row


def selected(o,context,group):
    path=o/'selections'/f'{context}-{group}.json'
    if path.exists():return read(path)
    tasks=[t for t in read(o/'tasks.json') if t['stage']=='inner_cv' and t['context']==context and t['group']==group];split=read(o/'inner_splits.json')[context]
    scores=[inner_score(o,t) for t in tasks] if split['sufficient'] else []
    row=dict(context=context,group=group,split_hash=split['identity'],**choose_strength(scores,split['sufficient']),time=utc())
    write(path,row,immutable=True);return row


def launch(o,t,p,truth,env):
    folder=o/'tasks'/t['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completed.json').exists() or (folder/'logical_alias.json').exists() or (folder/'failure.json').exists():return
    attempts=[r for r in journal(o/'attempts.jsonl') if r['task']==t['name']];assert len(attempts)<2,'one interruption recovery maximum'
    append(o/'attempts.jsonl',dict(task=t['name'],attempt=len(attempts)+1,time=utc()));began=time.perf_counter()
    with (folder/'worker.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',str(o),'--task',t['name']],cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        for line in proc.stdout:
            log.write(line);log.flush();event=json.loads(line)
            if event.get('event')!='oracle':continue
            bd=folder/'batches'/f"b{event['batch']:02}";sel=read(bd/'selection.json');q=sel['query_count_before'];ids=sel['selected_ids']
            assert file_hash(bd/'selection.json')==event['selection_sha'] and q==event['batch']*5 and q<50
            assert len(ids)==len(set(ids))==5 and not set(ids)&set(sel['known_ids'])
            if (bd/'reveal.json').exists():receipt=read(bd/'reveal.json')
            else:
                receipt=dict(ids=ids,labels={c:truth[c] for c in ids},selection_sha=event['selection_sha'],query_count_before=q,query_count_after=q+5,revealed_at=utc(),protocol=p['identity'])
                write(bd/'reveal.json',receipt,immutable=True)
                for i,c in enumerate(ids):append(o/'oracle_access.jsonl',dict(task=t['name'],candidate_id=c,query=q+i+1,receipt_sha=file_hash(bd/'reveal.json'),time=utc()))
            proc.stdin.write(json.dumps(receipt)+'\n');proc.stdin.flush()
        code=proc.wait()
    if code:
        numerical=any(read(f).get('status')=='NUMERICAL_FAILURE' for f in folder.rglob('summary.json'))
        write(folder/'failure.json',dict(returncode=code,numerical=numerical,time=utc()))
        if not numerical:raise RuntimeError('engineering failure: '+t['name'])
    append(o/'task_execution.jsonl',dict(task=t['name'],purpose=t['purpose'],seconds=time.perf_counter()-began,returncode=code,time=utc()))
    print(json.dumps(dict(task=t['name'],seconds=time.perf_counter()-began,returncode=code)),flush=True)


def run(o,engineering_only=False):
    p=verify(o,private=True);assert read(o/'synthetic_checks.json')['passed'];lock=(o/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter();tasks=read(o/'tasks.json');truth=read(o/'private/evaluation_A.json');env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    append(o/'controller_sessions.jsonl',dict(time=utc(),pid=os.getpid()))
    for t in tasks:
        if t['stage']!='engineering':continue
        launch(o,t,p,truth,env);assert (o/'tasks'/t['name']/'completed.json').exists(),'required engineering case failed'
    write(o/'engineering_checks.json',dict(passed=True,cases=[read(o/'tasks'/t['name']/'fit/comparison.json') for t in tasks if t['stage']=='engineering'],synthetic_sha=file_hash(o/'synthetic_checks.json')))
    if engineering_only:return
    for ctx in read(o/'contexts.json'):
        begun=time.perf_counter()
        inner=[t for t in tasks if t['stage']=='inner_cv' and t['context']==ctx['context']]
        for t in inner:
            if t['sufficient']:launch(o,t,p,truth,env);inner_score(o,t)
        for g in ['C','F']:selected(o,ctx['context'],g)
        if ctx['context']=='offline-f0':
            elapsed=time.perf_counter()-begun;write(o/'pilot_timing.json',dict(context=ctx['context'],tasks=24,seconds=elapsed,remaining_estimate_seconds=[elapsed*9+600,elapsed*11+2000],includes_diagnostics_and_disk=True,time=utc()))
            print(json.dumps(dict(pilot_seconds=elapsed,remaining_minutes=[(elapsed*9+600)/60,(elapsed*11+2000)/60])),flush=True)
        for t in [t for t in tasks if t['stage']==ctx['stage'] and t.get('context')==ctx['context']]:
            folder=o/'tasks'/t['name'];selection=selected(o,ctx['context'],t['group']);w=0. if t['protocol_type']=='REPAIRED' else selection['lambda_w']
            assignment=dict(lambda_w=w,selection_sha=file_hash(o/'selections'/f"{ctx['context']}-{t['group']}.json"),initial_only=t['stage']=='replay')
            if not (folder/'assignment.json').exists():write(folder/'assignment.json',assignment,immutable=True)
            if t['protocol_type']=='REPAIRED_SELECT_REG' and w==0.:
                target=t['name'].replace('REPAIRED_SELECT_REG','REPAIRED');src=o/'tasks'/target
                assert (src/'completed.json').exists() or (src/'failure.json').exists()
                write(folder/'logical_alias.json',dict(target=target,reason='selected zero: identical initialization/data/protocol',source_completion_sha=file_hash(src/('completed.json' if (src/'completed.json').exists() else 'failure.json')),additional_updates=0,additional_oracle_calls=0),immutable=True)
            else:launch(o,t,p,truth,env)
    write(o/'execution_complete.json',dict(controller_wall_seconds=time.perf_counter()-start,tasks=len(tasks),failures=[t['name'] for t in tasks if (o/'tasks'/t['name']/'failure.json').exists()],time=utc()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','engineering','worker','verify']);ap.add_argument('output',type=Path);ap.add_argument('--task');ap.add_argument('--prompt',type=Path);a=ap.parse_args();o=a.output.resolve()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.command=='prepare':prepare(o,a.prompt)
    elif a.command in ['run','engineering']:run(o,a.command=='engineering')
    elif a.command=='worker':worker(o,a.task)
    else:print(verify(o,private=True)['identity'])
if __name__=='__main__':main()
