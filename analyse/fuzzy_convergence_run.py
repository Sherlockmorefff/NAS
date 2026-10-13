"""Sequential CPU controller; a guarded worker sees only its observed labels."""
from __future__ import annotations
import argparse,fcntl,os,platform,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from analyse.fuzzy_convergence_core import *
from analyse.fuzzy_loss_interaction_run import install_guard,oracle_request,verify_protocol as verify_old,copy_file

OLD=ROOT/'results/posthoc/fuzzy-loss-interaction-v1-20260919T140246Z'
NEW=['analyse/fuzzy_convergence_'+n+'.py' for n in ['core','run','evaluate','recompute']]+['tests/test_fuzzy_convergence.py','configs/fuzzy_convergence_v1.json']


def verify(o):
    p=read(o/'protocol.json');assert digest({k:v for k,v in p.items() if k!='identity'})==p['identity']
    effective=dict(p['sources'])
    if (o/'analysis_amendment.json').exists():
        amendment=read(o/'analysis_amendment.json')
        assert amendment['original_protocol']==p['identity']
        assert digest({k:v for k,v in amendment.items() if k!='identity'})==amendment['identity']
        assert set(amendment['sources'])=={'analyse/fuzzy_convergence_run.py','analyse/fuzzy_convergence_recompute.py'}
        for n,change in amendment['sources'].items():
            assert change['original']==p['sources'][n]
            effective[n]=change['current']
    for n,h in effective.items():assert file_hash(ROOT/n)==h,n
    for n,h in p['inputs'].items():assert file_hash(o/n)==h,n
    return p


def analytic_gp(state,ids):
    g=gp_from_state(state);x=state['x'];pos={c:i for i,c in enumerate(state['cache']['ids'])};xx=state['cache']['embeddings'][[pos[c] for c in ids]]
    with torch.no_grad():
        cov=g.covar_module(x).to_dense()+torch.diag(state['fixed_noise_normalized']+g.likelihood.second_noise)
        L,info=torch.linalg.cholesky_ex(cov)
        if info.item():
            for k in range(3):
                L,info=torch.linalg.cholesky_ex(cov+torch.eye(len(x),dtype=torch.float64)*1e-6*10**k)
                if not info.item():break
        assert not info.item()
        alpha=torch.cholesky_solve((state['target']-g.mean_module(x))[:,None],L)
        mu=(g.mean_module(xx)+g.covar_module(xx,x).to_dense().matmul(alpha).squeeze(1))*state['scale']+state['scaler']['y_mean']
    return mu.tolist()


def reuse_controls(o,pold):
    rows=[];oldenv=read(OLD/'workspace_before.json')
    assert all(oldenv[k]==v for k,v in dict(torch=torch.__version__,gpytorch=gpytorch.__version__,numpy=np.__version__).items())
    assert oldenv['python']==sys.version and oldenv['hostname']==platform.node()
    tasks=[t for t in read(OLD/'tasks.json') if t['stage']=='replay']
    for task in tasks:
        src=OLD/'tasks'/task['name'];dst=o/'imports'/task['name'];done=read(src/'completed.json')
        assert done['protocol']==pold['identity']
        for n,h in done['files'].items():assert file_hash(src/n)==h,(task['name'],n)
        known=read(OLD/task['label_file']);scaler=scaler_from_training(read(o/'features.json'),sorted(known),known)
        maxerror=0.;files={};count=0
        for b in range(10):
            bd=src/'batches'/f'b{b:02}';s=read(bd/'selection.json');r=read(bd/'reveal.json')
            assert s['known_ids']==sorted(known) and s['known_labels_hash']==digest(known)
            assert r['selection_sha']==file_hash(bd/'selection.json') and s['saved_at']<=r['revealed_at']
            assert len(s['selected_ids'])==5 and not set(s['selected_ids'])&set(known)
            if task['group']!='RANDOM':
                gs=load(bd/'gp/latest.pt');ids=[v['candidate_id'] for v in s['rows']];mu=analytic_gp(gs,ids)
                err=max(abs(v-row['gp_mean']) for v,row in zip(mu,s['rows']));assert err<=1e-8;maxerror=max(maxerror,err)
                assert [ids[i] for i in sorted(range(len(ids)),key=lambda i:(-mu[i],ids[i]))[:5]]==s['selected_ids']
                assert gs['scaler']==scaler and gs['train_ids']==sorted(known)
            else:assert s['selected_ids']==read(o/'starts.json')[str(task['search_seed'])]['random_order'][5*b:5*b+5]
            for cid in r['ids']:known[cid]=r['labels'][cid];count+=1
        assert count==50 and len(known)==70
        # Only necessary original inputs, endpoint states and oracle evidence, no old diagnostics/probes.
        wanted=['completed.json','state.pt','initial_access.json','access_boundary.json','timing.json']
        wanted += [str(f.relative_to(src)) for f in (src/'batches').rglob('*') if f.is_file() and f.name in ['selection.json','reveal.json','latest.pt']]
        wanted += [str(f.relative_to(src)) for f in (src/'mapping').rglob('post100.pt')]
        for n in sorted(set(wanted)):
            if (src/n).exists():copy_file(src/n,dst/n);files[n]=file_hash(dst/n)
        row=dict(task=task,source=str(src),source_protocol=pold['identity'],source_manifest_sha=file_hash(src/'completed.json'),files=files,
                 model_scores_recomputed=True,max_gp_mean_error=maxerror,selection_verified=True,initial_accesses=20,logical_queries=50,
                 imported=True,additional_optimizer_updates=0,environment_verified=True)
        rows.append(row)
    write(o/'reuse_manifest.json',dict(passed=True,trajectories=rows,initial_accesses=400,logical_queries=1000,new_oracle_calls=0))


def prepare(o,prompt):
    assert not (o/'protocol.json').exists();o.mkdir(parents=True,exist_ok=True)
    old=verify_old(OLD);cfg=read(ROOT/'configs/fuzzy_convergence_v1.json')
    workspace=dict(hostname=platform.node(),project=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=subprocess.check_output(['git','status','--short'],text=True),
        python=sys.version,torch=torch.__version__,gpytorch=gpytorch.__version__,numpy=np.__version__,cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,
        device='cpu',threads=1,time=utc())
    write(o/'workspace_before.json',workspace);(o/'git_diff_before.patch').write_text(subprocess.check_output(['git','diff','--binary'],text=True))
    inputs=['features.json','candidate_identity.json','folds.json','starts.json','data_checks.json','recipe_verified.json','config.json','REQUEST.md','tasks.json']
    for n in inputs[:-3]:copy_file(OLD/n,o/n)
    copy_file(prompt,o/'REQUEST.md');copy_file(ROOT/'configs/fuzzy_convergence_v1.json',o/'config.json')
    features=read(o/'features.json');assert len(features['ids'])==179 and len(set(features['architectures']))==96
    assert digest(features['ids'])==old['A_id_hash']
    for f,expected in zip(read(o/'folds.json'),cfg['fold_parameters']):assert [f['epsilon'],f['lambda_F']]==expected
    assert sorted(read(o/'starts.json'))==[str(s) for s in range(2026091900,2026091905)]
    tasks=[];label_hashes={}
    for t in read(OLD/'tasks.json'):
        if t['group']=='RANDOM':continue
        task=copy.deepcopy(t);task['protocol_type']='ADAPTIVE';tasks.append(task)
        for n in ['weight_file','label_file']:
            copy_file(OLD/task[n],o/task[n])
            if n=='weight_file':inputs.append(task[n])
            else:label_hashes[task[n]]=file_hash(o/task[n])
        initial=load(o/task['weight_file']);assert initial['untrained'] and initial['model_hash']==tree_hash(JointMetricModel(12,seed=task['model_seed']).state_dict())
    assert len(tasks)==60;write(o/'tasks.json',tasks)
    copy_file(OLD/'private/evaluation_A.json',o/'private/evaluation_A.json')
    assert file_hash(o/'private/evaluation_A.json')==old['private_A_sha']
    reuse_controls(o,old);inputs.append('reuse_manifest.json')
    sources=set(old['sources'])|set(NEW)
    for n in sources:copy_file(ROOT/n,o/'source_snapshot'/n)
    for n in ['AGENTS.md','README.md','docs/repository_layout.md','docs/experiment_artifact_guide.md']:copy_file(ROOT/n,o/'reference_snapshot'/n)
    copy_file(OLD/'protocol.json',o/'reference_snapshot/old_protocol.json');copy_file(OLD/'execution_amendment.json',o/'reference_snapshot/old_execution_amendment.json')
    for n in ['source_audit.md','report.md','workspace_before.json']:copy_file(OLD/n,o/'reference_snapshot'/('old_'+n))
    protected=read(OLD/'protected_before.json');assert all(file_hash(Path(n) if Path(n).is_absolute() else ROOT/n)==h for n,h in protected.items())
    write(o/'protected_before.json',protected)
    p=dict(version='fuzzy-convergence-v1',config=cfg,sources={n:file_hash(ROOT/n) for n in sorted(sources)},inputs={n:file_hash(o/n) for n in sorted(set(inputs))},
        label_hashes=label_hashes,private_A_sha=old['private_A_sha'],old_protocol=old['identity'],old_source=str(OLD),created_at=utc(),formal_updates_before_seal=0,
        interpretations=dict(lr_epoch_first_post_state='change after check t: t+1; a valid window start must be >= t+1',
            plateau_continuous='the last three scheduled check windows must be valid and fail only gradient thresholds',
            fixed_status='FIXED_STEP_STOP, no adaptive window was executed',neighbor_ties='diagnostic only',
            adaptive_joint_intervention='extension + predetermined scheduler + stopping; not a steps-only causal test',
            resume='one process-interruption recovery per task; redo updates charged; numerical failure no retry',
            scoring='same fixed candidate_id tie rule; outer labels only after worker sealing',
            gp_mean_units='.0005 raw accuracy = .05 percentage points'))
    p['identity']=digest(p);write(o/'protocol.json',p,immutable=True)
    (o/'source_audit.md').write_text(f'''# 当前代码与协议核查\n\n主机 {workspace['hostname']}；项目 {ROOT}；分支 {workspace['branch']}；HEAD {workspace['HEAD']}。完整工作区见 workspace_before.json，原差异未覆盖。\n\n旧协议 {old['identity']} 的全部源码、输入及唯一清单修订已逐项核对。原始说明完整保留 reference_snapshot/old_source_audit.md 与 old_report.md；原始数据179候选/96架构/三seed同配方及五折、五起点逐字节沿用，见 data_checks.json、recipe_verified.json、folds.json、starts.json、reuse_manifest.json。基础编码与VAE原校验通过，不读取B、测试或混合oracle。该A池已反复用于开发，不是独立验证。\n\n调用链：fuzzy_convergence_run.prepare复制经校验的无监督缓存与A专用标签；guarded worker仅获取本折训练标签或本起点初始20与已选收据。fuzzy_loss_interaction_core的scaler_from_training/observed_data/components保留训练集标准化、原N2严格绝对差、配对SE噪声权重、MSE/BCE/表示范数与2337参数网络。surrogate.fuzzy_metric.JointMetricModel冻结12维VAE输入、4维HP、8维映射及线性头，公共函数未修改。\n\n新增fuzzy_convergence_core只改训练长度、预定调度与独立诊断；每步当前表示与当前距离重算，三个实际加权项合并一次反传。记录post_update状态；完整窗口分别61/31条loss、4个几何或预测/梯度快照。原100/20步前缀与原检查点比较参数、Adam与RNG，禁止据外层表现选停止点。零邻域权重不进入停止门槛。调度换lr后从t+1重新收集完整窗口，连续3次全过才稳定；终态保留最后6个检查点覆盖最终3窗口。\n\nGP沿用analyse.fuzzy_metric_replay.ReplayGP；GPyTorch ExactMarginalLogLikelihood.forward源码确认log_probability最后除训练样本数；double、ConstantMean、ScaleKernel/各向同性Matern5/2、固定SE噪声+可学额外噪声、原约束、jitter1e-6和3次Cholesky尝试均保留。新GP每次重建，输入hash包含映射/标准化/版本/ID/已知标签。每个检查使用克隆模型求预测以免污染训练缓存，映射前后hash核验。\n\n旧15条FIXED及5条随机逐源文件hash、环境、原始选择与150个GP检查点解析复算通过后导入；导入400初始和1000查询与本轮实际300初始和750查询分开计数。新45离线映射/90GP及15条ADAPTIVE回放/150GP从未训练初始化开始，绝不把旧终点作为起点。\n\n控制器先收封存选择再返回仅5条标签；worker实际尝试读取隐藏A、混合oracle、VAE以及其他轨迹均拒绝。回放每10新标签更新映射、每5重建GP；50后不再拟合。有限未稳照常按最终均值继续并标fallback；数值失败停止该轨迹，保留不完整结果。\n\n资源：CPU单进程单线程，硬上限映射127500、GP75000、GP拟合390；合成各60另列。范围无GNN/VAE训练、B/test访问、补标或新方法。没有修改旧实验、公共函数、分支或HEAD。\n''')
    print(json.dumps(dict(prepared=str(o),protocol=p['identity'],reused=20,tasks=60)),flush=True)


def finish(folder,p,task,accesses,denied,started):
    write(folder/'access_boundary.json',dict(protected_reads=sorted(accesses),denied=denied,full_A_received=False,B_test_reads=0))
    write(folder/'timing.json',dict(TIMES,**PERF,task_seconds=time.perf_counter()-started))
    files={str(f.relative_to(folder)):file_hash(f) for f in folder.rglob('*') if f.is_file() and f.name!='completed.json' and not f.name.endswith('.log') and not f.is_relative_to(folder/'runtime_tmp')}
    write(folder/'completed.json',dict(protocol=p['identity'],task=task,files=files,time=utc()),immutable=True)


def worker(o,name):
    started=time.perf_counter();p=read(o/'protocol.json');task=next(t for t in read(o/'tasks.json') if t['name']==name)
    folder=o/'tasks'/name;folder.mkdir(parents=True,exist_ok=True)
    accesses,denied=install_guard(o,folder,task,p);verify(o)
    if (folder/'completed.json').exists():return
    for blocked in [o/'private/evaluation_A.json',OLD/'private/evaluation_A.json',ROOT/'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2/oracle_validation.json',ROOT/'checkpoints/joint_model_pipeline_global4_best.pth',o/'initial_labels/start2026091904.json' if task['label_file']!='initial_labels/start2026091904.json' else o/'initial_labels/start2026091903.json']:
        try:blocked.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('hidden read was not denied')
    assert file_hash(o/task['label_file'])==p['label_hashes'][task['label_file']]
    torch.manual_seed(task['model_seed']);np.random.seed(task['model_seed']);random.seed(task['model_seed'])
    labels=read(o/task['label_file']);features=read(o/'features.json');weights=load(o/task['weight_file'])
    assert weights['untrained'] and tree_hash(weights['model'])==weights['model_hash']
    scaler=scaler_from_training(features,sorted(labels),labels);budget=Budget(o/'ledger',p['identity'])
    meta={k:task[k] for k in ['stage','group','model_seed','protocol_type']};meta['task']=name
    if task['stage']=='offline':
        meta['fold']=task['fold'];data=observed_data(features,scaler,labels,task['epsilon'])
        mapped=fit_mapping(folder/'mapping',weights['model'],data,task['lambda_f'],budget,dict(meta,representation_version=0))
        for mt,path in [('FIXED','post100.pt'),('ADAPTIVE','latest.pt')]:
            ms=load(folder/'mapping'/path);model=reload_model(ms['model']);model.eval();before=tree_hash(model.state_dict());cache=embedding_cache(model,data)
            gp=fit_gp(folder/f'gp-{mt}',cache,data,budget,dict(meta,mapping_protocol=mt,representation_version=0))
            assert before==tree_hash(model.state_dict())==gp['mapping_hash_after']
            for gt,gpath in [('FIXED','post020.pt'),('ADAPTIVE','latest.pt')]:
                gs=load(folder/f'gp-{mt}'/gpath);rows=predict(cache,data['ids'],task['query_ids'],gs)
                write(folder/f'predictions-{mt}-{gt}.json',dict(rows=rows,train_ids=data['ids'],query_ids=task['query_ids'],
                    mapping_protocol=mt,gp_protocol=gt,mapping_hash=cache['model_hash'],gp_hash=gs['model_hash'],sealed_at=utc()),immutable=True)
        finish(folder,p,task,accesses,denied,started);return
    meta['search_seed']=task['search_seed']
    if (folder/'state.pt').exists():state=load(folder/'state.pt');assert state['scaler']==scaler;restore_rng(state['rng'])
    else:
        state=dict(protocol=p['identity'],known=labels,query_order=[],model=weights['model'],scaler=scaler,version=-1,rng=rng_state())
        save(folder/'state.pt',state);write(folder/'initial_access.json',dict(ids=task['initial_ids'],count=20,labels_hash=digest(labels),protocol=p['identity']),immutable=True)
    while len(state['query_order'])<50:
        q=len(state['query_order']);d=folder/'batches'/f'b{q//5:02}';unknown=sorted(set(features['ids'])-set(state['known']))
        if not (d/'selection.json').exists():
            data=observed_data(features,scaler,state['known'],.01);version=q//10
            if state['version']<version:
                mapped=fit_mapping(folder/'mapping'/f'known{q:02}',state['model'],data,task['lambda_f'],budget,dict(meta,new_queries=q,representation_version=version))
                state.update(model=mapped['model'],mapping_status=mapped['status'],version=version,rng=rng_state());save(folder/'state.pt',state)
            model=reload_model(state['model']);model.eval();before=tree_hash(model.state_dict());cache=embedding_cache(model,data)
            gp=fit_gp(d/'gp',cache,data,budget,dict(meta,new_queries=q,representation_version=version))
            assert before==tree_hash(model.state_dict())==gp['mapping_hash_after']
            rows=predict(cache,data['ids'],unknown,gp);selected=[r['candidate_id'] for r in sorted(rows,key=lambda r:(-r['gp_mean'],r['candidate_id']))[:5]]
            write(d/'selection.json',dict(rows=rows,selected_ids=selected,known_ids=sorted(state['known']),known_labels_hash=digest(state['known']),
                query_count_before=q,remaining_budget=50-q,representation_version=version,scaler_hash=digest(scaler),model_hash=before,gp_hash=gp['model_hash'],
                mapping_status=state['mapping_status'],gp_status=gp['status'],fallback=state['mapping_status']!='STABLE_WITHIN_TOLERANCE' or gp['status']!='STABLE_WITHIN_TOLERANCE',
                protocol=p['identity'],saved_at=utc()),immutable=True)
        selection=read(d/'selection.json');assert selection['known_ids']==sorted(state['known']) and selection['known_labels_hash']==digest(state['known'])
        receipt=oracle_request(o,folder,task,p,q,selection)
        for cid in receipt['ids']:
            assert cid not in state['known'];state['known'][cid]=receipt['labels'][cid];state['query_order'].append(cid)
        state['rng']=rng_state();save(folder/'state.pt',state)
    assert len(state['known'])==70;finish(folder,p,task,accesses,denied,started)


def prefix_check(o,t):
    folder=o/'tasks'/t['name'];old=OLD/'tasks'/t['name'];rows=[]
    for kind,newfile,oldfile in [('mapping','mapping/post100.pt','mapping/post100.pt'),('gp','gp-FIXED/post020.pt','gp/latest.pt')]:
        a=load(folder/newfile);b=load(old/oldfile)
        row=dict(task=t['name'],kind=kind,model_exact=tree_hash(a['model'])==tree_hash(b['model']),
            adam_exact=tree_hash(a['optimizer'])==tree_hash(b['optimizer']),rng_exact=tree_hash(a['rng'])==tree_hash(b['rng']),
            old_file=str(old/oldfile),old_sha=file_hash(old/oldfile),new_sha=file_hash(folder/newfile))
        assert row['model_exact'] and row['adam_exact'] and row['rng_exact'],row
        copy_file(old/oldfile,o/'prefix_reference'/t['name']/oldfile);rows.append(row)
    write(folder/'prefix_verification.json',rows);return rows


def run(o):
    p=verify(o);assert read(o/'engineering_checks.json')['passed']
    lock=(o/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=read(o/'tasks.json');wall=time.perf_counter();truth=read(o/'private/evaluation_A.json');failures=[]
    append(o/'controller_sessions.jsonl',dict(time=utc(),pid=os.getpid()))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    for index,t in enumerate(tasks):
        folder=o/'tasks'/t['name']
        if (folder/'completed.json').exists():continue
        attempts=[r for r in journal(o/'attempts.jsonl') if r['task']==t['name']]
        assert len(attempts)<2,'one interruption recovery per task'
        if (folder/'failure.json').exists():failures.append(t['name']);continue
        append(o/'attempts.jsonl',dict(task=t['name'],time=utc(),attempt=len(attempts)+1))
        began=time.perf_counter();folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'analyse/fuzzy_convergence_run.py'),'worker',str(o),'--task',t['name']],
                cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            for line in proc.stdout:
                log.write(line);log.flush()
                event=json.loads(line)
                if event.get('event')!='oracle':continue
                bd=folder/'batches'/f"b{event['batch']:02}";sel=read(bd/'selection.json');q=sel['query_count_before'];ids=sel['selected_ids']
                assert file_hash(bd/'selection.json')==event['selection_sha'] and q==event['batch']*5 and q<50
                assert len(ids)==len(set(ids))==5 and not set(ids)&set(sel['known_ids'])
                if (bd/'reveal.json').exists():receipt=read(bd/'reveal.json')
                else:
                    receipt=dict(ids=ids,labels={c:truth[c] for c in ids},selection_sha=event['selection_sha'],query_count_before=q,query_count_after=q+5,
                                 revealed_at=utc(),protocol=p['identity'])
                    write(bd/'reveal.json',receipt,immutable=True)
                    for i,c in enumerate(ids):append(o/'oracle_access.jsonl',dict(task=t['name'],candidate_id=c,query=q+i+1,receipt_sha=file_hash(bd/'reveal.json'),time=utc()))
                proc.stdin.write(json.dumps(receipt)+'\n');proc.stdin.flush()
            code=proc.wait()
        if code:
            numerical=any(read(f).get('status')=='NUMERICAL_FAILURE' for f in folder.rglob('summary.json'))
            write(folder/'failure.json',dict(returncode=code,numerical=numerical,time=utc(),message='no retry for numerical or deterministic implementation failure'))
            failures.append(t['name'])
            if not numerical:raise RuntimeError('engineering failure: '+t['name'])
        elif t['stage']=='offline':prefix_check(o,t)
        append(o/'task_execution.jsonl',dict(task=t['name'],seconds=time.perf_counter()-began,returncode=code,time=utc()))
        print(json.dumps(dict(task=t['name'],completed=index+1,total=60,seconds=time.perf_counter()-began,returncode=code)),flush=True)
        if index==2:
            elapsed=time.perf_counter()-wall
            write(o/'pilot_timing.json',dict(first_tasks=[x['name'] for x in tasks[:3]],wall_seconds=elapsed,
                remaining_estimate_seconds=[elapsed*14+elapsed*8,elapsed*14+elapsed*25],
                basis='measured complete f0/s0 C/F/W, including extension, gradients, distances, checkpoint I/O, two GPs and prefix verification; replay size/early stops differ',time=utc()))
    write(o/'execution_complete.json',dict(tasks=60,failures=failures,controller_wall_seconds=time.perf_counter()-wall,time=utc()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker','verify']);parser.add_argument('output',type=Path)
    parser.add_argument('--prompt',type=Path);parser.add_argument('--task');a=parser.parse_args();o=a.output.resolve()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.command=='prepare':prepare(o,a.prompt)
    elif a.command=='run':run(o)
    elif a.command=='worker':worker(o,a.task)
    else:print(verify(o)['identity'])


if __name__=='__main__':main()
