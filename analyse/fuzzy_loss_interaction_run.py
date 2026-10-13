"""A-only original fuzzy loss experiment with a separate historical oracle."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
STARTED=time.perf_counter()
import numpy as np
import torch
from analyse.fuzzy_loss_interaction_core import *
from analyse.fuzzy_metric_offline_core import _label_arrays
from analyse.fuzzy_loss_interaction_diagnostics import diagnose_checkpoint,one_step,sensitivity
from analyse.joint_configuration_metric_gate import hp_feature_vector

BASE=ROOT/'results/posthoc/fuzzy-metric-stage2-20260910-v1'
PREP=ROOT/'results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2'
CONFIG=ROOT/'configs/fuzzy_loss_interaction_v1.json'
NEW=['analyse/fuzzy_loss_interaction_'+n+'.py' for n in ['core','diagnostics','run','evaluate']]+['tests/test_fuzzy_loss_interaction.py','configs/fuzzy_loss_interaction_v1.json']


def copy_file(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert file_hash(source)==file_hash(destination)
    else:shutil.copyfile(source,destination)


def verify_protocol(o):
    p=read(o/'protocol.json')
    assert digest({k:v for k,v in p.items() if k!='identity'})==p['identity']
    effective=dict(p['sources'])
    if (o/'execution_amendment.json').exists():
        amendment=read(o/'execution_amendment.json');assert amendment['original_protocol']==p['identity']
        assert digest({k:v for k,v in amendment.items() if k!='identity'})==amendment['identity']
        for n,item in amendment['sources'].items():
            assert item['original']==p['sources'][n]
            effective[n]=item['current']
    for n,h in effective.items():assert file_hash(ROOT/n)==h,n
    for n,h in p['inputs'].items():assert file_hash(o/n)==h,n
    return p


def prepare(o,prompt):
    assert not (o/'protocol.json').exists(),'use verify/run to resume a sealed experiment'
    o.mkdir(parents=True,exist_ok=True)
    cfg=read(CONFIG);old=read(BASE/'protocol.json');part=read(BASE/'partition.json')
    assert digest({k:v for k,v in old.items() if k!='protocol_hash'})==old['protocol_hash']
    assert digest(part)==old['partition_hash']
    ids=sorted(part['A_ids']);assert digest(ids)==cfg['A_id_hash'] and len(ids)==179
    sources={n:dict(original=h,current=file_hash(ROOT/n)) for n,h in old['source_identity'].items()}
    assert all(x['original']==x['current'] for x in sources.values()),sources
    manifest=read(PREP/'manifest.json')
    assert file_hash(Path(manifest['checkpoint']))==manifest['checkpoint_sha256']
    raw_candidates=read(PREP/'candidates.json');enc=read(PREP/'base_encoding.json')
    assert digest(raw_candidates)==manifest['pool_hash']
    assert digest(enc)==manifest['base_encoding_hash']
    candidates={r['candidate_id']:r for r in raw_candidates}
    candidates=[candidates[c] for c in ids]
    architectures=[r['architecture_fingerprint'] for r in candidates]
    assert len(set(architectures))==96
    assert all(c['candidate_id']==c['joint_fingerprint'] and c['normalization']=='none' and c['hp_mode']=='global4' for c in candidates)
    recipes={c['training_protocol_identity'] for c in candidates};assert len(recipes)==1
    features=dict(ids=ids,architectures=architectures,z=[enc[c['architecture_fingerprint']] for c in candidates],hp=[hp_feature_vector(c) for c in candidates])
    assert np.asarray(features['z']).shape==(179,12) and np.asarray(features['hp']).shape==(179,4)
    write(o/'features.json',features);write(o/'candidate_identity.json',candidates)
    folds=[]
    for i,f in enumerate(part['folds']):
        sel=read(BASE/f'selection/fold{i}.json');choice=sel['settings']['N2']
        assert (choice['epsilon'],choice['lambda_f'])==tuple(cfg['fold_parameters'][i])
        assert (sel['settings']['N1']['epsilon'],sel['settings']['N1']['lambda_f'])==tuple(cfg['fold_parameters'][i])
        train,query=sorted(f['train_ids']),sorted(f['query_ids'])
        assert set(train)|set(query)==set(ids) and not set(train)&set(query)
        arch=dict(zip(ids,architectures));assert not {arch[c] for c in train}&{arch[c] for c in query}
        folds.append(dict(fold=i,train_ids=train,query_ids=query,epsilon=choice['epsilon'],lambda_F=choice['lambda_f'],
                          diagnostic_fallback=not sel['N1_N2_joint_prediction_gate_satisfied'],source_selection_sha=file_hash(BASE/f'selection/fold{i}.json'),
                          source_inner=f['inner']))
        copy_file(BASE/f'selection/fold{i}.json',o/f'original_selection/fold{i}.json')
    assert sorted(c for f in folds for c in f['query_ids'])==ids
    write(o/'folds.json',folds)
    starts={}
    for seed in cfg['search_seeds']:
        initial=np.random.default_rng(seed).permutation(ids)[:20].tolist()
        remaining=sorted(set(ids)-set(initial))
        random_seed=int.from_bytes(hashlib.sha256(f'fuzzy-loss-interaction-v1:random:{seed}'.encode()).digest(),'big')
        random_order=np.random.Generator(np.random.PCG64(random_seed)).permutation(remaining).tolist()
        starts[str(seed)]=dict(initial_ids=initial,remaining_ids=remaining,initial_id_hash=digest(initial),remaining_id_hash=digest(remaining),
                               initial_architectures=len({arch[c] for c in initial}),random_seed=str(random_seed),random_order=random_order)
    write(o/'starts.json',starts)
    write(o/'identity_selection_boundary.json',dict(time=utc(),label_reads_in_prepare_before_identity_freeze=0,starts_sha=file_hash(o/'starts.json')))
    labels=read(BASE/'A_validation.json');assert set(labels)==set(ids)
    _label_arrays(ids,labels)
    assert digest(labels)==read(BASE/'label_boundary.json')['exported_hash']
    copy_file(BASE/'A_validation.json',o/'private/evaluation_A.json')
    tasks=[];inputs=['features.json','candidate_identity.json','folds.json','starts.json','config.json','REQUEST.md','tasks.json'];label_hashes={}
    for f in folds:
        for seed in cfg['model_seeds']:
            wp=f"initial_weights/f{f['fold']}-s{seed}.pt"
            model=JointMetricModel(12,seed=seed)
            save(o/wp,dict(model=model.state_dict(),seed=seed,untrained=True,parameter_count=sum(p.numel() for p in model.parameters()),model_hash=tree_hash(model.state_dict())))
            inputs.append(wp)
            lp=f"train_labels/f{f['fold']}-s{seed}.json";write(o/lp,{c:labels[c] for c in f['train_ids']});label_hashes[lp]=file_hash(o/lp)
            for group,mult in [('C',0.),('F',1.),('W',.1)]:
                tasks.append(dict(name=f"offline-f{f['fold']}-s{seed}-{group}",stage='offline',group=group,model_seed=seed,fold=f['fold'],
                                  epsilon=f['epsilon'],lambda_F=f['lambda_F'],lambda_f=mult*f['lambda_F'],
                                  train_ids=f['train_ids'],query_ids=f['query_ids'],weight_file=wp,label_file=lp))
    for seed in cfg['search_seeds']:
        wp=f'initial_weights/start{seed}-s0.pt';model=JointMetricModel(12,seed=0)
        save(o/wp,dict(model=model.state_dict(),seed=0,untrained=True,parameter_count=sum(p.numel() for p in model.parameters()),model_hash=tree_hash(model.state_dict())))
        inputs.append(wp)
        lp=f'initial_labels/start{seed}.json';write(o/lp,{c:labels[c] for c in starts[str(seed)]['initial_ids']});label_hashes[lp]=file_hash(o/lp)
        for group,weight in [('C',0.),('F',.1),('W',.01),('RANDOM',0.)]:
            tasks.append(dict(name=f'replay-start{seed}-{group}',stage='replay',group=group,model_seed=0,search_seed=seed,epsilon=.01,lambda_F=.1,lambda_f=weight,
                              initial_ids=starts[str(seed)]['initial_ids'],weight_file=wp,label_file=lp))
    write(o/'tasks.json',tasks);copy_file(CONFIG,o/'config.json');copy_file(prompt,o/'REQUEST.md')
    source_names=sorted(set(NEW+list(old['source_identity'])))
    for n in source_names:copy_file(ROOT/n,o/'source_snapshot'/n)
    for n in ['AGENTS.md','README.md','docs/repository_layout.md','docs/experiment_artifact_guide.md','docs/Fuzzy_Metric_Stage2_Results_20260910.md']:
        copy_file(ROOT/n,o/'reference_snapshot'/n)
    workspace=dict(hostname=platform.node(),project=str(ROOT),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                   HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                   tracked_diff_sha=digest(subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True)),python=sys.version,torch=torch.__version__,
                   gpytorch=gpytorch.__version__,numpy=np.__version__,cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,device='cpu',threads=1,time=utc())
    write(o/'workspace_before.json',workspace)
    (o/'git_diff_before.patch').write_text(subprocess.check_output(['git','diff','--binary'],cwd=ROOT,text=True))
    protected={n:file_hash(ROOT/n) for n in old['source_identity']}
    protected.update({str(BASE/n):file_hash(BASE/n) for n in ['A_validation.json','protocol.json','partition.json']})
    protected[manifest['checkpoint']]=manifest['checkpoint_sha256']
    write(o/'protected_before.json',protected)
    write(o/'source_comparison.json',sources)
    write(o/'data_checks.json',dict(passed=True,candidates=179,architectures=96,training_protocol_identity=list(recipes),seeds=[0,1,2],
          grouped_folds=True,untrained_parameter_count=sum(p.numel() for p in model.parameters()),old_protocol=old['protocol_hash'],
          B_label_reads=0,test_label_reads=0,VAE_checkpoint=manifest['checkpoint'],VAE_sha=manifest['checkpoint_sha256'],prepared_manifest=manifest))
    p=dict(version='fuzzy-loss-interaction-v1',config=cfg,inputs={n:file_hash(o/n) for n in inputs},label_hashes=label_hashes,
           sources={n:file_hash(ROOT/n) for n in source_names},private_A_sha=file_hash(o/'private/evaluation_A.json'),
           original_protocol=old['protocol_hash'],original_partition_sha=file_hash(BASE/'partition.json'),created_at=utc(),
           formal_updates_before_seal=0,A_id_hash=digest(ids),interpretation_rules=Path(prompt).read_text().split('## 7.')[1].split('## 8.')[0])
    p['identity']=digest(p);write(o/'protocol.json',p,immutable=True)
    (o/'protocol.md').write_text('# 冻结协议\n\n'+f"协议标识：`{p['identity']}`。冻结时间：{p['created_at']}。\n\n"+
        'REQUEST.md 为完整授权协议；config.json 为实际固定设置；tasks.json 为45个离线任务和20条回放完整清单，folds.json/starts.json 保存全部身份。\n\n'+
        '随机参照独立流：SHA256("fuzzy-loss-interaction-v1:random:{seed}") 全256位大端整数传入 PCG64，对排序后的剩余159个ID排列。起点在本次准备读取成绩前生成。\n\n'+
        '每个映射阶段重新建立 Adam；离线三组从共同随机状态开始，回放每起点模型seed0，从初始20条重新拟合标准化并固定。所有诊断只读检查点，探针独立分支。检查点保存每个规定完成步；latest.pt每步原子更新。\n\n'+
        '近零梯度阈值1e-12；研究方向/胜平负容差1e-12原准确率；GP复算绝对容差1e-8，独立标量复算1e-14。\n\n'+
        '高分子组：当批未知池真实成绩最高的ceil(10%)个，真值同分按ID；预测区间固定宽度0.05，保留负值和超过1的区间。明显高估定义预测减真值≥0.05；未提高选前最好成绩的交集只描述机会占用，不作因果归因。\n\n'+
        '只有数据边界、有限数值、损失和分数复算决定工程门槛，MAE和近邻改善不决定是否回放。首个折/种子C/F/W为正式矩阵的一部分。最多每任务重试一次；首条C回放一次受控中断在收到首批成绩但未合并状态时验证恢复，计一次重启。\n\n'+
        '推进规则按下文，F候选条件具体为F−C均值>容差且胜>负，并且W−F均值≤容差；W候选优先按原协议条件。速度仅在相同最终成绩的起点比较首次达到该成绩的批末预算；存在速度收益独列，不冒充最终收益。\n\n'+p['interpretation_rules'])
    (o/'source_audit.md').write_text(source_audit_text(workspace,manifest))
    print(json.dumps(dict(prepared=str(o),protocol=p['identity'],tasks=len(tasks))),flush=True)


def source_audit_text(w,m):
    return f'''# 当前代码核查

主机 {w['hostname']}，项目 {ROOT}，分支 {w['branch']}，HEAD {w['HEAD']}。完整现有修改见 workspace_before.json/git_diff_before.patch。本轮只新增独立入口；公共实现与原阶段2冻结的8个源文件哈希全部一致，见 source_comparison.json。

surrogate/fuzzy_metric.py 的 JointMetricModel：12维缓存架构编码 detach，4项HP分别为log10(lr)、dropout、log2(hidden_dim)、log10(l2)，hp_encoder 4→16 SiLU，联合映射28→32→32→8，两层SiLU，Linear(8,1)预测头。参数数量由 data_checks.json 实测。未加载旧监督模型；VAE文件 {m['checkpoint']} 只计算SHA，训练器不能读取，基础编码沿用无性能监督的已冻结缓存。原预训练图成员不能精确确认，保留这一限制，不声称独立图预训练。

metric_loss 原样复用：训练标准化目标MSE；无序非自配对权重总和归一的BCE；q=1/(1+平方距离)；clip=max(dtype eps,1e-7)；p=0仍参与分离；零权重/无配对为0；表示各坐标平方和再候选平均。N2严格原准确率绝对差<epsilon，exp(-4*(差/epsilon)^2)。邻域三次配对差的样本SD除sqrt(3)得到均值SE，w=1/(1+(SE/epsilon)^2)。不是单次成绩SD，也不是190个独立观测。

analyse/fuzzy_metric_offline_core.py 的 _label_arrays 与 fit_predict 核实了训练部分输入float32/目标float64总体标准差（correction=0），z/HP尺度下限1e-6，y下限1e-8。映射目标转float32，Adam lr.001, betas(.9,.999), eps1e-8, weight_decay0, amsgradFalse，foreach/fused None，无裁剪、无调度、无早停。C保留相同N2原始量供诊断，权重0，工程检查比较其实际总损失/梯度与旧禁用邻域C等价。

analyse/fuzzy_metric_replay.py 的 ReplayGP 原样复用：detach后的8维表示转double，ConstantMean，ScaleKernel+各向同性Matern5/2。固定观测噪声=(原始SE/y_scale)^2，加可学额外噪声；额外噪声初始化原准确率方差1e-8，下限1e-16。GP seed0重建，Adam lr.05其它默认，20步exact MLL；double jitter1e-6，3次Cholesky尝试，max_cholesky_size800。已知SE用于训练，未读取留出/未知SE作区间。输出均值反标准化一次，方差乘尺度平方一次。每次GP保存输入、目标、已知SE、显式固定噪声、所有坐标、scaler、model/likelihood和Adam/RNG，复算不再优化。

共享hp_encoder/mapping接受三项梯度，regressor只接受回归；诊断autograd.grad不写.grad，不推进Adam/RNG。GP不向映射回传。本轮无R1、冠军损失、G/S、DKL、新核/维数、集成、P2或探索名额。

数据来自原阶段2 A_validation.json，179候选/96架构/同配方GNN seeds0/1/2。prepared_v2候选身份与编码只用于A白名单；不读混合oracle文件。A专用成绩哈希与旧label_boundary.json一致；原五折按架构隔离，内选参数及折0/2预定回退保留在folds.json和original_selection/。原配方Cora/public/NormalizeFeatures/global4/No-LN/full_batch, max300, patience80由既有A出口及候选training_protocol_identity审计继承；本轮不重新读取混合训练日志、B或测试成绩。数据已反复开发，只作开发回放。

新功能：fuzzy_loss_interaction_core.py固定原目标/检查点；diagnostics.py梯度、一步分叉及目标敏感性；run.py只读准备、独立oracle、进程隔离及预算；evaluate.py独立评价和分数复算。训练进程只获得对应折训练成绩或本轨迹初始/已选成绩。每批先写选择文件，controller核对后才写oracle收据；完整A只属于controller/独立评价器。协议/代码/输入在worker启动时验证哈希。原结果与原代码保留。
'''


def install_guard(o,folder,task,p):
    import tempfile
    runtime=folder/'runtime_tmp';runtime.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(runtime)
    allowed={str((o/n).resolve()) for n in p['inputs']}|{str((o/task['label_file']).resolve()),str((o/'protocol.json').resolve()),str((o/'execution_amendment.json').resolve())}
    accesses=set();denied=[]
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        own=path.is_relative_to(folder);ledger=path.is_relative_to(o/'ledger')
        if writing and not (own or ledger):raise PermissionError('write outside own task/ledger denied: '+str(path))
        protected=any(path.is_relative_to(ROOT/n) for n in ['results','checkpoints','data','Cora'])
        if protected and not (str(path) in allowed or own or ledger):
            denied.append(str(path));raise PermissionError('hidden label/other trajectory denied: '+str(path))
        if protected and not writing:accesses.add(str(path))
    sys.addaudithook(hook)
    return accesses,denied


def oracle_request(o,folder,task,p,q,selection):
    batch=q//5;d=folder/'batches'/f'b{batch:02}'
    if not (d/'reveal.json').exists():
        print(json.dumps(dict(event='oracle',task=task['name'],batch=batch,selection_sha=file_hash(d/'selection.json'))),flush=True)
        response=sys.stdin.readline()
        if not response:raise RuntimeError('oracle unavailable; selected batch remains saved')
        receipt=json.loads(response)
        assert receipt==read(d/'reveal.json')
    receipt=read(d/'reveal.json')
    assert receipt['selection_sha']==file_hash(d/'selection.json') and receipt['ids']==selection['selected_ids']
    assert set(receipt['labels'])==set(receipt['ids']) and receipt['query_count_after']==q+5
    return receipt


def finish_task(folder,p,task,accesses,denied,started):
    write(folder/'access_boundary.json',dict(allowed_protected_reads=sorted(accesses),denied_attempts=denied,full_A_received=False,B_test_reads=0))
    write(folder/'timing.json',dict(PERF,task_seconds=time.perf_counter()-started,process_import_seconds=started-STARTED))
    files={str(f.relative_to(folder)):file_hash(f) for f in folder.rglob('*') if f.is_file() and f.name!='completed.json' and not f.name.endswith('.log') and not f.is_relative_to(folder/'runtime_tmp')}
    write(folder/'completed.json',dict(protocol=p['identity'],task=task,files=files,completed_at=utc()),immutable=True)


def worker(o,name):
    started=time.perf_counter();p=read(o/'protocol.json');task=next(t for t in read(o/'tasks.json') if t['name']==name)
    folder=o/'tasks'/name;folder.mkdir(parents=True,exist_ok=True)
    accesses,denied=install_guard(o,folder,task,p);verify_protocol(o)
    if (folder/'completed.json').exists():
        done=read(folder/'completed.json');assert done['protocol']==p['identity']
        for n,h in done['files'].items():assert file_hash(folder/n)==h
        return
    # Execute a real denial check without opening any hidden content.
    for blocked in [o/'private/evaluation_A.json',BASE/'A_validation.json',PREP/'oracle_validation.json',ROOT/'checkpoints/joint_model_pipeline_global4_best.pth']:
        try:blocked.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('label guard failed')
    assert file_hash(o/task['label_file'])==p['label_hashes'][task['label_file']]
    torch.manual_seed(task['model_seed']);np.random.seed(task['model_seed']);random.seed(task['model_seed'])
    labels=read(o/task['label_file']);features=read(o/'features.json');weights=load(o/task['weight_file'])
    assert weights['untrained'] and tree_hash(weights['model'])==weights['model_hash']
    scaler=scaler_from_training(features,sorted(labels),labels)
    ledger=Ledger(o/'ledger',p['identity'])
    meta=dict(task=name,stage=task['stage'],group=task['group'],model_seed=task['model_seed'])
    if task['stage']=='offline':
        meta['fold']=task['fold'];data=observed_data(features,scaler,labels,task['epsilon'])
        state=fit_mapping(folder/'mapping',weights['model'],data,task['lambda_f'],ledger,meta)
        for step in [0,10,50,99]:
            d=folder/'diagnostics'/f'post{step:03}'
            if not (d/'rows.json').exists():diagnose_checkpoint(folder/'mapping'/f'post{step:03}.pt',data,task['lambda_f'],task['lambda_F'],d,meta)
            if task['group']=='C':one_step(folder/'mapping'/f'post{step:03}.pt',data,task['lambda_F'],folder/'probes'/f'post{step:03}',ledger,meta)
        if task['group']=='C' and not (folder/'sensitivity/rows.json').exists():sensitivity(folder/'mapping/post050.pt',data,task['lambda_F'],folder/'sensitivity',meta)
        cache=embedding_cache(reload_model(state['model']),data)
        gp=fit_gp(folder/'gp',cache,data,ledger,meta)
        rows=predict(cache,data['ids'],task['query_ids'],gp)
        write(folder/'predictions.json',dict(rows=rows,protocol=p['identity'],model_hash=cache['model_hash'],gp_hash=gp['model_hash'],
               train_ids=data['ids'],query_ids=task['query_ids'],saved_at=utc()),immutable=True)
        finish_task(folder,p,task,accesses,denied,started);return
    meta['search_seed']=task['search_seed'];random_group=task['group']=='RANDOM'
    if (folder/'state.pt').exists():
        state=load(folder/'state.pt');assert state['protocol']==p['identity'] and state['scaler']==scaler
        restore_rng(state['rng'])
    else:
        state=dict(protocol=p['identity'],known=labels,query_order=[],model=weights['model'],scaler=scaler,version=-1,rng=rng_state())
        save(folder/'state.pt',state)
        write(folder/'initial_access.json',dict(ids=task['initial_ids'],count=20,labels_hash=digest(labels),protocol=p['identity']),immutable=True)
    while len(state['query_order'])<50:
        q=len(state['query_order']);batch=q//5;d=folder/'batches'/f'b{batch:02}'
        assert len(state['known'])==20+q and len(set(state['query_order']))==q
        unknown=sorted(set(features['ids'])-set(state['known']))
        if not (d/'selection.json').exists():
            data=observed_data(features,scaler,state['known'],.01)
            if not random_group:
                version=q//10
                if state['version']<version:
                    event=folder/'mapping'/f'known{q:02}'
                    mapped=fit_mapping(event,state['model'],data,task['lambda_f'],ledger,dict(meta,new_queries=q,representation_version=version))
                    state.update(model=mapped['model'],version=version,rng=rng_state());save(folder/'state.pt',state)
                if q==0 and task['group']=='C' and not (folder/'sensitivity/rows.json').exists():
                    sensitivity(folder/'mapping/known00/post100.pt',data,.1,folder/'sensitivity',meta)
                cache=embedding_cache(reload_model(state['model']),data)
                gp=fit_gp(d/'gp',cache,data,ledger,dict(meta,new_queries=q,representation_version=state['version']))
                rows=predict(cache,data['ids'],unknown,gp)
                selected=[r['candidate_id'] for r in sorted(rows,key=lambda r:(-r['gp_mean'],r['candidate_id']))[:5]]
                model_hash,gp_hash=cache['model_hash'],gp['model_hash']
            else:
                rows=[];selected=read(o/'starts.json')[str(task['search_seed'])]['random_order'][q:q+5];model_hash=gp_hash=None
            write(d/'selection.json',dict(rows=rows,selected_ids=selected,known_ids=sorted(state['known']),known_labels_hash=digest(state['known']),
                  query_count_before=q,remaining_budget=50-q,representation_version=state['version'],scaler_hash=digest(scaler),
                  model_hash=model_hash,gp_hash=gp_hash,protocol=p['identity'],saved_at=utc()),immutable=True)
        selection=read(d/'selection.json')
        assert selection['known_ids']==sorted(state['known']) and selection['known_labels_hash']==digest(state['known'])
        receipt=oracle_request(o,folder,task,p,q,selection)
        if task['group']=='C' and task['search_seed']==2026091900 and q==0 and not (folder/'controlled_resume_once.json').exists():
            write(folder/'controlled_resume_once.json',dict(reason='receipt_saved_before_aggregation',logical_query_count_in_receipt=5,state_query_count=0,time=utc()))
            raise SystemExit(75)
        for cid in receipt['ids']:
            assert cid not in state['known'];state['known'][cid]=receipt['labels'][cid];state['query_order'].append(cid)
        state['rng']=rng_state();save(folder/'state.pt',state)
    assert len(state['known'])==70
    finish_task(folder,p,task,accesses,denied,started)


def validate_done(o,task,p):
    folder=o/'tasks'/task['name'];done=read(folder/'completed.json');assert done['protocol']==p['identity']
    for n,h in done['files'].items():assert file_hash(folder/n)==h,n


def respond_oracle(o,task,p,event,labels):
    folder=o/'tasks'/task['name'];batch=event['batch'];d=folder/'batches'/f'b{batch:02}'
    selection=read(d/'selection.json');assert event['selection_sha']==file_hash(d/'selection.json')
    known=set(task['initial_ids'])
    for b in range(batch):
        prior=read(folder/'batches'/f'b{b:02}'/'reveal.json');assert not known&set(prior['ids']);known.update(prior['ids'])
    assert selection['protocol']==p['identity'] and selection['query_count_before']==5*batch and set(selection['known_ids'])==known
    ids=selection['selected_ids'];assert len(ids)==len(set(ids))==5 and not known&set(ids) and set(ids)<=set(labels)
    path=d/'reveal.json'
    if path.exists():return read(path)
    receipt=dict(task=task['name'],batch=batch,ids=ids,labels={c:labels[c] for c in ids},selection_sha=event['selection_sha'],
                 query_count_before=5*batch,query_count_after=5*(batch+1),revealed_at=utc(),protocol=p['identity'])
    write(path,receipt,immutable=True)
    append(o/'oracle_access.jsonl',dict(task=task['name'],batch=batch,ids=ids,count=5,selection_sha=event['selection_sha'],receipt_sha=file_hash(path),time=receipt['revealed_at']))
    return receipt


def run(o):
    p=verify_protocol(o);assert read(o/'engineering_checks.json')['passed']
    lock=(o/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    labels=read(o/'private/evaluation_A.json');assert file_hash(o/'private/evaluation_A.json')==p['private_A_sha']
    tasks=read(o/'tasks.json');wall=time.perf_counter()
    append(o/'controller_sessions.jsonl',dict(event='start',time=utc(),pid=os.getpid()))
    for index,task in enumerate(tasks):
        if index==45:
            subprocess.run([sys.executable,'-B',str(ROOT/'analyse/fuzzy_loss_interaction_evaluate.py'),'gate',str(o)],check=True,cwd=ROOT)
        folder=o/'tasks'/task['name'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'completed.json').exists():validate_done(o,task,p);continue
        prior=journal(o/'attempts.jsonl');attempts=sum(r['task']==task['name'] for r in prior)
        success=False
        for attempt in range(attempts,2):
            start=time.perf_counter();append(o/'attempts.jsonl',dict(task=task['name'],attempt=attempt+1,time=utc()))
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(folder/'runtime_tmp'))
            (folder/'runtime_tmp').mkdir(exist_ok=True)
            with (folder/f'worker_attempt{attempt+1}.log').open('w') as err:
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'worker',str(o),'--task',task['name']],
                                      cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
                for line in proc.stdout:
                    event=json.loads(line);assert event['event']=='oracle' and event['task']==task['name']
                    receipt=respond_oracle(o,task,p,event,labels)
                    proc.stdin.write(json.dumps(receipt)+'\n');proc.stdin.flush()
                code=proc.wait()
            record=dict(task=task['name'],attempt=attempt+1,exit_code=code,wall_seconds=time.perf_counter()-start,time=utc())
            append(o/'task_execution.jsonl',record)
            if code==0:
                validate_done(o,task,p);success=True;break
            append(o/'failures_and_recovery.jsonl',dict(record,controlled=(code==75)))
        if not success:raise RuntimeError('task incomplete; retry limit reached: '+task['name'])
        print(json.dumps(dict(completed=index+1,total=len(tasks),task=task['name'],time=utc())),flush=True)
        if index==2:
            timings=[read(o/'tasks'/t['name']/'timing.json') for t in tasks[:3]]
            executions=journal(o/'task_execution.jsonl')
            elapsed=sum(r['wall_seconds'] for r in executions)
            write(o/'pilot_timing.json',dict(first_tasks=[t['name'] for t in tasks[:3]],wall_seconds=elapsed,components=timings,
                  remaining_estimate_seconds=[elapsed*14+120,elapsed*40+600],basis='measured three formal tasks; range includes smaller replay fits and 20 process startups',time=utc()))
    subprocess.run([sys.executable,'-B',str(ROOT/'analyse/fuzzy_loss_interaction_evaluate.py'),'all',str(o)],check=True,cwd=ROOT)
    write(o/'execution_complete.json',dict(protocol=p['identity'],tasks=65,controller_wall_seconds=time.perf_counter()-wall,time=utc()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','worker','verify']);parser.add_argument('output',type=Path)
    parser.add_argument('--prompt',type=Path);parser.add_argument('--task');a=parser.parse_args();o=a.output.resolve()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.command=='prepare':prepare(o,a.prompt)
    elif a.command=='run':run(o)
    elif a.command=='worker':
        try:worker(o,a.task)
        finally:write(o/'tasks'/a.task/f'worker_runtime_{os.getpid()}.json',dict(PERF,process_wall_seconds=time.perf_counter()-STARTED,time=utc()))
    else:print(verify_protocol(o)['identity'])


if __name__=='__main__':main()
