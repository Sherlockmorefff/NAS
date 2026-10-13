"""Write the fixed fivefold report and append status to the existing task document."""
from pathlib import Path
import sys,csv,io,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse import relative_metric_fivefold as f

def number(v,signed=False):
    if v is None:return 'NA'
    return format(v,'+.6f' if signed else '.6f')

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def csv_file(path,rows):
    if not rows:return
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
    for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    value=text.getvalue()
    if path.exists():assert path.read_text()==value
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)

def write_report(o):
    p=f.read(o/'protocol.json');env=f.read(o/'environment.json');runtime=f.read(o/'runtime_validation.json');ind=f.read(o/'independent_recompute.json')
    assert runtime['passed'] and ind['passed'];cost=runtime['cost'];overall=f.read(o/'evaluation/overall.json');paired=f.read(o/'evaluation/paired.json')
    coverage=f.read(o/'evaluation/coverage.json');consistency=f.read(o/'evaluation/fold_consistency.json');starts=f.read(o/'warmstart_checks.json')
    def row(metric,method,scope='all',weighting='candidate',partition='query'):
        return next(r for r in overall if all(r[k]==v for k,v in [('metric',metric),('method',method),('scope',scope),('weighting',weighting),('partition',partition)]))
    metrics=['global_order','local10_order','De_pp','Dstar_pp','E_pp','head_MAE_pp','knn_MAE_pp']
    names=['整体顺序 C ↑','局部10顺序 ↑','实际邻居差 De（百分点）↓','最佳邻居差 Dstar（百分点）↓','额外差 E（百分点）↓','预测头误差（百分点）↓','近邻均值预测误差（百分点）↓']
    controls=['R0-8D','R0-1D'];methods=controls+['R1-8D']
    lines=['# R1 固定五折开发复核（2026-09-11）','',
      '## 完成状态','',
      '沿用原 A 的五折和 15 个仅回归 100 步起点，完成 30 条各新增 200 步的固定轨迹。有效更新和实际更新均为 **6000**，废弃、重复、不确定更新均 **0**，300 步工程余量未使用。新增 GNN、VAE、GP 训练均 **0**。完成重入增加 **0** 步。',
      '', 'CONCLUSION_PLACEHOLDER', '',
      '本轮使用已参与开发的 A 验证成绩，属于开发复核；B 和测试成绩未读取。未新增通过门槛，没有选参数、换划分或延长终点。不能把小幅预测误差变化直接解释为最终 GNN 精度变化。',
      '', '## 实际位置与原始材料','',
      f"主机 `{env['host']}`；项目 `{env['project']}`；分支 `{env['branch']}`；完整 HEAD `{env['HEAD']}`。",
      f"Python `{env['python']}`（{env['version'].split()[0]}），PyTorch {env['torch']}，NumPy {env['numpy']}；CUDA 可用、{env['gpu_count']} 张 GPU。本轮单线程 CPU、float32，GPU 隐藏。完整环境和未提交文件列表见 `environment.json`。",
      '开始时已有的 `bo_phase4.py`、`eval_utils.py` 修改和既有未跟踪研究文件作为基线保留；未 pull、切分支、reset、提交或推送。',
      f"独立结果目录：`{o}`。初次零更新启动目录 `{o.with_name('relative-metric-r1-fivefold-20260911-v1')}` 原样保留。",'',
      f"本轮原始指令来自 `/Users/sherlockmore/Downloads/Codex_Server_R1_Fivefold_Validation_Prompt.md`，项目存档为 `{o/'input_originals/Codex_Server_R1_Fivefold_Validation_Prompt.md'}`，SHA256 `{p['original_prompt_sha256']}`。",
      '已读取项目适用 `AGENTS.md`、最新阶段任务文档、R1 单划分报告及冻结协议、阶段2协议/划分/任务矩阵、N2 五折模型清单。原文副本和绝对路径/完整哈希见 `input_originals/`、`input_sources.json`。',
      f"冻结协议身份 `{p['identity']}`；方法核心 `surrogate/relative_metric.py` 和原小实验入口均未改动。",'',
      '## 要求—代码—证据','',
      table(['要求','实际实现及本轮变化','验证依据'],[
        ['原五折及15起点','新增 fivefold.audit；复用 closure.completed_state，交叉核对 started/completed/清单/哈希','warmstart_checks.json：15项，无缺失；训练表示、距离最大差0；预测头、标签目标和SE平方与旧存档一致'],
        ['原损失、原采样不变','复用 relative_metric 的 validate_batch/build_plan/order_terms/objective；不修改核心','13项零更新检查；141/143/144/145独立标量、旧112前向一致；plan_checks.json'],
        ['逐折标签隔离','五折独立 worker；只允许自身输入和输出文件，拒绝其他折/留出标签路径','实际文件访问拦截测试；每折子进程；全部封存后 evaluation_label_access.json'],
        ['同起点和新Adam','make_branches/fresh_adam；原Adam有100步但两边均重置','actual_initial_objects.json；150节点实际加载模型、动量、RNG和采样位置'],
        ['固定6000步和恢复','新增 intents/receipts/latest 原子账本；完成重入跳过','update_ledger.json：6000项；runtime_validation.json；invocations/run02.json：0新增'],
        ['全局/局部诊断','新增 probe_groups 只读包装；两组原样各半','150探针，独立复算w/间隔/损失及分组占比；损失占比不表示梯度贡献'],
        ['三方法逐折评价','新增 fivefold_report，复用 evaluate_query；R0-1D用预测值作坐标','1611留出、6444训练分开；训练自排除；共同有效查询与架构等权'],
        ['独立复算和便携证据','新增 fivefold_recompute，只有Python标准库','8055行、150探针及全部分折/总体配对复算；independent_recompute.json'],
      ]),
      '## 起点、划分和尺度','',
      table(['折','训练候选','留出候选','训练架构','留出架构'],[[r['family'],r['train_count'],r['query_count'],r['train_architectures'],r['query_architectures']] for r in p['fold_info']]),
      '五折留出恰好覆盖 A 的179候选、96架构一次。每折训练/留出架构互斥。起点均来自原阶段2外层 regression 的有效完成状态；旧100步权重、HP4顺序、候选完整配置身份、fit_ids、标签目标和噪声存档逐一核验。所有15个起点均有旧预测头参照，无参照缺失。',
      table(['折','种子','有效原模型','SHA256前12位'],[[r['fold'],r['seed'],r['source']['model_path'],r['source']['artifact_hashes']['model.pt'][:12]] for r in starts]),
      '完整哈希、完成记录、初始化身份、旧表示/距离/预测误差以及标准化数值见 `warmstart_checks.json`。`warmstarts/` 仅抽取原模型和原尺度；未保留旧优化器或GP对象用于训练。',
      '标签只读取原 `A_validation.json` 白名单：同配方种子0/1/2，逐候选重新核验均值和 $SE=std(ddof=1)/\\sqrt{3}$。每折使用自己的旧训练尺度，未重新拟合。`inputs/fold*/` 仅含本折训练标签及无标签特征；留出标签单独置于 `evaluation_labels/`。',
      '', '## 固定设置与执行边界','',
      '冻结VAE缓存的12维编码与HP4进入原联合映射，输出8维表示及线性预测头。每个原起点深拷贝R0/R1，新建Adam：lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad/foreach/fused=False。全量本折训练；预测MSE逐点平均，大小权重0.001；顺序项R0=0、R1=0.01。无新增归一化、调度、早停、裁剪。',
      '每折每种子单独生成200步完整采样计划，PCG64(20260911+模型种子)，不加入折号。合法方向严格按绝对性能差之差>1e-12；全局/局部10每组无放回抽8个，不足全取、空组为0。每组mean(w×hinge)，两组各半，再对当前折全部中心等权平均；原噪声降权、间隔与分母均未改变。',
      '首步包含在各自200步预算内。0/1/50/100/200步均保存固定首计划探针；探针仅作记录，不参与选点、调参或终点选择。每步有更新意图、optimizer返回收据和原子保存状态；保存全局ID、随机状态、完整Adam与计划位置。',
      f"全部轨迹封存时间 `{f.read(o/'all_inference_sealed.json')['sealed_at']}`；统一留出评价开始 `{f.read(o/'evaluation_label_access.json')['time']}`。推理及精确距离近邻均先于成绩关联封存。",'',
      '## 留出主结果','',
      '每种子先拼接五折179条唯一候选，候选等权平均，再平均3种子。顺序分数用原0–1单位；误差和邻居差用百分点。相对变化均为R1减对照，顺序正值较好，误差负值较好。',
      table(['指标']+methods,[ [n]+[number(row(k,m)['mean']) for m in methods] for n,k in zip(names,metrics)]),
      table(['指标','R1−R0-8D','R1−R0-1D'],[[n]+[number(row(k,'R1-8D minus '+m)['mean'],True) for m in controls] for n,k in zip(names,metrics)]),
      '## 每折及每种子配对变化','',
      table(['对照','整体顺序提高且额外差降低的折数','对应折'],[[r['control'],str(r['both_improve_count'])+'/5',','.join(str(v['fold']) for v in r['folds'] if v['both_global_C_up_E_down']) or '无'] for r in consistency if r['weighting']=='candidate']),
      table(['折','对照','Δ整体C','Δ局部10','ΔE百分点','Δ头误差百分点','Δ近邻误差百分点'],[[fold,m]+[number(row(k,'R1-8D minus '+m,f'fold{fold}')['mean'],True) for k in ['global_order','local10_order','E_pp','head_MAE_pp','knn_MAE_pp']] for fold in range(5) for m in controls]),
      table(['种子','对照','Δ整体C','Δ局部10','ΔE百分点','Δ头误差百分点','Δ近邻误差百分点'],[[s,m]+[number(row(k,'R1-8D minus '+m)['seed_values'][s],True) for k in ['global_order','local10_order','E_pp','head_MAE_pp','knn_MAE_pp']] for s in range(3) for m in controls]),
      '误差变化并非各折一致：预测头误差在第1折（+0.046222百分点）和第3折（+0.026139）增加较多，第0、2折略降；相对R0-8D的近邻误差仅第2折下降（−0.155185），其余四折增加，第1折增加最多（+0.122963）。三个种子对R0-8D的C均增加，但只有种子0的E下降；头误差仅种子2下降，近邻误差仅种子0下降。相对R0-1D，近邻误差下降见于第0、2、3、4折，第1折反而增加0.272469百分点。局部排序未随整体C的小幅提高而改善，不能据此推断已学好性能接近候选的关系。',
      '各折×种子的完整变化见 `evaluation/paired.csv`；不是将45方法状态或8055行当作独立重复。折间训练候选重叠，三个模型种子共享同一批候选；本轮没有统计显著性结论。',
      '', '## 架构等权辅助结果','',
      '每种子先对同架构候选求均值，再对架构等权，最后平均3种子。训练诊断的重复折/候选条目也先在架构内平均；只作辅助诊断。',
      table(['对照','指标','候选等权变化','架构等权变化','方向相同'],[[m,n,number(row(k,'R1-8D minus '+m)['mean'],True),number(row(k,'R1-8D minus '+m,weighting='architecture')['mean'],True),str((row(k,'R1-8D minus '+m)['mean']>0)==(row(k,'R1-8D minus '+m,weighting='architecture')['mean']>0))] for m in controls for n,k in zip(names,metrics)]),
      '', '## 训练查询诊断（不混入主结果）','',
      '五折训练候选合计716个折/候选条目，每查询排除自身；三方法三种子共6444行。存在重复候选，不能理解为6444个独立样本。',
      table(['指标']+methods,[ [n]+[number(row(k,m,partition='train')['mean']) for m in methods] for n,k in zip(names,metrics)]),
      '', '## 探针与分组损失','',
      '完整150节点记录见 `probe_details.csv` 和各轨迹 `probes.json`：总顺序损失、激活比例、表示范数、共享映射梯度比/余弦均保留；全局和局部各自保存w五分位、均值和加权损失占比。下表为15起点对应轨迹的均值，w来自各自固定首计划。**损失占比不是梯度贡献比例。**',
    ]
    probes=[]
    for task in f.read(o/'task_matrix.json'):
        for step,v in f.read(o/'trajectories'/task['task_id']/'probes.json').items():
            for group,g in v['groups'].items():
                probes.append(dict(task=task['task_id'],fold=task['fold'],seed=task['seed'],branch=task['branch'],step=int(step),group=group,**{k:x for k,x in v.items() if k!='groups'},**{'group_'+k:x for k,x in g.items()}))
    probe_rows=[]
    for b in ['R0','R1']:
        for step in [0,1,50,100,200]:
            for g in ['global','local10']:
                selected=[r for r in probes if r['branch']==b and r['step']==step and r['group']==g]
                probe_rows.append([b,step,g]+[number(sum(r[k] for r in selected)/len(selected)) for k in ['group_w_mean','group_half_weighted_loss','group_loss_fraction','group_active_fraction']])
    lines += [table(['分支','新增步','组','w均值','半权损失','损失占比','组内激活比例'],probe_rows),
      '局部w较低属于固定噪声降权的结果，没有因此增大权重、改分母或去除降权。探针不用于事后改参数。','',
      '## 验证、故障与预算','',
      f"运行前13项检查全部通过，包含变长分母、独立标量公式、并列/空组、R0无顺序梯度、错折拒绝、实际文件边界、旧112前向一致和命令行冲突拒绝，优化更新0。实际重新加载150个节点，模型预测/探针、Adam步数、随机状态均核验；首步损失及实际参数变化精确复现，6000条更新ID与收据一一对应。",
      f"独立复算 {ind['counts']['rows']} 行和 {ind['counts']['probe_states']} 探针；顺序/ID/计数精确匹配，浮点差异见 `independent_recompute.json`。留出1611行、训练6444行；全局NA={coverage['NA_global']}，局部NA={coverage['NA_local']}，配对排除累计={coverage['paired_excluded']}，微小负E裁剪={coverage['negative_E_clipped']}。负E低于−1e-12原单位会报错，不填0掩盖。",
      '工程启动曾在v1被自身目录fsync的文件访问检查拦截，实际优化0；修复仅允许本折输出目录本身，跨折与留出路径仍拒绝。自动审批拒绝移动旧目录后，采用新的v2目录继续，v1原样保留。另有两次初始只读探查引号错误，均0更新。训练期间没有故障或状态重算。报告首次汇总时总激活率与分组激活率同名，改为分组前缀后继续，报告修复优化0。详情 `prior_zero_update_startup.json` 和v1 `failures/`、`logs/`。',
      '恢复验证再次执行完整入口，30条已完成任务全部跳过，新增0步；正式实际6000、有效6000、废弃0、重复0、不确定0、修复余量消耗0，总上限6300。未使用合成数据训练，工程标量小例只作零更新检查。','',
      '## 已执行且验证可运行的入口','',
      f'工作目录 `{ROOT}`，解释器 `{env["python"]}`。以下命令均实际执行；正式目录再次 `--run` 已验证不会新增更新。',
      '```sh',f'cd {ROOT}',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --audit',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --check',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --config configs/relative_metric_r1_fivefold_v1.json --run',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --report',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold_verify.py',
      f'/home/fangxy/.conda/envs/nas/bin/python -S analyse/relative_metric_fivefold_recompute.py {o}',
      '```','',
      '日志：`logs/audit.log`、`checks.log`、`run.log`、`run01-fold0..4.log`、`completed_reentry.log`、`runtime_validation.log`、`evaluation.log`、`independent_recompute.log`。独立复算只用标准库，可复制到无项目环境，按证据包README运行。',
      '', '## 交付文件与停止范围','',
      '新增代码：`analyse/relative_metric_fivefold.py`（审查/检查/五折训练/恢复）、`relative_metric_fivefold_report.py`（只读评价）、`relative_metric_fivefold_verify.py`（节点和账本检查）、`relative_metric_fivefold_recompute.py`（独立复算）、`relative_metric_fivefold_write_report.py`（报告）；新增配置 `configs/relative_metric_r1_fivefold_v1.json` 和测试 `tests/test_relative_metric_fivefold.py`。未修改原R1核心及小实验入口。',
      '新报告保存到 `docs/Fuzzy_Metric_R1_Fivefold_20260911.md`；阶段任务文档只追加第15节。`source_snapshot/` 保存冻结训练源码、配置和测试；`delivery_sources/`、`source_changes.patch` 保存其余新增代码及最终差异；`warmstart_checks.json` 保存15起点，`trajectories/` 保存30轨迹，`intents/receipts/update_ledger.json`保存全部更新证据。',
      '阶段3–6、GP/搜索及旧19项缺口均未启动或补做；原N2五折收尾与R1单划分结果未改写。不存在本轮关键材料阻塞。本轮到此结束；下一轮如需进入GP或搜索，应先明确新研究任务、预算及继续保留B/测试的边界，不能把本次开发结果当独立测试结论。','']
    report='\n'.join(lines)
    # Written only after reading and independently verifying the fixed results.
    report=report.replace('CONCLUSION_PLACEHOLDER','**当前R1跨折收益证据不足。** 相对R0-8D，整体顺序C增加0.001535，但额外邻居差E增加0.028777个百分点，只有第0、2折同时改善两者（2/5）；相对R0-1D，C降低0.004342、E减少0.177753个百分点，没有一折同时改善两者（0/5）。局部10顺序对两个对照均略降。预测头误差增加0.016202个百分点，近邻预测误差相对R0-8D增加0.012787、相对R0-1D减少0.277269个百分点。候选等权与架构等权的总体方向一致。')
    for name,data in [('paired',paired),('overall',overall),('summaries',f.read(o/'evaluation/summaries.json'))]:csv_file(o/'evaluation'/f'{name}.csv',data)
    csv_file(o/'probe_details.csv',probes)
    (o/'report.md').write_text(report)
    (ROOT/'docs/Fuzzy_Metric_R1_Fivefold_20260911.md').write_text(report)
    marker='## 15. R1固定五折开发复核（2026-09-11，本轮追加）'
    append='\n'+marker+'\n\n'+"本轮按 `Codex_Server_R1_Fivefold_Validation_Prompt.md` 完成固定复核。实际服务器 `ismc36`，项目 `/home/fangxy/D-VAE`，分支 `feature/accuracy-bottleneck-diagnostics`，HEAD `64a44c14316f11debaf01cf8583426acfe1410e6`。\n\n"+"- 已验证事实：原A179候选/96架构、五折141/143/143/145/144训练候选及38/36/36/34/35留出候选；15个外层仅回归M100起点有效，旧表示/距离/预测头和标签尺度复现，无缺失。\n- 已实现并验证：新增五折入口、逐折文件边界和独立进程、固定200步采样、全局/局部分组探针、6000/6300账本、只读评价和独立复算；原 `surrogate/relative_metric.py` 及单划分入口不改。\n- 已执行实验：30条×200=6000有效更新，实际6000，废弃/重复/不确定0，工程余量使用0；首步计入正式200步。新增GNN/VAE/GP训练0。初次v1目录访问错误在optimizer.step之前发生，更新0，目录原样保留，修复后在v2完成。\n- 验证通过：13项运行前零更新检查；150个节点实际重新加载；完成重入新增0步；1611留出和6444训练行分开；8055行及150探针独立复算通过。\n- 科学结论：当前R1跨折收益证据不足。相对R0-8D，C +0.001535、E +0.028777百分点，2/5折两者同时改善；相对R0-1D，C −0.004342、E −0.177753，0/5折同时改善。局部排序对两对照略降，头误差 +0.016202百分点；候选/架构等权总体方向一致。\n- 交付：`docs/Fuzzy_Metric_R1_Fivefold_20260911.md`；`results/posthoc/relative-metric-r1-fivefold-20260911-v2/` 的protocol、15起点、30轨迹、更新账本、逐候选和分折/总体配对表、独立复算入口及证据包。运行：`python analyse/relative_metric_fivefold.py --audit/--check/--run/--report`（各模式单独执行，实际环境及完整命令见新报告）；配置 `configs/relative_metric_r1_fivefold_v1.json`。\n- 待执行：本轮不再增加实验；阶段3–6、GP/搜索和旧19项保持未执行。没有本轮关键工程阻塞。后续方法修改、预算及阶段推进需另立任务；A已经用于开发，B/测试继续保留，不能称独立盲测。\n\n新增/拟改文件状态：本节所列五折适配代码均已实现且有验证；后续GP/搜索代码缺口沿用前文原阶段任务，不因本次结果删除。此前N2五折、R1单划分及阶段0–6任务内容原样保留，本节只追加本轮事实。\n"
    stage=ROOT/'docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md'
    baseline=(o/'input_originals/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md').read_bytes()
    expected=baseline+append.encode()
    if stage.read_bytes()==baseline:stage.write_bytes(expected)
    else:assert stage.read_bytes()==expected
    assert stage.read_bytes().startswith(baseline)
    (o/'updated_stage_tasks.md').write_bytes(stage.read_bytes())
    f.write(o/'task_document_append.json',dict(path=str(stage),old_sha256=f.file_hash(o/'input_originals/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md'),new_sha256=f.file_hash(stage),old_prefix_unchanged=True,section=15,append_bytes=len(append.encode())))
    print('Chinese report written; Stage Tasks section15 appended; prior text preserved.')
if __name__=='__main__':write_report(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else f.DEFAULT)
