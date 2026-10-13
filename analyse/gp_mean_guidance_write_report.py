"""Produce Chinese results, tables and append the latest GP guidance task status."""
from pathlib import Path
import sys,json,math,csv,io
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse import gp_mean_guidance_fivefold as f

def table(headers,rows):return '\n'+'\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n'
def num(v):return 'NA' if v is None else f'{v:.6f}'
def mean(values):return sum(values)/len(values)
def csv_write(path,rows):
    if not rows:return
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n');w.writeheader()
    for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    path.write_text(out.getvalue())

def report(o):
    p=f.read(o/'protocol.json');env=f.read(o/'environment.json');run=f.read(o/'runtime_validation.json');ind=f.read(o/'independent_recompute.json');assert run['passed'] and ind['passed']
    overall=f.read(o/'evaluation/overall.json');selection=f.read(o/'evaluation/selection_overall.json');tasks=f.read(o/'task_matrix.json')
    def row(metric,branch,scope='all',part='query',weighting='candidate'):
        return next(r for r in overall if r['scope']==scope and r['partition']==part and r['weighting']==weighting and r['metric']==metric and r['branch']==branch)
    def v(metric,branch,scope='all',part='query',weighting='candidate'):return row(metric,branch,scope,part,weighting)['value']
    def sv(metric,branch,scope='all',part='query'):return next(r['value'] for r in selection if r['metric']==metric and r['branch']==branch and r['scope']==scope and r['partition']==part)
    times=[];blocks=[];support=[]
    for t in tasks:
        d=f.load(o/t['input']);folder=o/'trajectories'/t['task_id'];tim=f.read(folder/'timing.json');times.append(dict(task=t['task_id'],fold=t['fold'],seed=t['seed'],branch=t['branch'],**tim))
        if t['branch']=='G':
            probes=f.read(folder/'probes.json')
            for block in range(4):
                start=next(r for r in probes if r['step']==block*50 and r['phase']=='after_refresh');end=next(r for r in probes if r['step']==(block+1)*50 and r['phase']=='before_refresh')
                blocks.append(dict(task=t['task_id'],fold=t['fold'],seed=t['seed'],block=block,old_teacher_MSE_start=start['guidance_loss'],old_teacher_MSE_end=end['guidance_loss'],
                  old_teacher_RMSE_start_pp=100*math.sqrt(start['guidance_loss'])*d['scaler']['y_scale'],old_teacher_RMSE_end_pp=100*math.sqrt(end['guidance_loss'])*d['scaler']['y_scale'],
                  gradient_start=start['guidance_shared_norm'],gradient_end=end['guidance_shared_norm'],weighted_ratio_start=start['weighted_ratio'],weighted_ratio_end=end['weighted_ratio'],cosine_start=start['cosine'],cosine_end=end['cosine']))
            diag=f.read(folder/'self_support_diagnostic.json');support.extend([dict(task=t['task_id'],**r) for r in diag['rows']])
    f.write(o/'timing_summary.json',times);f.write(o/'teacher_block_summary.json',blocks);f.write(o/'self_support_summary.json',support)
    allprobes=f.read(o/'actual_guidance_probes.json');gradients=[r['guidance_shared_norm'] for r in allprobes];ratios=[r['weighted_ratio'] for r in allprobes if r['weighted_ratio'] is not None];cos=[r['cosine'] for r in allprobes if r['cosine'] is not None]
    branch_times={b:{k:sum(r[k] for r in times if r['branch']==b) for k in ['mapping_update_seconds','gp_update_seconds','wall_seconds_this_invocation']} for b in ['C','G']}
    names=['GP MAE（百分点）↓','GP RMSE（百分点）↓','预测头MAE（百分点）↓','95%区间覆盖率↑','95%区间宽度（百分点）']
    metrics=['GP_MAE_pp','GP_RMSE_pp','head_MAE_pp','coverage95','width95_pp']
    sel_names=['GP排序相关','预测头排序相关','GP前5真实平均性能（%）','预测头前5真实平均性能（%）','GP前5距真实最佳5差（百分点）↓','预测头前5距真实最佳5差（百分点）↓']
    sel_metrics=['GP_Spearman','head_Spearman','GP_top5_true_mean_pp','head_top5_true_mean_pp','GP_top5_gap_pp','head_top5_gap_pp']
    lines=['# GP均值指导表示学习：固定五折开发实验（2026-09-13）','','## 结果与结论','',
      '本轮按最新Prompt完成代码审查、实现、验证、五折正式运行与独立复算。30条映射轨迹各新增200步，实际/有效6000步；180教师＋30终点GP，各20步，实际/有效4200步。GNN/VAE新增训练0。没有失败、重复、不确定优化更新；完成重入映射和GP均新增0。',
      '**GP平均预测误差改善，但跨折改善集中，GP前5选择未获益，区间覆盖退化。本版不能据此宣布搜索有效或替换B4＋P2。**',
      '主指标GP留出MAE由8.053454降至7.656827个百分点，变化−0.396627；三个模型种子的变化均为负，但只有第1、3折改善，主要来自第1折−2.397721。第0、2、4折分别增加0.438249、0.026370、0.107757。架构等权变化−0.393873，方向相同。',
      'G的GP误差仍高于同组预测头（7.656827 vs 6.890212个百分点）。GP前5实际平均性能由73.360000%变为73.249778%，下降0.110222个百分点；预测头前5则提高0.816000个百分点。GP排序相关由0.511399降至0.499074。预测头收益不能冒充GP选点收益。',
      '95%观测区间覆盖率由86.405959%降至78.770950%，降低7.635009个百分点；区间平均宽度由30.575363缩至22.091912个百分点。平均误差有所下降同时伴随更窄、覆盖更差的区间，不能直接用于不确定性驱动搜索。',
      '目前不支持优先启动GP搜索回放。可保留预测适配的开发线索；若以后专门追查预测头收益或改善集中原因，应另立固定验证/回放协议并明确预算。本轮不追加调参、输入对照、伪标签、VAE训练或搜索。A已用于多轮开发，不能称独立确认，也不能由本版结果反推VAE必然丢失信息。','',
      '## 实际位置与版本','',f"主机 `{env['host']}`；项目 `{env['cwd']}`；分支 `{env['branch']}`；完整HEAD `{env['HEAD']}`。",f"Python `{env['python']}`（{env['python_version'].split()[0]}），PyTorch {env['torch']}，GPyTorch {env['gpytorch']}，BoTorch {env['botorch']}，NumPy {env['numpy']}。CUDA可用：{env['cuda']}，GPU数{env['gpus']}；实际训练均单线程CPU。",
      '适用AGENTS及运行进程已核对；启动时无用户模型训练进程，未终止任何其他工作。当前未提交修改 `bo_phase4.py`、`eval_utils.py` 和既有未跟踪研究文件保留。未pull、reset、clean、切分支、提交或推送。',
      f"结果绝对路径：`{o}`。冻结协议身份：`{p['identity']}`。最新Prompt原文件 `/Users/sherlockmore/Downloads/Codex_Server_GP_Mean_Guidance_Prompt.md`，SHA256 `{p['prompt_sha']}`，原文保存在 `input_originals/`。",
      '旧源码、模型与数据绝对路径及哈希见 `input_sources.json`、`pretraining_validation.json`、`warmstart_checks.json`；完整环境、导入位置、脏工作树与进程见 `environment.json`。原R1核心/入口及其禁止GP的保护保持不变。','',
      '## 最新决定覆盖旧分析','',
      '固定VAE均值编码12维＋HP4，原M100起点→各新增200步→M300。输入编码对照暂缓，不是前置条件；不采用旧分析中的56维接口、M300起点或M500终点。旧PG_LBO分析原文已存档，另追加/标注最新执行决定，历史内容保留。旧N2/R1负结果、阶段2的640次账本及19项缺口不改。','',
      '## 要求—当前实现—问题—修改—验证证据','',
      table(['要求','当前实现/发现','修改位置','实际验证'],[
       ['固定15个共同起点','原外层regression M100有效；可复用原模型和尺度','gp_mean_guidance_fivefold.audit','15项恢复；旧训练表示与gp_train_inputs完全相同；目标/SE²差≤1e-12'],
       ['GP误差可导','旧offline_core.fit_predict的no_grad/NumPy推理会断梯度，旧边界本身正确','surrogate/gp_mean_guidance.py mean/guidance','独立dense Cholesky及有限差分；指导单项到查询、HP和映射，线性头无直接梯度'],
       ['同架构组排除','旧R1三元组不能代替GP教师','architecture_groups、worker分块','哈希3组在拟合前冻结；支持组与查询组架构互斥；仅本折训练标签'],
       ['原GP配置与缓存','ReplayGP类可复用；不能复用旧坐标prediction_strategy','fit_gp、condition/from_gp','210次独立拟合，全部缓存按新快照重建；210个dense均值/方差核验'],
       ['两类预算和恢复','旧R1只记录映射并正确禁止GP','新增Ledger/fit_gp/worker','6000映射＋4200GP逐步账本；4200个GP步骤状态实际重载；完成重入均0'],
       ['统一留出评价','旧R1主指标为距离，不满足本轮主指标','gp_mean_guidance_report/recompute','1074留出＋4296训练记录分开；前5和区间在标签关联前封存；标准库独立复算'],
      ]),
      '本轮未发现需要改变科学定义的工程问题；新入口用于补足旧实验不具备的功能。报告首次导出CSV时，逐折行和总体行字段不同导致导出报错，已改为合并列名后完成；此修复没有重训或改变任何指标。所有正常训练均按原固定协议完成，没有训练后改公式或丢弃轨迹。旧R1对象检查工具中的历史函数定位不代表本轮损失路径；实际调用位置单独记录在 `pretraining_validation.json.actual_imports`，本轮真实目标为 `surrogate.gp_mean_guidance.objective → guidance → mean`。','',
      '## 模型、标签与起点','',
      'VAE缓存12维，HP顺序 `[log10(lr),dropout,log2(hidden_dim),log10(l2)]`；HP4→16，与架构12拼接后28→32→32→8，原8→1线性头。冻结VAE及原z/hp/y尺度，未重新标准化或改变容量。每个原起点独立深拷贝C/G，新Adam空状态，所有HP/映射/预测头参数requires_grad开启。',
      table(['折','训练候选','留出候选','模型种子'],[[i,len(r['train_ids']),len(r['query_ids']),'0,1,2'] for i,r in enumerate(p['folds'])]),
      'A179候选、96架构，五折留出恰好覆盖一次；每折训练/留出架构互斥。标签仅来自原A专用三种子validation导出，均值和SE逐条核验。未解析B或测试成绩大库。原M100仅回归、lambda_f=0、representation_steps=100、保存的Adam步数100均核验。所有15个C终点权重与旧R1五折R0的M300只读参照完全一致；该参照没有替代M100起点，也没有用于选择设置。',
      '“未见架构”在本报告表示未见该架构的性能训练标签：VAE预训练可能见过结构本身。内部三组也不是严格独立交叉拟合，因为原映射/scaler已用过全部外层训练标签。训练进程只访问本折净化起点、训练标签和无标签特征，文件访问守卫拒绝全A、其他折和留出标签文件。','',
      '## 固定目标、教师与预算','',
      '$L_C=mean[(H(e)-\\widetilde y)^2]+0.001mean[\\|e\\|_2^2]$；$L_G=L_C+0.1L_{GP}$。大小项是每候选各维平方之和再平均；回归是逐点平方后平均。$L_{GP}$汇总三个组的全部候选后等权平均，未对不等人数的组简单等权。关闭邻域/三元组、伪标签、方差最小化和额外SE损失权重。',
      '唯一架构按SHA256("gp_mean_guidance_v1:"+指纹)、指纹排序，位置模3分组。指导组g的支持集为另外两组，同架构全部HP配置随组排除。新增步0/50/100/150重算完整训练表示并detach，各新拟合三个教师20步；块内核参数、支持坐标、Cholesky和alpha冻结，仅当前查询可导。新增200步后，两组分别用完整外层训练表示从头拟合终点GP20步。',
      '映射float32，GP及指导目标float64；预测头目标保留原float32。均值/SE按原外层尺度还原，无组内尺度，无新归一化、调度、裁剪、早停。映射Adam lr0.001、betas(0.9,0.999)、eps1e-8、weight_decay0、amsgrad/foreach/fused=False，每条仅在开头重置一次。GP每次独立种子0，Adam lr0.05及展开默认设置见配置，不扰动映射RNG。',
      'GP复用ReplayGP：ConstantMean、ScaleKernel、各向同性Matérn5/2，固定$(SE/s_T)^2$加学习噪声；额外噪声初值$(10^{-4}/s_T)^2$、下界$(10^{-8}/s_T)^2$，min_fixed_noise=0。Cholesky只有失败才按1e-6起重试，禁止无条件加观测噪声。实际全部拟合及独立条件化均不需要jitter，实际加入0。',
      table(['预算','有效','实际确认','废弃/重复/不确定','实际上限'],[['映射更新',6000,6000,'0/0/0',6300],['教师GP更新',3600,3600,'0/0/0','GP合计4410'],['终点GP更新',600,600,'0/0/0','GP合计4410'],['GP总更新',4200,4200,'0/0/0',4410],['新增GNN/VAE训练',0,0,'0/0/0',0]]),
      '保存更新前意图、optimizer返回收据、原子提交状态、RNG、模型/Adam、全局ID与步数；GP拟合1–20步全部留存。教师三组保存完整后才标记ready；恢复重用已完成块。关键节点区分刷新前/后：C共90节点，G共150节点，合计240个实际重载验证。',
      f"全部预测封存：`{f.read(o/'all_predictions_sealed.json')['time']}`；评价读取标签：`{f.read(o/'evaluation_label_access.json')['time']}`。额外真实GNN标签0。",'',
      '## 留出主指标及固定次指标','',
      '每种子拼接179条唯一留出候选，先候选等权、再平均三种子。相关性和前5另在每折/种子内计算，先平均种子再平均折。C仅回归，G加入GP均值指导；G−C为配对变化。',
      table(['指标','C','G','G−C'],[[n,num(v(k,'C')),num(v(k,'G')),num(v(k,'G-C'))] for n,k in zip(names,metrics)]),
      table(['指标（每折后汇总）','C','G','G−C'],[[n,num(sv(k,'C')),num(sv(k,'G')),num(sv(k,'G-C'))] for n,k in zip(sel_names,sel_metrics)]),
      '预测不夹断到0–1。观测方差=潜在方差×原尺度²＋额外噪声×原尺度²＋本折训练SE²均值，95%区间使用±1.96标准差；未用隐藏查询自身SE。覆盖率是实测描述，不是保证。','',
      '## 每折与每种子变化','',
      table(['折','ΔGP MAE pp','Δ头MAE pp','ΔGP前5真实性能 pp','Δ头前5真实性能 pp','Δ覆盖率'],[[i,num(v('GP_MAE_pp','G-C',f'fold{i}')),num(v('head_MAE_pp','G-C',f'fold{i}')),num(sv('GP_top5_true_mean_pp','G-C',f'fold{i}')),num(sv('head_top5_true_mean_pp','G-C',f'fold{i}')),num(v('coverage95','G-C',f'fold{i}'))] for i in range(5)]),
      table(['种子','C GP MAE pp','G GP MAE pp','差值 pp','头MAE差值 pp'],[[s,num(row('GP_MAE_pp','C')['seed_values'][s]),num(row('GP_MAE_pp','G')['seed_values'][s]),num(row('GP_MAE_pp','G-C')['seed_values'][s]),num(row('head_MAE_pp','G-C')['seed_values'][s])] for s in range(3)]),
      'GP MAE改善折为1、3；GP前5平均性能改善也为1、3，但第0、2、4折下降。预测头MAE改善主要来自第2折−2.224206和第4折−0.862067个百分点，第1、3折反而略增；预测头前5在4/5折提高，第1折下降。完整每折×种子配对见 `evaluation/paired.csv` 和 `selection_paired.csv`，不把模型种子当作新标签批次。',
      table(['权重方式','C GP MAE pp','G GP MAE pp','G−C pp'],[[w,num(v('GP_MAE_pp','C',weighting=w)),num(v('GP_MAE_pp','G',weighting=w)),num(v('GP_MAE_pp','G-C',weighting=w))] for w in ['candidate','architecture']]),
      '架构等权先在每种子内平均同架构候选，再平均架构、最后平均种子；方向与候选等权一致。五折训练集重叠、1074行共享候选，不能作为1074个独立实验。本轮不宣称显著、等价或搜索收益。','',
      '## 训练误差与教师误差必须区分','',
      table(['终点重拟合完整训练GP/头','C训练','G训练','C留出','G留出'],[[n,num(v(k,'C',part='train')),num(v(k,'G',part='train')),num(v(k,'C')),num(v(k,'G'))] for n,k in zip(names,metrics)]),
      '训练汇总共4296行，包含716个重复折/候选条目×三种子×两组，与1074留出主记录分开。终点GP训练查询包含自己的训练支持，低训练误差不代表推广。下表只比较同一冻结教师在本块前后的指导误差，不能与终点GP误差混算。',
      table(['训练块','旧教师标准化MSE起','旧教师标准化MSE止','旧教师RMSE起 pp','旧教师RMSE止 pp'],[[b]+[num(mean([r[k] for r in blocks if r['block']==b])) for k in ['old_teacher_MSE_start','old_teacher_MSE_end','old_teacher_RMSE_start_pp','old_teacher_RMSE_end_pp']] for b in range(4)]),
      f"60个训练块中旧教师MSE下降 {sum(r['old_teacher_MSE_end']<r['old_teacher_MSE_start'] for r in blocks)}/60。仅此不能判方法有效；主结论仍取决于外层重新拟合GP。",'',
      '## 指导梯度、支持排除与数值证据','',
      f"135个真实指导探针单独使用autograd.grad复算：共享映射/HP指导梯度范数范围 {num(min(gradients))}–{num(max(gradients))}；0.1加权后与回归梯度之比范围 {num(min(ratios))}–{num(max(ratios))}，平均 {num(mean(ratios))}；夹角余弦范围 {num(min(cos))}–{num(max(cos))}，平均 {num(mean(cos))}。零范数NA实际 {sum(r['NA_reason'] is not None for r in allprobes)} 项。",
      '指导输入、HP和映射梯度均有限，预测头最后线性层对指导项直接梯度为None，符合定义；预测头由回归项训练。VAE输入梯度为None，冻结支持/alpha不持有计算图。所有实际输入缓存、原模型、原源码哈希前后核对。每块快照、教师metadata、组支持ID、映射版本与scaler均在运行后逐一核验。',
      table(['初始诊断支持','标准化MSE均值','均值输入梯度范数均值','MSE输入梯度范数均值'],[[kind]+[num(mean([r[k] for r in support if r['conditioning']==kind])) for k in ['MSE','mean_query_gradient_norm','MSE_query_gradient_norm']] for kind in ['full_training_support','excluded_group']]),
      '该附加诊断沿用每个已拟合初始教师的同一参数，只重做条件化和矩阵求解，新增GP更新0。排除整组也减少支持数量，不是纯粹单点删除因果实验；内部误差不是独立交叉拟合成绩。',
      f"运行前7项检查通过，全部禁止optimizer.step。非重合固定小例独立矩阵均值误差 {f.read(o/'numeric_preflight.json')['mean_maxdiff']:.3g}，有限差分梯度最大误差 {f.read(o/'numeric_preflight.json')['finite_difference_maxdiff']:.3g}；容差在训练前固定。运行后240个映射节点、4200个GP拟合步骤状态实际加载；420个GP步骤还前向核对。210个真实GP的均值/方差由独立NumPy dense-Cholesky公式检查通过。",
      f"标准库独立复算5370候选记录、60个折/种子/组/分区记录，覆盖全部MAE、RMSE、排名相关、前5、区间及配对汇总。最大数值差 {max(ind['max_absolute_differences'].values()):.3g}，预设容差1e-10；前5名单/覆盖判定一致，NA与缺失真实计数见 `evaluation/coverage.json`。没有合成训练或未计账训练。",'',
      '## 实际计算时间','',
      table(['组（15轨迹合计）','映射更新计时秒','GP更新计时秒','总墙钟秒，含保存/推理/探针'],[[b]+[num(branch_times[b][k]) for k in ['mapping_update_seconds','gp_update_seconds','wall_seconds_this_invocation']] for b in ['C','G']]),
      f"G/C合计墙钟比 {branch_times['G']['wall_seconds_this_invocation']/branch_times['C']['wall_seconds_this_invocation']:.3f}。两组只有标签与映射更新次数匹配，G另有180次教师拟合，不能称总计算预算相同。各轨迹计时见 `timing_summary.csv`；矩阵诊断时间另存每条G轨迹 `self_support_diagnostic.json`。",'',
      '## 实际可运行命令及交付','',f'目录 `{ROOT}`；下面命令均已在服务器执行。正式目录重入已验证不增加映射或GP更新。','```sh',f'cd {ROOT}',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_fivefold.py --audit',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_fivefold.py --check',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_fivefold.py --config configs/gp_mean_guidance_fivefold_v1.json --run',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_fivefold.py --report',
      'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_verify.py',
      f'/home/fangxy/.conda/envs/nas/bin/python -S analyse/gp_mean_guidance_recompute.py {o}',
      '```','',
      '日志分别位于 `logs/audit.log`、`check.log`、`run.log`、`run01-fold*.log`、`completed_reentry.log`、`runtime_validation.log`、`evaluation.log`、`independent_recompute.log`。`invocations/run02.json`证明完成重入0更新。',
      '新增文件：`surrogate/gp_mean_guidance.py`；`analyse/gp_mean_guidance_fivefold.py`、`gp_mean_guidance_report.py`、`gp_mean_guidance_verify.py`、`gp_mean_guidance_recompute.py`、`gp_mean_guidance_write_report.py`；`tests/test_gp_mean_guidance.py`；`configs/gp_mean_guidance_fivefold_v1.json`；新报告 `docs/GP_Mean_Guidance_Fivefold_20260913.md`。任务文档仅追加第16节，PG_LBO分析保留原文并标注最新覆盖决定。',
      '证据：`protocol.json`、`source_snapshot/`、`input_sources.json`、`warmstart_checks.json`、`groups.json`、`trajectories/`、`intents/receipts/`、`update_ledger.json`、`evaluation/`、`actual_guidance_probes.json`、`runtime_validation.json`、`independent_recompute.json`、源码差异与可携带证据包。GP库重试jitter实际0，无失败轨迹；科学结论不依赖训练目标是否下降。',
      '本轮无关键输入缺失、无科学规则阻塞。映射恢复余量300步、GP恢复余量210步均未使用；未补旧19项，未改变B4＋P2主线，也未启动后续搜索。','']
    text='\n'.join(lines);(o/'report.md').write_text(text);(ROOT/'docs/GP_Mean_Guidance_Fivefold_20260913.md').write_text(text)
    for name,records in [('paired',f.read(o/'evaluation/paired.json')),('overall',overall),('selection_paired',f.read(o/'evaluation/selection_paired.json')),('selection_overall',selection),('per_task',f.read(o/'evaluation/per_task.json'))]:csv_write(o/'evaluation'/f'{name}.csv',records)
    for name,records in [('timing_summary',times),('teacher_block_summary',blocks),('actual_guidance_probes',allprobes),('self_support_summary',support)]:csv_write(o/f'{name}.csv',records)
    stage=ROOT/'docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md';baseline=(o/'input_originals'/stage.name).read_bytes()
    append='''
## 16. GP均值指导固定五折（2026-09-13，本轮追加）

最新决定来自 `Codex_Server_GP_Mean_Guidance_Prompt.md`，优先于旧PG_LBO分析：输入对照暂缓、不再是前置条件；固定VAE12＋HP4、原15个外层仅回归M100，共同起点各新增200步到M300。旧56维/M300→M500安排被覆盖，历史分析保留。

- 已实现：可微GP均值指导、按架构哈希内部三组及支持排除、50步块内冻结教师和刷新缓存、映射/GP双预算恢复、独立进程标签边界、封存后统一GP/预测头/前5/区间评价。
- 已完成：30条映射轨迹、6000实际/有效更新；180教师＋30终点GP，4200实际/有效更新；废弃/重复/不确定0，修复余量使用0；新增GNN/VAE训练0。完成重入两个计数均不增加。
- 已验证：15原M100身份/尺度/标签与表示；7项零更新检查；240映射节点、4200GP步骤实际重载；210个GP独立dense公式；15个C终点与旧R1五折R0权重精确一致；1074留出和4296训练记录及全部配对独立复算通过。
- 实际结果：GP留出MAE C8.053454→G7.656827百分点（−0.396627），三种子均改善但仅第1、3折改善，主要来自第1折。架构等权−0.393873。G的GP仍弱于G预测头MAE6.890212。GP前5平均真实性能−0.110222百分点，预测头前5＋0.816000百分点；区间覆盖86.405959%→78.770950%，宽度30.575363→22.091912百分点。
- 结论：GP均值适配有平均预测收益，但集中且未转化为GP前5收益，区间覆盖退化；不能宣布搜索有效或更换B4＋P2。暂不优先推进GP搜索回放；后续若追查预测头/集中收益，另立固定协议和预算。本轮停止，不调权重、不启动输入对照、伪标签、VAE训练或搜索。
- 交付：`docs/GP_Mean_Guidance_Fivefold_20260913.md`；`results/posthoc/gp-mean-guidance-fivefold-20260913-v1/`；配置 `configs/gp_mean_guidance_fivefold_v1.json`；入口 `analyse/gp_mean_guidance_fivefold.py --audit/--check/--run/--report`（模式分开），只读验证 `gp_mean_guidance_verify.py` 和独立复算 `gp_mean_guidance_recompute.py`。完整实际命令、数据/源码绝对路径、日志和账本见新报告。

本节只追加本轮事实。N2/R1负结果、阶段2原640次账本及19项缺口、阶段0–6历史要求保留。A仍是开发数据；“未见”限定未见该架构的性能训练标签，VAE可能见过结构。B和测试继续保留。没有本轮工程阻塞，不能将工程检查通过替代科学效果。
'''
    expected=baseline+append.encode()
    if stage.read_bytes()==baseline:stage.write_bytes(expected)
    else:assert stage.read_bytes()==expected
    (o/'updated_stage_tasks.md').write_bytes(expected);f.write(o/'task_append.json',dict(path=str(stage),old_prefix_unchanged=expected.startswith(baseline),new_sha=f.file_hash(stage),section=16))
    print(json.dumps(dict(report=str(o/'report.md'),GP_MAE_C=v('GP_MAE_pp','C'),GP_MAE_G=v('GP_MAE_pp','G'),timing=branch_times,guide_gradient_min=min(gradients),guide_gradient_max=max(gradients),block_MSE_decreases=sum(r['old_teacher_MSE_end']<r['old_teacher_MSE_start'] for r in blocks))))
if __name__=='__main__':report(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else f.DEFAULT)
