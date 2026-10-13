"""Write Chinese R1 report from independently verified saved artifacts; append task status once."""
from pathlib import Path
import argparse
import csv
import difflib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from analyse.relative_metric_pilot import DEFAULT,read,write,file_hash
NAMES={'global_order':'全体顺序 C','local10_order':'局部10顺序','De_pp':'实际5邻居差 De','Dstar_pp':'最佳5邻居差 Dstar','E_pp':'额外邻居差距 E','Drandom_pp':'随机参照差','De_minus_random_pp':'实际减随机差','head_MAE_pp':'预测头误差','knn_MAE_pp':'5邻居均值预测误差'}
METHODS=['R0-8D','R0-1D','R1-8D']


def write_report(o):
    env=read(o/'environment.json');p=read(o/'protocol.json');runtime=read(o/'runtime_validation.json');ind=read(o/'independent_recompute.json')
    assert runtime['passed'] and ind['passed'] and runtime['cost']['effective_updates']==1200
    allmetrics=read(o/'evaluation/overall.json');probes={f.parent.name:read(f) for f in sorted((o/'trajectories').glob('*/probes.json'))}
    cover=read(o/'evaluation/coverage.json');seal=read(o/'all_inference_sealed.json');access=read(o/'evaluation_label_access.json')
    def metric(part,key,method,weight='candidate'):
        return next(r for r in allmetrics if r['partition']==part and r['metric']==key and r['method']==method and r['weighting']==weight)
    def fmt(x,n=6):return 'NA' if x is None else f'{x:.{n}f}'
    lines=['# R1 固定小实验：审查、实现与结果（2026-09-11）','',
    '**本轮完成：6条匹配续训轨迹、1200次实际更新，工程检查与独立复算通过。科学结果呈现小幅几何收益和预测取舍，不进入下一阶段。**','',
    'R1相对匹配的八维回归对照，在29个留出候选上，全体顺序平均提高0.000980，额外邻居差距降低0.186973个百分点；三个种子的两个方向均有利。相对同一回归模型的一维预测坐标，也分别提高0.002468、降低0.133027个百分点。不过R1预测头误差增加0.035008个百分点，三个种子均变差；5邻居均值预测虽平均略好，种子方向不一致。不能只凭几何变化宣布完整方法有效。','',
    '训练集方面，R1相对八维对照的全体顺序提高0.002512，固定顺序探针损失在三个种子均略低；但额外邻居差距反而增加0.031905个百分点，三个种子均变差。局部10顺序未稳定改善。因此只观察到部分顺序目标的变化，尚未证明精细邻居关系全面学成。与一维参照差异较小，也不足以确认多维表示有稳定实用优势。','',
    '范围严格限于原fold0内部112训练候选/60架构与29留出候选/16架构，架构互斥。29候选已多次用于开发，87条种子—留出记录仍只有29个候选；这是开发试验，不是新盲测或独立确认。没有新设通过百分比、显著性判断或事后改参数。GP误差、不确定性和搜索收益均为“本轮未评价”。','',
    '## 1. 实际位置与版本','',
    f'- 主机 `{env["host"]}`，项目/Git根 `{env["project"]}`；分支 `{env["branch"]}`；完整HEAD `{env["git_head"]}`。',
    f'- Python `{env["python"]}`，3.10.20；PyTorch `{env["torch"]}`、NumPy `{env["numpy"]}`。CUDA可用，两张RTX4500 Ada，驱动595.45.04；本轮实际单线程CPU，隐藏GPU。',
    '- 依据用户 `/Users/sherlockmore/Downloads/Codex_Server_R1_Audit_Implement_Pilot_Prompt.md`；原文副本在 `input_originals/`。最新任务文档、原阶段2、后续诊断、只读补充、相对几何和N2五折收尾报告均已读取。版本和原字节副本可追溯。',
    f'- 冻结协议身份 `{p["identity"]}`。当前未提交修改作为起点；`bo_phase4.py`、`eval_utils.py` 等旧工作保留，没有拉取、切分支、清理、提交或推送。',
    f'- 独立结果根 `{o}`。本报告相对文件名以该目录为根，项目源码路径另行标明。','',
    '## 2. 三组固定终点结果','',
    'R0-8D与R0-1D共享同一新R0终点和预测头；一维坐标是训练/留出候选各自由该模型预测的准确率，不是真实训练成绩。R1-8D仅增加固定顺序项。以下主结果先对每种子共同有效查询等权，再平均三种子；顺序为0–1，所有性能差/误差为百分点。','']
    for part,title in [('query','留出：每种子29个查询，参考112个训练候选'),('train','训练：每种子112个查询，逐查询排除自身，参考111个')]:
        lines+=['### '+title,'','| 指标 | R0-8D | R0-1D | R1-8D |','|---|---:|---:|---:|']
        for key,name in NAMES.items():lines.append('| '+name+' | '+' | '.join(fmt(metric(part,key,m)['mean']) for m in METHODS)+' |')
        lines+=['','逐种子配对（R1减对照）：','', '| 对照 / 指标 | 种子0 | 种子1 | 种子2 | 三种子均值 |','|---|---:|---:|---:|---:|']
        for control in METHODS[:2]:
            for key in NAMES:
                r=metric(part,key,'R1-8D minus '+control)
                lines.append(f'| {control} / {NAMES[key]} | '+' | '.join(fmt(v) for v in r['seed_values']+[r['mean']])+' |')
        lines+=['']
    lines+=['### 架构等权辅助结果','',
    '先平均同架构的HP查询，再平均架构；留出16架构。没有用架构等权替换主结果。','',
    '| 留出配对 | 顺序变化 | 额外差距变化，百分点 | 预测头误差变化，百分点 |','|---|---:|---:|---:|']
    for control in METHODS[:2]:lines.append(f'| R1减{control} | '+' | '.join(fmt(metric('query',key,'R1-8D minus '+control,'architecture')['mean']) for key in ['global_order','E_pp','head_MAE_pp'])+' |')
    lines+=['','辅助汇总仍呈现小幅几何改善及头误差增加。所有指标、种子范围、排除ID与两种汇总权重均在 `evaluation/overall.json`、`summaries.json`、`paired.json`；1269条逐查询明细在 `per_query.json`。训练和留出、三个种子和三个方法的表示不混合选邻居。','',
    '## 3. 起点、标签和实际训练规则','',
    '- 三起点为诊断目录 `tasks/regression-s{0,1,2}-mapping/M100.pt`。按原协议reuse、原started/completed任务身份及artifact哈希核对，不按日期或尝试编号猜模型。它们对应原内部回归任务：seed0为attempt01，seed1/2为attempt02。',
    '- 三个起点均为100步、原112/29完整ID顺序和同一候选架构/HP身份。原模型权重与诊断M100一致，保存预处理不重新拟合；训练表示、留出预测头和5邻居距离最大复现差全部0。训练标签标准化目标和SE²还原后与原检查点的训练target/noise最大差也全部0（容差1e-12）。',
    '- 标签只取旧只读补充已分离的 `training_labels.json` 112条；每候选GNN种子0/1/2均值与样本标准误逐项核验，容差1e-15。其余A/B及测试标签没有解析。留出均值只由最后独立评价进程读取。缓存编码12维、HP4和原float32变换保持一致。',
    '- 旧Adam状态完整存在，步数100。本轮明确对两边统一新建Adam、丢弃旧动量，不能写成“旧状态缺失”；旧M300不用于匹配终点。每种子从相同M100深拷贝两个独立分支，实测无共享参数存储。',
    '- 每分支200次有效更新，终点原100＋新增200＝300。Adam lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad=False、foreach=False、fused=False；更新HP分支、小映射和线性预测头。全112前向，无归一化、调度、早停或梯度裁剪。',
    '- R0使用MSE＋0.001大小项；R1加0.01顺序项。大小项严格为每候选表示平方和的均值。独立梯度检查确认R0与旧仅回归损失及每个参数梯度逐位一致，没有BCE残留梯度。','',
    '令 $\\tilde y_i=(y_i-\\mu)/s$ 使用保存的训练尺度，$\\delta_{ij}=|y_i-y_j|$。只保留三个不同ID且原始差距间隔大于 $10^{-12}$ 的关系，近/远方向不由噪声或ID改变。','',
    '$$\n m_{ijk}=\\min\\{ |\\tilde y_i-\\tilde y_k|^2-|\\tilde y_i-\\tilde y_j|^2,1\\},\\quad\n \\ell_{ijk}=\\max(0,m_{ijk}+\\|e_i-e_j\\|^2-\\|e_i-e_k\\|^2).\n$$','',
    '$$\n g=\\delta_{ik}-\\delta_{ij},\\quad u=2SE_i+SE_j+SE_k,\\quad w=\\frac{g^2}{g^2+u^2}.\n$$','',
    '每中心全局合法无序对抽8个，另在真实性能差最近10个其他训练候选内抽8个；组内无放回，组间允许重叠。每组取mean(w×hinge)，两组各半、112中心等权；空组记0，不改变分母，不除权重总和。SE只降权，不是置信度或显著性检验。','',
    '计划用独立PCG64、seed=20260911＋模型种子，完整200步在更新前保存。三份索引形状都是200×112×2×8×2；本次没有空组/不足8组。float64构造转float32最大间隔误差约2.98e-8，无正间隔消失；目标/权重误差见核验文件。全部680873个合法关系的一维真实成绩嵌入最大hinge为1.11e-16，符合数学可实现性；不据此保证网络必然学会。','',
    '## 4. 固定训练探针与实际更新','',
    '在新增0/1/50/100/200步均用同一第一步采样表。下表为固定探针，不能与每步随机表的损失曲线混称。梯度比仅在HP分支和小映射上计算，已乘0.01；预测头梯度另存，不进入该比分母。','',
    '| 轨迹 | 新增步 | 顺序损失 | 回归MSE | 激活比例 | 加权顺序/回归梯度比 | 梯度余弦 | 平均表示范数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for task,values in probes.items():
        for k in [0,1,50,100,200]:
            r=values[str(k)]
            lines.append(f'| {task} | {k} | '+' | '.join(fmt(r[key]) for key in ['order_loss','regression_loss','active_fraction','weighted_order_to_regression_ratio','gradient_cosine','embedding_norm_mean'])+' |')
    lines+=['','R1三个终点的固定顺序损失均略低于匹配R0；梯度存在，未发生意外断连。方向余弦可正可负，0步加权梯度比约0.023–0.044；所有节点原值、w分位数、头梯度及NA原因均保留。未按探针曲线调整权重或追加步数。R0的探针顺序梯度只作诊断，不进入优化。','',
    '6条轨迹各首步就是正式第1步。首步前/后总损失及分量已从实际检查点精确复算，参数最大变化与原日志一致；Adam实际计数1、缓存输入和旧模型不变。其余每步损失、系数、梯度、采样位置及全局更新ID均保存；全部有限。30个规定节点已实际重新加载模型和Adam，并前向复现对应预测，最大差0；没有额外优化步。','',
    '## 5. 审查发现、修改后证据','',
    '实现前表在 `audit_before.md`，早于新增代码和优化；最终表在 `audit_after.md`。独立审查为只读代理，未同时编辑文件或执行优化。旧入口与新R1不兼容，不代表旧N2试验本身违规；本轮没有发现足以动摇旧N2收尾结论的实质错误。','']
    audit='''| 要求 | 原代码/状态与实际后果 | 本轮修改及实际路径 | 验收证据/修改后状态 |
|---|---|---|---|
| 仅映射续训，无GP | offline_core.fit_predict会重拟合尺度、初始化并拟合GP；不符新协议 | 新relative_metric_pilot，直接恢复M100；复用JointMetricModel | runtime实际导入仅映射相关路径，未加载gpytorch/botorch/torch_geometric；旧源码未改 |
| 同划分/同标签/起点 | 原M100可恢复；初版audit只校验分离标签内部一致性 | 与原任务身份、gp_train_targets和固定训练noise交叉核验 | warmstart_checks：3/3，ID/架构/权重/表示/头/距离及训练目标差0 |
| 两边新Adam、独立参数 | 旧diagnose.restore_mapping会加载旧动量 | core.make_branches/fresh_adam，存实际param_groups和结构 | 旧动量存在但统一重置；无共享storage；11项检查和30节点恢复 |
| R1精确损失 | 旧metric_loss是BCE且除总权重，不可继承 | 新surrogate/relative_metric.py：legal_pairs/build_plan/order_terms/objective | 标量公式、正确/错误方向、梯度、并列、SE0、空/不足8组、降权比例、R0梯度逐位核对 |
| 隔离标签/随机流 | 旧context对象不宜直传 | validate_batch只接受显式无标签特征和训练白名单；独立PCG64固定全计划 | 禁用文件读取器及污染哨兵通过；实际trainer0留出读取；六轨迹先封存后分进程评价 |
| 可靠恢复/计费 | 初版节点非原子复制可留下半文件，step0中断可缺初始化证据 | freeze_copy原子写并核对既有字节；初始化对象先落盘；每步intent→atomic latest | 半文件拒绝测试；30实际节点恢复；completed重入0新增更新；1200账本与真实Adam步数一致 |
| 有效配置与真实路径 | 默认值/旧CLI不能代表R1实跑 | CONFIG固定且CLI冲突报错；protocol/input/source摘要、有效对象导出 | tests CLI冲突拒绝、effective_config及actual_initial_objects；源码冻结后训练未修改 |
| 三组公平评价 | 旧relative_geometry纯函数可用，旧run不支持新3组与训练自排除 | 新relative_metric_report，复用evaluate_query；R0-1D用模型预测 | 9空间1269条，所有引用仅训练池，自身不入参考；两套汇总权重 |
| 独立复算覆盖报告 | 初版复算漏最终overall校验 | 新relative_metric_recompute独立标准库实现；补最终均值/范围和8D来源比对 | 全部几何计数、误差、配对/汇总及30探针复算通过；无重训 |
'''
    lines+=audit.splitlines()+['',
    '## 6. 独立核验与实际成本','',
    f'- 11项零更新工程检查通过；首轮9项通过后增加原子节点与CLI冲突检查。30个真实诊断节点恢复通过，完成后同协议重入新增更新0。',
    f'- 标准库独立复算1269条记录、{cover["global_pairs"]}个全体参考对、30个固定探针/53760个采样关系。计数和邻居ID精确一致；全体/局部顺序、De/Dstar/E及两类预测误差最大差0，距离最大差8.88e-16，随机参照计算差最大3.55e-14个百分点。探针float64独立值与float32训练值最大差1.20e-8，小于1e-6容差。',
    f'- 性能并列排除{cover["performance_ties"]}对，距离并列0；全体/局部NA0、共同查询排除0、E小负数裁剪0。777万对不是独立样本量。',
    f'- 全部推理封存时间 `{seal["sealed_at"]}`；首次R1留出关联 `{access["time"]}`。训练与评价PID分别为{seal["training_process_pid"]}/{access["pid"]}；评价后全部封存文件哈希不变。',
    '- 审查期一次字段名错误导致audit中断，计0更新；一次本地python别名不存在导致补丁未立即应用，随后用python3修正并重新部署，仍0更新。原始失败及两次预训练版本副本保留。报告复算补强仅改核验脚本，训练源码/轨迹未改、未重训。','',
    '| 实际计算 | 数量 |','|---|---:|','| 完整有效小映射续训轨迹 | 6/6 |','| 每轨迹新增有效步数 | 200 |','| 有效/确认实际更新 | 1200/1200 |','| 废弃或重复更新 / 不确定预留 | 0 / 0 |','| 1260上限中的工程余量消耗 | 0/60 |','| 合成优化更新 | 0 |','| GNN / VAE / GP新增训练 | 0 / 0 / 0 |','| 新增搜索、补19项、阶段3–6、最终测试 | 均0 |','',
    f'正式运行内部耗时{read(o/"invocations/run01.json")["elapsed_seconds"]:.3f}秒；完成重入{read(o/"invocations/run02.json")["elapsed_seconds"]:.3f}秒。这是小映射及保存开销，不是GNN训练时间。所有优化意图在 `intents/`，每条轨迹的latest含模型/Adam/RNG/采样位置/已提交全局ID/计费；若意图与提交无法核实会保守预留，本次无此情况。','',
    '## 7. 已执行入口与交付','',
    '以下从服务器项目根执行；audit/check零更新，run仅恢复未完成轨迹，完整结果只读核对；report只读已保存推理结果。','',
    '```bash','cd /home/fangxy/D-VAE','export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=',
    '/home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_pilot.py --audit',
    '/home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_pilot.py --check',
    '/home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_pilot.py --config configs/relative_metric_r1_pilot_v1.json --run',
    '/home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_pilot.py --report',
    '/home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_verify_saved.py',
    '/home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_recompute.py results/posthoc/relative-metric-r1-pilot-20260911-v1',
    '/home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_write_report.py --write','```','',
    '代码保存在既有surrogate/analyse/tests/configs结构。`protocol.json`、`effective_config.json`、`input_sources.json`、`warmstart_checks.json`、`plans/`及源码/补丁给出身份；`trajectories/`包含6终点、30节点、损失/梯度、选邻和恢复状态；`evaluation/`包含全量明细及汇总；`independent_recompute.json`、`runtime_validation.json`、`logs/`和执行清单给出实测。`evidence_bundle.zip`及`RECOMPUTE.md`支持脱离服务器工程、用标准库复算。任务文档只追加第十四节。','',
    '## 8. 边界与停止','',
    '本轮工程范围完整完成，无待补轨迹。结论按预定类型属于“留出几何略好但预测头变差”的取舍；5邻居预测并非三个种子一致更好，训练细近邻也没有同步改善。不能宣布完整方法有效、搜索收益或多维空间的稳定额外价值。','',
    '是否值得另立固定的跨折确认需要后续评审；本轮不自动开展五折、GP、搜索或其他参数实验。不得据此把旧N2失败归因于已证明的三角不等式问题，也不推广成所有相对监督有效/无效。旧N2收尾、原五折负结果、640账本及19项缺口保持原样。阶段3–6的原有方法接口、真实评估/缓存、确认矩阵与先验证后测试入口仍待实现；R1小实验不替代这些任务。','']
    text='\n'.join(lines)
    report_path=ROOT/'docs/Fuzzy_Metric_R1_Pilot_20260911.md'
    report_path.write_text(text);(o/'report.md').write_text(text)
    (o/'audit_after.md').write_text('# R1最终要求—代码—实际证据\n\n训练前后状态分列，原审查表保留。\n\n'+audit)
    task=ROOT/'docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md';before=o/'input_originals/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md'
    prefix=before.read_bytes();current=task.read_bytes()
    addition=f'''

## 十四、R1相对顺序新版本固定小实验（2026-09-11追加）

**本轮工程执行完整完成；科学结果为小幅几何收益伴随预测取舍，不自动进入下一阶段。** 依据用户 `Codex_Server_R1_Audit_Implement_Pilot_Prompt.md`。追加起点SHA-256 `{file_hash(before)}`，前文原字节保留。

本节的R1是新“相对性能顺序损失”版本；R0是匹配的仅回归续训。二者不是第五节阶段3中“冻结/在线回归”方法简称的改写。旧N2收尾、原五折负结果、640账本及缺19项保持原样，阶段2不据此宣布通过。

| 本轮任务 | 实现与实际证据 | 状态 |
|---|---|---|
| 当前代码审查 | audit_before/after，旧fit_predict/恢复入口与新协议不兼容项单列 | 已完成；独立只读代理审查并修复标签来源绑定、原子节点与初始化恢复 |
| 正确起点/数据 | 原fold0内部112训练/60架构＋29留出/16架构；回归M100种子0/1/2 | 三起点权重/ID/预处理、表示/头/距离、原训练目标和SE核对差0；两侧架构互斥 |
| 新核心 | surrogate/relative_metric.py；JointMetricModel复用，固定相对顺序/SE降权/全局与局部采样 | 200步计划先冻结；双方统一新Adam；不保留BCE或总权重分母 |
| 运行/恢复 | analyse/relative_metric_pilot.py，configs/relative_metric_r1_pilot_v1.json | 6/6轨迹，各新增200有效步；实际更新1200，无失败/重复/不确定更新，重入0追加 |
| 验证/评价 | tests/test_relative_metric_pilot.py、relative_metric_verify_saved.py、relative_metric_report.py、relative_metric_recompute.py | 11项零更新检查、30真实节点恢复及全量标准库独立复算通过；9空间1269明细 |
| 报告与交付 | docs/Fuzzy_Metric_R1_Pilot_20260911.md；{o.relative_to(ROOT)} | 协议/计划/来源/有效对象/6轨迹/账本/预测/汇总/日志/补丁与便携证据包完整 |

留出主结果（29查询×三种子；C为0–1，误差/差距为百分点）：

| 方法 | C | 额外差距E | 预测头误差 | 5邻居均值误差 |
|---|---:|---:|---:|---:|
| R0-8D | 0.701707 | 6.354789 | 6.411259 | 6.212414 |
| R0-1D | 0.700219 | 6.300843 | 6.411259 | 6.231724 |
| R1-8D | 0.702687 | 6.167816 | 6.446267 | 6.093180 |

R1相比两个对照的C/E方向在三种子均改善，但预测头误差三个种子均增加，近邻预测方向不一致。训练集相对R0-8D的C提高0.002512，E反而增加0.031905个百分点；不能称细近邻目标全面学成。R0-1D与R0-8D头相同，不重复计算预测改进。29候选此前多次用于开发，不是87个独立候选或新的盲测；GP/不确定性/搜索本轮未评价。

实际命令已运行：`relative_metric_pilot.py --audit`、`--check`、`--config configs/relative_metric_r1_pilot_v1.json --run`（完整后重入0更新）、`--report`；`relative_metric_verify_saved.py`、`relative_metric_recompute.py <结果目录>`、`relative_metric_write_report.py --write`。均从服务器项目根使用nas环境Python执行，详细命令见新报告。

新增GNN/VAE/GP训练均0、合成优化更新0；正式小映射新增1200次实际/有效更新，1260上限中的60步工程余量未用。未补旧19项、未启动阶段3–6或最终测试。

**后续待定：** 是否另立固定跨折确认留待评审，不预设通过/最优参数。第5–8节阶段3–6既有代码缺口及完成标准保留；本轮没有实现或启动这些后续实验。当前R1结果仅提供开发取舍证据，不能改变旧N2结论或宣称搜索收益。
'''
    desired=prefix+addition.encode()
    if current not in (prefix,desired):raise RuntimeError('task document changed concurrently; refusing overwrite')
    task.write_bytes(desired);(o/'updated_stage_tasks.md').write_bytes(desired)
    (o/'task_document_append.patch').write_text(''.join(difflib.unified_diff(prefix.decode().splitlines(True),desired.decode().splitlines(True),fromfile=str(before),tofile=str(task))))
    # Flat CSV companions; complete nested evidence remains in JSON.
    for name in ['per_query','summaries','paired','overall']:
        rows=read(o/'evaluation'/f'{name}.json');keys=list(rows[0])
        with (o/'evaluation'/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    print(str(report_path));print('task document append verified; scientific result: geometry/prediction tradeoff')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=DEFAULT);a.add_argument('--write',action='store_true',required=True);write_report(a.parse_args().output.resolve())
