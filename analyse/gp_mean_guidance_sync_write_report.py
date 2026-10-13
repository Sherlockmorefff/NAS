"""Write the completed synchronized-guidance report and append task status."""
from pathlib import Path
import sys,json,statistics,csv,inspect,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse import gp_mean_guidance_sync_fivefold as f
from surrogate import gp_mean_guidance_sync as core

def table(headers,rows):
    fmt=lambda x: ('NA' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x))
    return '\n|'+' | '.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('|'+' | '.join(fmt(v) for v in row)+'|\n' for row in rows)+'\n'

def write_report(o):
    p=f.read(o/'protocol.json');v=f.read(o/'runtime_validation.json');ind=f.read(o/'independent_recompute.json');env=f.read(o/'environment.json')
    assert v['passed'] and ind['passed'];overall=f.read(o/'evaluation/overall.json');selection=f.read(o/'evaluation/selection_overall.json')
    def value(metric,b,scope='all',part='query',weight='candidate'):
        return next(r['value'] for r in overall if r['metric']==metric and r['branch']==b and r['scope']==scope and r['partition']==part and r['weighting']==weight)
    def select(metric,b,scope='all',part='query'):
        return next(r['value'] for r in selection if r['metric']==metric and r['branch']==b and r['scope']==scope and r['partition']==part)
    stats=[];timing=[]
    for branch in ['C','G','S']:
        parent=o/('trajectories' if branch=='S' else 'references')
        for path in sorted(parent.glob('*-'+branch+'/timing.json')):timing.append(dict(branch=branch,task=path.parent.name,**f.read(path)))
        rows=[r for r in timing if r['branch']==branch];stats.append(dict(branch=branch,**{k:sum(r[k] for r in rows) for k in ['mapping_update_seconds','gp_update_seconds','wall_seconds_this_invocation']}))
    meter={}
    for row in [r for r in timing if r['branch']=='S']:
        for category,values in row['solve_records'].items():
            dest=meter.setdefault(category,dict(conditionings=0,seconds=0.,cholesky_attempts=0,max_jitter=0.,nonzero_jitter=0))
            for key,val in values.items():dest[key]=max(dest[key],val) if key=='max_jitter' else dest[key]+val
    assert meter['formal_mapping_forward']['conditionings']==9000
    f.write(o/'timing_summary.json',dict(per_task=timing,group_totals=stats,sync_conditionings=meter,mapping_backward_seconds=sum(r['mapping_backward_seconds'] for r in timing if r['branch']=='S'),GP_parameter_fit_forward_backward_steps=3900,GP_saved_conditionings=195,validation=v['validation_solve_records']))
    probes=f.read(o/'actual_guidance_probes.json');refresh=f.read(o/'refresh_comparisons.json');blocks=f.read(o/'block_sync_errors.json');preflight=f.read(o/'real_preflight.json')['rows']
    def describe(key):
        vals=[r[key] for r in probes if r[key] is not None];return min(vals),statistics.mean(vals),max(vals)
    for name,rows in [('gradient_probes',probes),('refresh_comparisons',refresh),('block_sync_errors',blocks),('timing',stats)]:
        with (o/(name+'.csv')).open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
    seal=f.read(o/'all_predictions_sealed.json');reveal=f.read(o/'evaluation_label_access.json');maxerr=max(ind['max_absolute_differences'].values())
    report=r'''# GP参考与查询同步指导：五折开发实验（2026-09-15）

## 结论

**本轮按固定协议完整完成，但同步版没有形成一致的预测与候选选择收益。结束本版，不自动进入搜索，也不追加权重、周期、核、划分或输入对照。**

S的GP留出MAE为7.819337个百分点，较旧G增加0.162510，较仅回归C下降0.234117。相对G只有第2折、模型种子0改善；相对C只有第1、2折改善，收益以第1折为主，模型种子2反而更差。GP RMSE同时高于C/G。不能把平均MAE较C的小幅下降称为稳定推广收益。

GP前5真实均值S=72.708444%，比G低0.541333、比C低0.651556个百分点。预先固定的“前5中最好成绩”S=78.382222%，比G高0.768889，但仍比C低0.164444个百分点；与真实最佳5个的重合率也只恢复到26.67%，低于C的29.33%。排序相关性提高到0.548957，但不等于最优候选选择已改善。

预测头MAE为6.994078个百分点，比G差0.103866、比C好0.542034。预测头前5均值比G仅高0.064个百分点，主要由第3折带动；其前5中最好成绩反而比G低0.640、比C低0.684444个百分点。不能用预测头的某一个平均量替代GP选择结论。

95%观测区间覆盖率S=78.957169%，与G的78.770950%接近，仍低于C的86.405959%。区间覆盖不足的问题未解决，不直接启用不确定性探索。当前证据不值得优先另立搜索回放；B4＋P2主线不变。

本次同时改变参考坐标同步与参考端梯度，不能单独归因于其中一项。A已多轮用于开发，五折训练集重叠、候选和模型种子共享标签，1611条记录不是1611个独立实验。本轮没有统计显著或独立确认结论，也不能由此断言VAE必然丢失信息。

## 实际位置与身份

'''
    report+=f"主机 `{env['host']}`；项目 `{ROOT}`；分支 `{env['branch']}`；完整HEAD `{env['HEAD']}`。\n\nPython `{env['python']}`（3.10.20），PyTorch {env['torch']}，NumPy {env['numpy']}，GPyTorch {env['gpytorch']}，BoTorch {env['botorch']}。CUDA可用、2块GPU；本轮实际只使用单线程CPU，映射float32，GP和指导目标float64。\n\n"
    report+=f"最新Prompt原路径 `/Users/sherlockmore/Downloads/Codex_Server_GP_Synchronized_Guidance_Prompt.md`，SHA256 `{p['prompt_sha']}`；原文已存于 `input_originals/`。新协议身份 `{p['identity']}`，旧C/G协议身份 `{p['old_protocol']}` 实际核对通过。\n\n结果绝对路径 `{o}`。适用AGENTS、导入路径与运行进程见 `environment.json`；连接中断后的重新核对见 `reconnection_state.json`。只读准备连接曾超时，恢复时确认没有S训练更新，不计入优化预算。没有终止其他用户任务。既有 `bo_phase4.py`、`eval_utils.py` 修改和研究文件保留，不拉取、切分支、清理、提交或推送。\n\n"
    report+=r'''## 要求—当前实现—新增实现—验证证据

| 要求 | 原实现与审查发现 | 本轮新增实现 | 实际证据 |
|---|---|---|---|
| 保留C/G | 原协议绑定旧源码、结果有效 | 单独S入口和独立配置，封存30份对照引用 | audit_findings、old_reference_manifest、input_sources；旧协议哈希不改 |
| 原M100/原组/原尺度 | 15个净化warmstart可复用 | 核对原100步模型、ID顺序、scaler.fit_ids、目标/SE与旧表示 | 15项warmstart_checks；原45个第0块GP参数与S重新拟合完全一致 |
| 当前参考和查询完整梯度 | 旧condition会detach，符合旧G定义但不满足S | synchronized_mean每步重算K_SS、Cholesky、alpha，全图保留 | NumPy前向、完整有限差分、两侧分解、平移/旋转方向导数、重复点检查 |
| 预算/恢复不依赖旧CONFIG | 旧映射/GP保存工具可复用 | 新入口独立CONFIG和账本；复用无副作用数值及IO工具 | 3000/3900逐步账本，150映射节点与3900 GP步加载 |
| timing已存、completed未存 | 旧入口重复生成耗时可能触发不可变写冲突 | 新finalize_timing保留已存且一致的训练计时 | 零更新模拟、15个真实timing重入、正式运行入口重入0更新 |
| 新旧统一评价 | C/G已有封存预测和名单；旧报告未含最好值及重合率 | 从原名单统一复算新固定次指标 | 1611留出＋6444训练记录、90折/种子/方法/分区记录；独立复算 |

没有改变旧核心源码，也没有通过修改旧模块CONFIG绕过保护。同步训练的真实调用路径是 `surrogate.gp_mean_guidance_sync.objective → guidance → synchronized_mean → matern52/solve_cholesky`；`condition/from_gp`只用于GP拟合后身份存档和终点推理，不进入S训练损失。通用initial_objects工具中的历史函数定位不是本轮损失路径。

## 模型、目标、标签和数值规则

输入仍为冻结VAE均值编码12维与HP4；HP顺序为 `[log10(lr), dropout, log2(hidden_dim), log10(l2)]`。HP4→16，拼接架构12后28→32→32→8，再用8→1线性预测头。原15个 `fold{0..4}-refit-regression-s{0..2}` 仅回归M100起点各新增200步到M300；没有用旧C/G的M300初始化。原z/hp/y尺度和模型容量不变。

A179候选、96架构；五折训练/留出候选为141/38、143/36、143/36、145/34、144/35，同架构性能训练标签不跨外层边界。三个原validation训练种子的均值和样本SE逐条核验。VAE可能见过结构本身，不能说整个架构信息从未见过。

内部三组直接复用旧G：唯一架构按SHA256("gp_mean_guidance_v1:"+指纹)、指纹排序，位置模3。同一架构的全部HP配置随组排除。原M100和scaler已用全部外层训练标签，因此内部三组不是严格独立交叉拟合成绩。

$L=L_{reg}+0.001L_{size}+0.1L_{sync}$，其中 $L_{size}=mean[\|e\|_2^2]$；回归与指导均逐候选等权，指导不是三个不同人数组的简单等权平均。

$$
A_g=K_{\phi_g}(E_{S_g},E_{S_g})+diag[(SE/s_T)^2+\eta_g],\quad
\alpha_g=solve(A_g,\widetilde y_{S_g}-m_g).
$$

$$
\mu_g=m_g+K_{\phi_g}(E_{T_g},E_{S_g})\alpha_g,\qquad
L_{sync}=\frac{1}{|T|}\sum_i(\mu_{g(i)}-\widetilde y_i)^2.
$$

每步用同一次模型前向得到全部当前训练表示，再取参考/查询；GP参数、真实标签/SE和输入尺度冻结，当前参考、查询、K_SS、Cholesky和alpha均保留梯度。不显式求逆，不跨更新缓存计算图。使用vector_norm处理零距离，未增加距离地板或改变Matérn5/2核。

新增0/50/100/150步各保存当前表示的detached快照，从头拟合3个GP各20步，只复用其参数；每块50次映射更新不重置Adam。200步终点用完整外层训练表示从头拟合一个GP20步。映射Adam lr=.001、betas=(.9,.999)、eps=1e-8、weight_decay=0、amsgrad/foreach/fused=False，无调度、裁剪、早停和最好节点选择。

GP为原ReplayGP：float64、ConstantMean＋ScaleKernel各向同性Matérn5/2、FixedNoise；观测噪声$(SE/s_T)^2$，学习噪声初值$(10^{-4}/s_T)^2$、下界$(10^{-8}/s_T)^2$，min_fixed_noise=0。每次独立seed0，Adam lr=.05及展开默认设置见配置，20步。Cholesky先不加jitter，失败才按1e-6/1e-5/1e-4重试；本轮GP拟合、动态同步求解和独立条件化实际jitter全部为0。

准备器只读A专用导出与净化起点，worker只读本折训练标签和无标签特征；文件访问守卫实际拒绝本折留出标签、其他折标签和全A旧标签路径。没有读B或测试成绩、没有补旧19项、没有新增真实标签。冻结输入缓存与实际使用旧文件哈希前后核对。

'''
    report+=f"全部S预测、区间和前5名单封存于 `{seal['time']}`，统一评价读取标签于 `{reveal['time']}`。C/G引用清单在S训练前封存。旧C/G结果来自仅回归C与旧指导G，未把B4当C。\n\n"
    report+='## 总体留出结果\n\n主误差先按每种子179条唯一留出候选计算，再平均3种子；排名及前5先在各折/种子内计算，再平均种子、折。所有前5按预测降序、candidate_id升序封存，不使用P2或架构去重。\n'
    metrics=[('GP_MAE_pp','GP MAE（百分点）↓'),('GP_RMSE_pp','GP RMSE（百分点）↓'),('head_MAE_pp','头MAE（百分点）↓'),('head_RMSE_pp','头RMSE（百分点）↓'),('coverage95','95%覆盖率↑'),('width95_pp','95%宽度（百分点）')]
    report+=table(['指标','C','G','S','S−G','S−C'],[[label]+[value(k,b) for b in ['C','G','S','S-G','S-C']] for k,label in metrics])
    keys=[('GP_Spearman','GP排序相关'),('head_Spearman','头排序相关'),('GP_top5_true_mean_pp','GP前5真实均值（%）'),('head_top5_true_mean_pp','头前5真实均值（%）'),('GP_top5_true_best_pp','GP前5中最好（%）'),('head_top5_true_best_pp','头前5中最好（%）'),('GP_top5_gap_pp','GP与真实最佳5均值差（百分点）↓'),('head_top5_gap_pp','头与真实最佳5均值差（百分点）↓'),('GP_top5_overlap','GP与真实最佳5重合率'),('head_top5_overlap','头与真实最佳5重合率')]
    report+=table(['折内指标','C','G','S','S−G','S−C'],[[label]+[select(k,b) for b in ['C','G','S','S-G','S-C']] for k,label in keys])
    report+='\n预测不夹断到0–1。均值还原原y尺度；观测方差为潜在方差×尺度²＋额外噪声×尺度²＋本折训练SE²均值，区间为均值±1.96标准差，不使用隐藏候选自身SE。区间宽度增加并未带来足够覆盖恢复。\n\n## 逐折、种子和贡献\n'
    report+=table(['折','S−G GP MAE','S−C GP MAE','S−G GP前5均值','S−C GP前5均值','S−G GP前5最好','S−C GP前5最好'],[[i,value('GP_MAE_pp','S-G',f'fold{i}'),value('GP_MAE_pp','S-C',f'fold{i}'),select('GP_top5_true_mean_pp','S-G',f'fold{i}'),select('GP_top5_true_mean_pp','S-C',f'fold{i}'),select('GP_top5_true_best_pp','S-G',f'fold{i}'),select('GP_top5_true_best_pp','S-C',f'fold{i}')] for i in range(5)])
    seedvals={b:next(r['seed_values'] for r in overall if r['branch']==b and r['scope']=='all' and r['partition']=='query' and r['weighting']=='candidate' and r['metric']=='GP_MAE_pp') for b in ['C','G','S','S-G','S-C']}
    report+=table(['种子','C GP MAE','G GP MAE','S GP MAE','S−G','S−C'],[[i]+[seedvals[b][i] for b in seedvals] for i in range(3)])
    report+=table(['MAE权重','C','G','S','S−G','S−C'],[[weight]+[value('GP_MAE_pp',b,weight=weight) for b in ['C','G','S','S-G','S-C']] for weight in ['candidate','architecture']])
    report+='GP前5均值相对G在0/2/4折提高，但1/3折的下降更大；相对C仅第3折提高。头前5均值相对G仅第3折增加，第0折相同，其他3折下降。完整逐折、逐种子、各固定指标与候选明细见 `evaluation/paired.csv`、`selection_paired.csv`、`per_candidate.csv`，没有丢弃任何折。\n\n## 训练与同步目标\n'
    report+=table(['完整训练集上的终点模型','C训练','G训练','S训练','C留出','G留出','S留出'],[[label]+[value(k,b,part=part) for part in ['train','query'] for b in ['C','G','S']] for k,label in metrics])
    report+='训练汇总6444行含跨折重复支持和相同候选，独立于留出主指标。终点GP训练查询包含自身支持，不能把其低训练误差当推广。同步指导每个查询排除其整架构组，和终点完整训练GP是不同目标。\n'
    report+=table(['块','当前同步MSE块首均值','当前同步MSE块尾均值'],[[b,statistics.mean(r['start_sync_MSE'] for r in blocks if r['block']==b),statistics.mean(r['end_sync_MSE'] for r in blocks if r['block']==b)] for b in range(4)])
    increases=[r for r in blocks if r['end_sync_MSE']>=r['start_sync_MSE']]
    report+=f"\n同步MSE下降 {len(blocks)-len(increases)}/60 块；未下降块为 {[(r['task'],r['block']) for r in increases]}，完整保留。45个块刷新点固定同一当前支持/查询，旧参数MSE平均 {statistics.mean(r['old_parameters_current_support_MSE'] for r in refresh):.9f}，新参数 {statistics.mean(r['new_parameters_current_support_MSE'] for r in refresh):.9f}，变化 {statistics.mean(r['delta'] for r in refresh):+.9f}。见 `refresh_comparisons.csv`，实际核对前后模型权重和表示完全相同。\n\n这反映S的参数刷新变化较小，不能与旧G的旧参考块末目标混为同一种误差；也不能据此宣称单独证明参考同步或支持梯度有效。Prompt提供的旧45次只读分析仅作为动机，本轮未重复该分析。\n\n"
    report+='## 机制与恢复证据\n\n7项零更新检查5.14秒通过。独立NumPy前向最大差1.11e-16，完整有限差分最大差1.55e-10，预设容差atol=2e-6、rtol=2e-5、步长1e-6。支持/查询独立叶子的分量和与完整梯度差1.39e-17；平移梯度和3.69e-17，旋转方向导数0。对角线、重复、极近和全重合坐标前向与反向均有限；不改变原噪声规则。\n'
    report+=table(['135个实际探针','最小','平均','最大'],[[key,*describe(key)] for key in ['guidance_shared_norm','query_shared_norm','support_shared_norm','weighted_ratio','cosine','shared_decomposition_maxdiff','translation_gradient_sum_norm']])
    report+=f"\n135个真实探针已独立重算，HP与映射完整梯度有限；最后线性头对指导项直接梯度为None，头仍由回归训练。float32完整参数梯度与查询＋支持分量的最大差 {max(r['shared_decomposition_maxdiff'] for r in probes):.3g}；真实embedding平移梯度和最大 {max(r['translation_gradient_sum_norm'] for r in probes):.3g}，双精度小例另通过方向导数检查。零范数NA为 {sum(r['NA_reason'] is not None for r in probes)}；梯度不是Adam实际更新方向，没有据此调参。关闭指导的损失与全部参数梯度在15起点和旧C定义完全一致；旧G与S第0块前向最大差 {max(r['old_G_forward_maxdiff'] for r in preflight):.3g}，不要求梯度相同。\n\n"
    report+=f"150个关键映射节点实际恢复模型、Adam、RNG并前向一致；3900个GP拟合步骤状态实际加载，其中390个前向核对；195个真实GP由独立NumPy核矩阵与Cholesky公式验证均值/方差。15个真实timing文件重入保持原字节，完成运行入口重入映射/GP均0。节点区分刷新前/后；动态计算图每次重新建立，不保存为下一步旧图。\n\n独立标准库复算8055记录：1611留出（新S537＋旧C/G1074）、6444训练；覆盖全部固定误差、相关、前5均值/最好/重合率、区间及配对，最大数值差 {maxerr:.3g}，容差1e-10。封存名单与区间单位一致。\n\n"
    report+='## 预算、条件化与实测耗时\n'
    report+=table(['新增项目','有效/实际确认','废弃/重复/不确定','实际上限'],[['S映射更新',3000,'0/0/0',3150],['内部教师GP更新',3600,'0/0/0','GP总4095'],['终点GP更新',300,'0/0/0','GP总4095'],['GP总拟合/更新','195/3900','0/0/0',4095],['GNN/VAE训练','0/0','0/0/0','0/0'],['C/G重训','0/0','0/0/0','0/0']])
    report+=table(['组（15条轨迹合计）','映射更新秒','GP更新秒','总墙钟秒（含保存、推理、探针）'],[[r['branch'],r['mapping_update_seconds'],r['gp_update_seconds'],r['wall_seconds_this_invocation']] for r in stats])
    report+=f"\nS/C墙钟比 {stats[2]['wall_seconds_this_invocation']/stats[0]['wall_seconds_this_invocation']:.3f}，S/G比 {stats[2]['wall_seconds_this_invocation']/stats[1]['wall_seconds_this_invocation']:.3f}。C/G取旧保存计时，非同时运行的严格速度基准；三组标签和映射步数一致，但总计算量不同。\n"
    report+=table(['S条件化分类','实际次数','前向求解计时秒','非零jitter'],[[k,r['conditionings'],r['seconds'],r['nonzero_jitter']] for k,r in meter.items()])
    report+=f"\n9000次为3000次正式映射前向×3组，只含动态条件化；正式反向3000次，累计 {sum(r['mapping_backward_seconds'] for r in timing if r['branch']=='S'):.6f} 秒（包含整个损失反向，非仅求解）。训练中完整/分量探针另各405次条件化；运行后验证再各405次，求解前向累计 {sum(r['seconds'] for r in v['validation_solve_records'].values()):.6f} 秒，完整状态验证 {v['validation_seconds']:.6f} 秒。GP拟合3900次前向/反向和195次参数拟合后的条件化另计，不包含在9000中；独立矩阵检查195次。零更新预检总耗时单列5.14秒，不当作正式更新。细分记录见 `timing_summary.json`、各轨迹 `timing.json` 与 `runtime_validation.json`。\n\n"
    report+='## 实际可运行命令与文件\n\n以下命令均已在服务器执行，运行入口完成重入为0更新：\n```sh\ncd /home/fangxy/D-VAE\nCUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_sync_fivefold.py --audit\nCUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_sync_fivefold.py --check\nCUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_sync_fivefold.py --config configs/gp_mean_guidance_sync_fivefold_v1.json --run\nCUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_sync_verify.py\nCUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/gp_mean_guidance_sync_fivefold.py --report\n/home/fangxy/.conda/envs/nas/bin/python -S analyse/gp_mean_guidance_sync_recompute.py results/posthoc/gp-mean-guidance-sync-fivefold-v1\n```\n\n新增源码绝对路径：\n'
    sources=['surrogate/gp_mean_guidance_sync.py','analyse/gp_mean_guidance_sync_fivefold.py','analyse/gp_mean_guidance_sync_report.py','analyse/gp_mean_guidance_sync_verify.py','analyse/gp_mean_guidance_sync_recompute.py','analyse/gp_mean_guidance_sync_write_report.py','tests/test_gp_mean_guidance_sync.py','configs/gp_mean_guidance_sync_fivefold_v1.json']
    report+='\n'.join('- `'+str(ROOT/r)+'`' for r in sources)+'\n\n'
    report+=f"报告 `{ROOT/'docs/GP_Mean_Guidance_Sync_Fivefold_20260915.md'}`；任务文档 `{ROOT/'docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md'}` 仅追加第17节，保留旧C/G、N2/R1和19项记录。训练源码与输入身份在 `protocol.json/source_snapshot/`；对照引用在 `old_reference_manifest.json`；逐步优化账本在 `intents/receipts/update_ledger.json`；全部关键状态在 `trajectories/`；报告/复算日志在 `logs/`。`preservation_validation.json` 和源码差异记录最终保全检查；可携带 `evidence.zip` 附独立标准库脚本。\n\n本轮无缺失任务、无科学规则变更、无训练故障。150次映射与195次GP恢复余量未使用。同步版按原预算结束，不自动启动下一轮。\n"
    (o/'report.md').write_text(report);(ROOT/'docs/GP_Mean_Guidance_Sync_Fivefold_20260915.md').write_text(report)
    original=(o/'input_originals/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md').read_bytes();taskpath=ROOT/'docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md'
    marker='\n## 17. GP参考与查询同步指导：2026-09-15完成\n'
    addition=marker+f'''\n最新用户Prompt为 `Codex_Server_GP_Synchronized_Guidance_Prompt.md`，SHA256 `{p['prompt_sha']}`。本节追加，不覆盖第16节C/G及此前N2/R1负结果、阶段2原640次账本和19项缺口；输入对照仍暂缓，不是前置条件。

**已实施并验证**：独立S入口，不修改旧GP核心、运行器及旧配置；同15个仅回归M100、VAE12＋HP4、8维映射，各新增200步到M300；每步当前参考和查询完整梯度，每50步三教师20步，终点完整训练GP20步。只冻结GP参数与标签/尺度，Cholesky和alpha不detach。timing/completed之间中断的收尾在新入口修复。

**实际预算**：映射3000、GP拟合195次/3900步；实际=有效，废弃/重复/不确定0。GNN、VAE、C/G重训0。完成重入0更新，150个映射节点和3900个GP步实际加载，195个GP独立矩阵检查，135个真实完整/分量梯度探针。正式可微条件化9000次；7项零更新检查通过。全部S预测封存后统一评价，1611留出＋6444训练记录独立复算通过。

**实际结果**：S GP MAE7.819337个百分点，相对G +0.162510、相对C −0.234117；GP前5均值相对G −0.541333、相对C −0.651556个百分点；GP前5中最好相对G +0.768889、相对C −0.164444；95%覆盖78.957169%。同步MSE下降59/60块，但外层未形成一致收益。S总墙钟73.321900秒，历史C27.184534/G55.434091秒，含保存/探针、非同时基准。

**结论/下一步状态**：结束同步版，当前不建议优先另立搜索回放，不启用不确定性探索；不追加权重/周期/核/划分/输入实验，不改变B4＋P2。后续是否再提出方法须单独固定目标、协议及预算，当前没有待启动的已授权实验。本轮无关键阻塞或待补轨迹。

实际入口 `analyse/gp_mean_guidance_sync_fivefold.py`（--audit/--check/--run/--report）；核验 `analyse/gp_mean_guidance_sync_verify.py`；独立复算 `analyse/gp_mean_guidance_sync_recompute.py`；配置 `configs/gp_mean_guidance_sync_fivefold_v1.json`。源码位于 `{ROOT}`，完整已验证命令见 `docs/GP_Mean_Guidance_Sync_Fivefold_20260915.md`。

结果目录 `{o}`；协议身份 `{p['identity']}`；原C/G身份 `{p['old_protocol']}`不改。审查对照表、细分梯度/求解与耗时、S−C/S−G逐折种子明细、恢复记录、源码差异、旧文件哈希保全和证据包均见新结果目录。
'''
    if marker.encode() not in taskpath.read_bytes():
        assert taskpath.read_bytes()==original,'task document changed externally; preserve before append'
        taskpath.write_bytes(original+addition.encode())
    else:assert taskpath.read_bytes()==original+addition.encode()
    (o/'updated_stage_tasks.md').write_bytes(taskpath.read_bytes())
    f.write(o/'report_source_identity.json',dict(sources={rel:f.file_hash(ROOT/rel) for rel in sources},actual_loss_path=inspect.getfile(core.objective),original_task_doc_sha=hashlib.sha256(original).hexdigest(),updated_task_doc_sha=f.file_hash(taskpath)))
    print(json.dumps(dict(report=str(ROOT/'docs/GP_Mean_Guidance_Sync_Fivefold_20260915.md'),tasks=str(taskpath),cost=v['cost'],optimizer_updates=0)))
if __name__=='__main__':write_report(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else f.DEFAULT)
