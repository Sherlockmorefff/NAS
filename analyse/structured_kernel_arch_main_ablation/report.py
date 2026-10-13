import csv
import difflib
import json
import shlex
import shutil
import tarfile
from pathlib import Path
from .common import *
from .runner import counters


def rows(path):return list(csv.DictReader(path.open())) if path.exists() else []


def fmt(v):
    if isinstance(v,float):return f'{v:.6f}'
    if isinstance(v,list):return '['+', '.join(fmt(x) for x in v)+']'
    return str(v)


def table(data,keys):
    if not data:return '暂无完整有效结果。\n'
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in data)+'\n'


def write(root,name,value):(root/name).write_text(value,encoding='utf-8')


def commands(root):
    text='# 实际运行命令\n\n服务器 ismc36-ts，原仓库 /home/fangxy/D-VAE。每条由入口记录，内部子进程单独列出。测试命令与完整输出见 acceptance/。\n\n'
    for e in [json.loads(l) for l in (root/'commands.jsonl').read_text().splitlines()]:
        env=' '.join(f'{k}={shlex.quote(v)}' for k,v in e.get('env',{}).items() if v is not None)
        text+=e['time']+'\n\n```bash\ncd '+shlex.quote(e['cwd'])+'\n'+env+' '+shlex.join(e['argv'])+'\n```\n\n'
    write(root,'actual_commands.md',text)


def report(root,repo):
    a=root/'analysis';c=root/'C_preparation';states=read_json(root/'module_status.json')['modules']
    summary=read_json(a/'summary.json') if (a/'summary.json').exists() else {'state':'INCOMPLETE'}
    cs=read_json(c/'summary.json') if (c/'summary.json').exists() else {'state':'BLOCKED_C','scientific_state':'BLOCKED_C'}
    selection=read_json(root/'method_selection.json') if (root/'method_selection.json').exists() else {'method_selection':'INCONCLUSIVE','recommended_method':'W-I','requires_independent_confirmation':True}
    metrics=rows(a/'Q0_metrics_summary.csv');pairs=rows(a/'paired_comparisons.csv');traces=rows(a/'champion_trace.csv');fixed=read_json(a/'fixed_parameter_diagnostics.json') if (a/'fixed_parameter_diagnostics.json').exists() else []
    optimization=read_json(a/'optimization_summary.json') if (a/'optimization_summary.json').exists() else {}
    oracle=read_json(root/'oracle_summary.json') if (root/'oracle_summary.json').exists() else {}
    comp=summary.get('comparisons',{});primary=comp.get('W_HI_minus_W_I_n71',{});under=summary.get('underestimation',{})
    prediction='共同Q0的真值top10平均欠估量与MAE同时下降，支持本次初始化下尾部均值误差减轻；不等于偏差已消失。' if under.get('top10_underestimate_reduced') and under.get('top10_MAE_reduced') else '共同Q0的真值top10欠估量与MAE未同时下降，不能宣称低估已解决。'
    if primary.get('mean_pp',0)>1e-9 and not (under.get('top10_underestimate_reduced') and under.get('top10_MAE_reduced')):prediction='开发搜索改善，低估未解决。'+prediction
    elif primary.get('mean_pp',0)<=1e-9 and under.get('top10_underestimate_reduced') and under.get('top10_MAE_reduced'):prediction='预测诊断改善，尚未转化为搜索收益。'+prediction
    text='# 第四轮：独立架构主项消融的开发对照\n\n**本轮只研究A开发池，非独立C确认。** 新核W-HI删除独立架构主项，保留固定1/3 HP与2/3架构×HP项，其他拟合/采集预算与旧W-I一致。完整新协方差和逆矩阵重新计算，没有用旧均值减去旧架构贡献冒充新后验。\n\n'
    text+='主搜索结果（全部10组；百分点）：\n\n'+table([{'comparison':k,**v} for k,v in comp.items()],['comparison','mean_pp','CI95_pp','wins','ties','losses'])
    text+='\n主终点为n71 W-HI−W-I，其余为辅助。2000次配对bootstrap，seed=2026100631，所有比较共享十组重采样索引；这是固定A缓存和初始化的条件区间，不是新架构/新训练seed/独立数据集确认。胜平负按raw accuracy差绝对值≤1e-12为平。\n\n'
    text+=table(pairs,['seed','initial_contains_champion','W_HI_n70_pp','W_I_n70_pp','W_HI_n71_pp','W_I_n71_pp','W_HI_minus_W_I_n71_pp','W_HI_minus_RAW16_n71_pp','W_HI_minus_Random_n71_pp','W_HI_greedy_increment_pp'])
    text+='\n曲线积分均值（0..20缓存simple regret梯形积分/20）：\n\n'+table([{'method':k,'regret_integral_pp':v} for k,v in summary.get('regret_integral_means',{}).items()],['method','regret_integral_pp'])
    text+='\n冠军命中（含初始化；逐组首次query见analysis/champion_hits.csv）：\n\n'+table([{'method':k,'hits_n71':v} for k,v in summary.get('champion_hits',{}).items()],['method','hits_n71'])
    text+='\n第21槽为新旧GP的终端greedy，Random仍是第21次随机；终端查询ID和最终返回ID分别保存在 returned_candidates.json。最终返回始终取已观测71个中的最高均值，并按原完整ID规则破平。\n\n'
    text+='共同T0之后的Q0预测诊断（每组129个；恰好10个真值top10，独立scorer在所有正式轨迹封存后构造）：\n\n'
    text+=table(metrics,['method','top10_signed_bias_pp','top10_mean_underestimate_pp','top10_MAE_pp','top10_mean_latent_sd_pp','top10_mean_prospective_sd_pp','top10_prospective_coverage90','top10_prospective_is90_pp'])
    text+='\n'+prediction+'\n\n'
    text+='全Q0与预测top5：\n\n'+table(metrics,['method','mae_pp','rmse_pp','spearman','nlpd','coverage90','is90_pp','latent_coverage90','retrospective_nlpd','predicted_top5_true_mean_percent','predicted_top5_bias_pp','regret5_pp'])
    text+='\nNLPD用raw accuracy密度/nat，其余误差和宽度为百分点，coverage为比例。前瞻方差=latent+extra+median_T(SE²/s_y²)，再乘s_y²；Q的SE只用于retrospective。latent方差始终单列，未替换成单次GNN方差。原scorer的ceil(10%×129)=13指标保留为legacy_q_top10percent_bias_pp，不与本轮固定top10混淆。\n\n'
    text+=f"零优化同参数检查完成 {len(fixed)}/10：保留旧W-I五个物理参数，替换完整核并重新求C_new⁻¹；无optimizer调用，也未用于采集。旧n50的完整后验向量未单独存档，因此核验采用训练NLL和实际封存第1次qLogEI前向复现；n70均值另与封存greedy分数直接比较。误差与缺失原因逐状态记录，未重新拟合旧模型。\n\n"
    text+='固定参数结果只说明在旧物理参数处更改核组成的影响；正式W-HI还包含重新拟合。两者必须分开看，不能据Q结果挑固定参数。两个初始已观测冠军的2026100206/2026100208始终保留，但不算未知冠军修复。\n\n'
    initial=[r for r in traces if r['n']=='50']
    text+=table(initial,['seed','method','champion_observed','mu_percent','bias_pp','latent_sd_pp','remaining_mean_rank','actual_qlogei_rank','first_hit_query'])
    text+='\nchampion_trace.csv保存所有状态；unknown_champion_trace.csv只保留冠军尚未观测的状态。命中后自身标签驱动的拟合不算偏差修复，n70不报告第21次qLogEI。未知均值、std、实际采集排名、全十组搜索收益分别报告。未进行删点或换观测历史干预，不声称独立架构项是唯一失败原因；205旧架构项合计并非负值。\n\n'
    text+=f"研发建议：{selection.get('recommended_method')}；状态{selection.get('method_selection')}。严格按冻结规则：完整有效十组的平均n71差>1e-9pp，且平均regret积分不比W-I高超过1e-9pp，才推荐W-HI待独立验证。{selection.get('interval_note','')} 这不是科学PASS，不改写原BLOCKED或DEVELOPMENT_REPLAY_NO_STABLE_GAIN。\n"
    write(root,'A_RESULTS_AND_DIAGNOSTICS.md',text)
    history=read_json(c/'historical_use_registry.json') if (c/'historical_use_registry.json').exists() else {}
    feature=read_json(c/'extended_feature_manifest.json') if (c/'extended_feature_manifest.json').exists() else {}
    text='# 无标签C准备与来源限制\n\n本模块未读取、预测或采集评分任何C成绩；未训练GNN或更新VAE。\n\n'
    text+=f"状态：{cs.get('scientific_state')}。实际decoder调用 {cs.get('decoder_calls_started',0)}/5000；生成保留 {cs.get('generated_architectures',0)} 个新图候选、{cs.get('generated_configurations',0)} 个完整HP配置。已证实合格独立候选：{cs.get('eligible_configurations',0)}；架构：{cs.get('eligible_architectures',0)}。停止原因：{cs.get('generation_stop_reason','见错误记录')}。\n\n"
    text+='原192来自16384次标准正态latent解码产生的261个fresh unique中的抽样；它们不是整个合法空间的完整枚举。本轮沿用冻结JointSpaceVAE、六种原算子、1..5算子节点、原ordered DAG与静态合法性校验；未扩空间或按预测性能筛选。生成seed=2026100601，逐个确定性解码，保留真实latent来源和图；实际图的encoder均值与生成latent分开保存。\n\n'
    text+='global4沿用原四个HP regime，每个架构按SeedSequence([2026100601,zero_based_architecture_index,1])抽样；原float32实现将lr映射到10^[-4,-1.5]、dropout映射到[.1,.6]，hidden/L2按原离散支持与取整。实际候选落在原四个regime内，未使用预测挑HP；四个组合按实现后的lr/dropout/hidden/L2去重。\n\n'
    text+=table([{'outcome':k,'count':v} for k,v in cs.get('draw_outcomes',{}).items()],['outcome','count'])
    text+=f"\n历史索引共 {history.get('known_fingerprint_count')} 个指纹，可取得完整无标签图的身份数 {history.get('full_graph_identity_count')}；另有 {cs.get('missing_history_full_graphs')} 个仅指纹成员缺完整图。对可得图逐一使用真实有向带类型/属性精确同构；单指纹命中保守排除，不冒充已证明同构。未打开legacy混合成绩或test缓存来补图，后续用途记录也非穷尽证明。\n\n"
    text+='因此PROVENANCE_UNRESOLVED候选只作无标签准备存档，不能纳入独立C或因为不在旧192里就称未开发使用。A以外、重新编号或换HP都不建立独立性。无标签表示暴露本身与性能开发使用分开标记；保守排除清单不代表每个暴露图都已使用性能标签。\n\n'
    text+=f"WL编码状态：{cs.get('encoding_status')}。旧A各轮签名ID保留，新签名逐h0→h1→h2排序追加；按轮重映射旧phi坐标并补零，大小特征置于扩展基底末尾；n_max/e_max仍取A。A正式输入文件、尺度与GP拟合未使用这个扩展。\n\n"
    if feature:
        text+=table(feature['added'],['h','old_count','new_signature_count','C_OOV_node_occurrences'])
        text+=f"\n扩展psi维度 {feature['psi_dimension']}；旧A坐标嵌入完全一致={feature['A_coordinate_reembedding_exact']}，A距离最大误差 {feature['A_distance_max_abs_error']:.3e}；Matérn核最大误差 {max(feature['A_Matern5_2_kernel_max_abs_errors'].values()):.3e}；非同构WL碰撞对数 {len(feature['nonisomorphic_WL_collisions'])}。碰撞清单完整保留，不把WL作为完整图身份。\n"
    text+='\n全体存档C结构参与字典是transductive无标签结构准备；不称性能泄漏，也不声称模型从未见过C结构。字典冻结后遇新签名报编码阻断，不使用UNK或静默丢弃。未来HP/scaler/距离中位数必须仅从当时已观测T拟合。\n'
    write(root,'C_preparation_report.md',text)
    selected=selection.get('recommended_method','W-I');latent='L-HI（同1/3 HP+2/3 latent×HP核）' if selected=='W-HI' else 'L-I（与最终W-I同权重核）'
    text=f'''# 下一轮独立确认准备（未执行）

当前研发候选为 **{selected}**，需独立确认。先解决C的完整历史图/用途来源缺口；本轮存档的PROVENANCE_UNRESOLVED图不能直接转为独立确认池。没有因开发收益方向自动启动后续实验。

性能揭示前冻结最终方法、RAW16和纯Random。若要声称显式结构表示优于latent，应增加 **{latent}**，保持相同最终核、优化与采集预算；不能以不同核的旧latent结果替代。本轮没有运行此latent正式拟合。

候选生成/入选、精确历史同构排除、完整配置身份、无标签字典及编码应先封存。预注册Cora/public/NormalizeFeatures/full_batch/global4/No-LN/300/80与训练seed清单（示例R=3须下一轮明确冻结），明确先封存预测/决策再揭示缓存标签的oracle边界；不读取节点test成绩。全C无标签结构参与字典的暴露须披露，任何HP/scaler/距离标定只由已观测T决定。

预先冻结共同初始化50、20 qLogEI + 1终端greedy（Random为21个随机）、搜索seed清单和主终点n71返回最佳观测均值；预测评估另固定共享T/Q与top候选构造，后验区间仍以缓存seed均值为目标。停止条件、完整性与配对分析应在标签获取前写定，不能按C表现改核或增训练seed。

预算只作公式：完整缓存需 **去重完整配置U × 预定GNN seeds数R**；完整180×3为 **540个未来任务**，本轮为0。预测评估和搜索可以共享同一份预先冻结缓存，重复返回配置不能算独立训练测量。若仅小池试验，<71配置无法执行50+20+1无重复闭环；接近71会过度覆盖池，不能冒充大池条件。看小C成绩后再改方法则该池转为开发池，后续确认需另留新架构。

CPU代理预算公式为：需拟合的方法数×搜索组数×21状态×3冷起点；Random不拟合。训练耗时须根据下一轮设备及正式任务实测给出，不推测额外GPU或总小时数。Cora新架构确认不等于跨数据集或节点test确认。
'''
    write(root,'NEXT_INDEPENDENT_CONFIRMATION_PLAN.md',text)
    counts=counters(root);counts.update(qlogei_queries=sum(1 for p in (root/'trajectories').glob('*/decisions/*.json') if read_json(p)['kind']=='qlogei'),
        greedy_queries=sum(1 for p in (root/'trajectories').glob('*/decisions/*.json') if read_json(p)['kind']=='greedy'),
        A_logical_releases=oracle.get('logical_releases',0),unique_A_candidates=oracle.get('unique_candidates',0),
        physical_A_label_file_opens=oracle.get('physical_A_label_file_opens',0)+summary.get('physical_scoring_A_label_file_opens',0),
        fixed_parameter_checks=len(fixed),old_GP_refits=0,new_GNN_training=0,VAE_updates=0,C_performance_reveals=0,C_GP_predictions_or_acquisition_scores=0,B_performance_SE_reads=0,node_test_scores=0,KG=0)
    write_json(root/'summary.json',{'purpose':'A_development_single_kernel_intervention_and_unlabeled_C_preparation','independent_confirmation':False,
        'modules':states,'counts':counts,'search':comp,'Q0_diagnosis':under,'method_selection':selection,'optimization':optimization,'C':cs})
    baseline=read_json(root/'baseline.json');protected=baseline['protected_files']
    if any(sha256(repo/p)!=h for p,h in protected.items()):raise ValueError('User files changed')
    if read_json(repo/BLOCKED/'status.json')['state']!='BLOCKED' or read_json(repo/REPLAY/'summary.json')['state']!='DEVELOPMENT_REPLAY_NO_STABLE_GAIN':raise ValueError('Historical result state changed')
    write_json(root/'final_source_audit.json',{'user_files_unchanged':protected,'old_states_preserved':True,'frozen_A_input_unchanged':True,'time':now()})
    text='# 第四轮结果：核对照与无标签新候选池准备\n\n**A开发实验，非独立确认。本轮新增GNN训练、VAE更新、C性能预测/揭示、B成绩/SE读取、节点test评分、KG均为0。**\n\n'
    text+='目的：检验删除独立架构主项是否减轻低估并改善搜索；并在原合法空间实际生成无标签新图、审计独立C的来源和编码。\n\n'
    text+=table([{'module':m,'state':v['state']} for m,v in states.items()],['module','state'])
    text+=f"\n实际A预算：{counts['formal_fits']}/210拟合、{counts['logical_starts']}/630起点、{counts['qlogei_queries']}/200次qLogEI、{counts['greedy_queries']}/10次greedy、{counts['A_logical_releases']}/710逻辑释放；没有重跑W-I/RAW16/Random。固定旧参数诊断{len(fixed)}/10，无优化。\n\n"
    if primary:
        text+=f"主终点n71：W-HI−W-I平均 **{primary['mean_pp']:+.6f} pp**，条件95%区间 **{fmt(primary['CI95_pp'])}**，{primary['wins']}胜/{primary['ties']}平/{primary['losses']}负。所有十组均保留，含两组初始化已有冠军者。\n\n"
    text+=table([{'comparison':k,**v} for k,v in comp.items() if k.endswith('n71')],['comparison','mean_pp','CI95_pp','wins','ties','losses'])
    text+='\n'+prediction+' “均值改善”“std导致的探索变化”“观测自身标签后拟合”“搜索终点收益”分别分析；完整对照见 [A结果与诊断](A_RESULTS_AND_DIAGNOSTICS.md)。\n\n'
    text+=f"研发建议：**{selected}**（{selection.get('method_selection')}），仍需独立确认。{selection.get('interval_note','')} 不改写旧BLOCKED、DEVELOPMENT_REPLAY_NO_STABLE_GAIN或旧7/10门槛。\n\n"
    text+=f"C：实际{cs.get('decoder_calls_started',0)}次解码，保留{cs.get('generated_architectures',0)}个图/{cs.get('generated_configurations',0)}个配置；已证实独立可用配置{cs.get('eligible_configurations',0)}，状态 **{cs.get('scientific_state')}**。原192只是抽样池；历史仅指纹条目缺完整图与完整用途证据，不能把新生成或哈希未命中当独立证明。独立字典扩展状态{cs.get('encoding_status')}，A原特征不变。\n\n"
    text+='事实是完整封存的A新核对照及无标签图/编码；核干预改变整个协方差和拟合参数，不能据此确认唯一失败机制。bootstrap只描述固定缓存初始化波动。下一步先解决C来源缺口，再按[未来计划](NEXT_INDEPENDENT_CONFIRMATION_PLAN.md)预注册最终方法/RAW16/Random及必要的同核latent对照；未执行后续120/540次训练。\n\n'
    text+='优化、数据访问与工作计数：\n\n'+table([{'counter':k,'actual':v} for k,v in counts.items()],['counter','actual'])
    text+=f"\nAdam告警{optimization.get('adam_warnings')}，LBFGS预算告警{optimization.get('lbfgs_budget_warnings')}，失败起点{optimization.get('failed_starts')}；有限预算完成不证明全局最优。数值jitter事件与物理重算另记，不把Adam告警视为失败，也未追加预算。\n\n"
    text+='交付包括[协议](protocol.json)、[真实命令](actual_commands.md)、[代码diff](code_diff.patch)、[C准备报告](C_preparation_report.md)、[GPT分析请求](GPT_ANALYSIS_REQUEST.md)、[恢复验收](resume_verification.json)。模型/optimizer/RNG保留服务器，分享包保留足够复算的代码、参数摘要、预测/决策/观测及哈希。\n\n'
    text+=f"恢复命令：\n\n```bash\nssh ismc36-ts\ncd /home/fangxy/D-VAE\nOMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/fangxy/.conda/envs/nas/bin/python -B -m analyse.structured_kernel_arch_main_ablation run --run-dir {root.relative_to(repo)} --resume\n```\n"
    write(root,'README_RESULTS.md',text)
    request='''# GPT分析请求

请读取本包README、protocol、run_identity、source_manifest、A_RESULTS_AND_DIAGNOSTICS、method_selection及C准备材料，并把事实、解释、待验证假说分开。

1. 核精确为sf2[(1/3)kh+(2/3)ka kh]且5参数、显式分支、sf2单乘吗？是否完整重算C_new与后验，旧代码不变？210拟合/630起点/200qLogEI+10greedy是否完整，失败与告警是否如实处理？
2. 去独立架构主项改变了mu还是std？共同T0/Q0真值top10的欠估量和MAE是否同时改善，签名偏差、全池误差/校准是否付出代价？必须区分固定旧参数的10个零优化后验与正式重拟合W-HI，不能用Q选参数。
3. 预测变化是否转化为全十组搜索收益？按主n71 W-HI−W-I、RAW16/既定Random辅助、n70/greedy/AUC、同索引2000配对bootstrap评估，包括负收益seed和初始冠军。不能把重复返回当独立GNN测量、把条件区间称独立确认、把研发推荐当PASS。
4. 冠军尚未观测的轨迹是否真正改善？观察自身标签后posterior提升不得算修复；n70没有第21步qLogEI。201和205是否同方向？不要把205原架构项总和也说成负值；不作单一原因推断。
5. C是否实际新生成，原192是否只是抽样？精确图同构、历史指纹图缺口、未知用途与无标签结构暴露是否区分？PROVENANCE_UNRESOLVED是否被错误纳入独立C？append-only WL是否保留各轮旧ID，正确重映射A并保持距离/核等价、大小尺度不变？碰撞/OOV是否诚实记录？
6. C本轮零性能揭示/预测/采集评分、零GNN/VAE训练是否成立？下一轮至少最终方法、RAW16、Random；若主张结构表示优势，需要相同最终核的latent对照（W-HI不能用旧L-I代替）。训练预算U×R，180×3=540仅是未来计划。

核查测试、断点续跑零新增工作、旧用户代码和旧BLOCKED/NO_STABLE_GAIN状态保护。指出具体证据路径、严重问题和未能确认的事项；评价是否值得在来源真正独立的新C上继续，禁止看C成绩后调参、加seed救结果或重切旧池伪造独立性。
'''
    write(root,'GPT_ANALYSIS_REQUEST.md',request)
    patch=[]
    for rel in read_json(root/'run_identity.json')['code']:
        lines=(repo/rel).read_text().splitlines(keepends=True);patch.extend(difflib.unified_diff([],lines,fromfile='/dev/null',tofile='b/'+rel))
        destination=root/'code'/rel;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/rel,destination)
    write(root,'code_diff.patch',''.join(patch));commands(root)


def package(root,repo):
    commands(root);manifest={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix in ('.pt','.pth','.sqlite','.tmp') or p.name in ('SHA256SUMS','PACKAGE_MANIFEST.json','controller.lock') or p.name.endswith(('-wal','-shm')):continue
        manifest[str(p.relative_to(root))]=sha256(p)
    write_json(root/'PACKAGE_MANIFEST.json',{'files':manifest,'server_root':str(root),'omitted':'model/optimizer/RNG checkpoints and SQLite; checkpoint hashes and access ledgers are included'})
    manifest['PACKAGE_MANIFEST.json']=sha256(root/'PACKAGE_MANIFEST.json')
    write(root,'SHA256SUMS',''.join(f'{h}  {p}\n' for p,h in manifest.items()))
    path=root.with_suffix('.tar.gz')
    with tarfile.open(path,'w:gz') as tar:
        for rel in [*manifest,'SHA256SUMS']:tar.add(root/rel,arcname=root.name+'/'+rel,recursive=False)
    path.with_suffix(path.suffix+'.sha256').write_text(sha256(path)+'  '+path.name+'\n')
    print(json.dumps({'archive':str(path),'sha256':sha256(path),'bytes':path.stat().st_size,'files':len(manifest)+1}),flush=True)
