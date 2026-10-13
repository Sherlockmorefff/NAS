"""Write Chinese report and append-only Stage_Tasks record from verified results."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyse.relative_metric_pilot import read,write,file_hash

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def pct(x):return f'{100*x:.4f}%'
def cost(x):return '池内不可达' if x=='pool_unattainable' else str(x)
def make(o):
    a=o/'analysis';s=read(a/'summary.json');off=read(a/'offline_results.json');rep=read(a/'replay_results.json');fold=read(a/'fold_summary.json');pairs=read(a/'paired_results.json')
    runtime=read(o/'verification/runtime.json');ind=read(o/'verification/independent.json');assert runtime['passed'] and ind['passed']
    timing=read(a/'timing.json');env=read(o/'environment.json');p=read(o/'protocol.json');diagnostics=read(a/'training_diagnostics.json');traces=read(a/'champion_traces.json')
    br=[d for d in diagnostics if d['method']=='BR_HEAD'];trigger=sum(d['steps']*d['mean_nonzero_fraction'] for d in br)/10000
    loss0=statistics.mean(d['initial_loss']['ranking'] for d in br);losslast=statistics.mean(d['final_loss']['ranking'] for d in br)
    exact=['fold','seed','method','N','champion_rank']
    detail='# 固定矩阵完整结果\n\n折编号 0–4，模型种子 0–2。候选身份、原始分数及全排序保留在对应轨迹；不拼接跨折分数。\n\n## 离线逐折逐种子\n\n'
    detail+=table(['折','种子','方法','留出 N','冠军名次','前10命中','前10最好','差距/百分点','头 MAE/百分点','Spearman'],
        [[r[k] for k in exact]+[r['K']['10']['hit'],pct(r['K']['10']['best']),f"{r['K']['10']['gap_pp']:.6f}",f"{100*r['head_MAE']:.6f}",f"{r['spearman']:.6f}"] for r in off])
    detail+='\n## 固定辅助预算与随机理论参照\n\n任何并列冠军入围即算命中；本轮每折 N 均大于 20，辅助预算没有不适用项。随机参照按 $1-\\binom{N-|W|}{K}/\\binom{N}{K}$ 计算。\n\n'
    detail+=table(['轨迹','K','命中','最好','差距/百分点','随机理论命中率'],[[r['task'],k,r['K'][str(k)]['hit'],pct(r['K'][str(k)]['best']),f"{r['K'][str(k)]['gap_pp']:.6f}",f"{r['K'][str(k)]['random_hit_probability']:.6f}"] for r in off for k in [1,5,10,20]])
    detail+='\n## 回放全部起点\n\n所有起点初始均未命中冠军，均有刷新空间；没有重抽或删除起点。查询序号采用预先固定的批内排序，仅用于成本曲线，不影响同批五个选择。\n\n'
    detail+=table(['起点','方法','初始最好','初始命中','最终最好','最终命中','首次冠军','首次刷新','首次+0.5百分点','新选50自身最好'],
        [[r['seed'],r['method'],pct(r['initial_best']),r['initial_hit'],pct(r['final_best']),r['final_hit'],cost(r['first_champion_query']),cost(r['strict_improvement_query']),cost(r['improve_half_pp_query']),pct(r['new_selected_best'])] for r in rep])
    detail+='\n## 固定回放预算\n\n'+table(['起点','方法','新增次数','最好','最优差距/百分点','冠军命中'],[[r['seed'],r['method'],b,pct(r['checkpoints'][str(b)]['best']),f"{r['checkpoints'][str(b)]['gap_pp']:.6f}",r['checkpoints'][str(b)]['hit']] for r in rep for b in [5,10,20,30,40,50]])
    detail+='\n## 全部配对差\n\n差值为左侧方法减右侧方法。成绩和曲线均值单位为百分点；命中为 0/1 差。数组顺序为起点 2026091600–2026091604。\n\n'
    detail+=table(['配对','指标','五个差值','均值','中位数','胜/平/负'],[[pair,k,', '.join(f'{x:.6f}' for x in v['values']),f"{v['mean']:.6f}",f"{v['median']:.6f}",f"{v['win']}/{v['tie']}/{v['loss']}"] for pair,z in pairs.items() for k,v in z.items()])
    (a/'complete_tables.md').write_text(detail)
    trace='# 所有起点的冠军排序记录\n\n分析仅在回放全部完成后读取池内冠军身份；训练和选点进程没有这个标记。分数是未裁剪的原准确率预测值。冠军已揭示后记“已知”，不再参加候选排序。RANDOM 的名次是封存随机顺序的剩余名次，没有预测分数。\n\n'
    trace+=table(['起点','方法','批号','选择前新增数','表示版本','冠军名次','预测分数','本批命中'],[[r['seed'],r['method'],r['batch'],r['queries_before'],r['version'],
        '已知' if r['champions'][0]['already_known'] else r['champions'][0]['rank'],
        '—' if r['champions'][0]['score'] is None else f"{r['champions'][0]['score']:.9f}",int(r['champions'][0]['selected'])] for r in traces])
    (a/'champion_rankings.md').write_text(trace)
    audit=[
        ['当前工作区','AGENTS.md；environment.json；workspace_before.json','已有 bo_phase4.py、eval_utils.py 修改及未跟踪研究文件；不是干净默认分支','保留现场，不切换/拉取/重置','verification/preservation.json'],
        ['输入与候选身份','surrogate/fuzzy_metric.py: JointMetricModel；prepare；features.json','架构编码12维；HP为4→16；拼接后28→32→32→8，预测头8→1；HP顺序吻合','复用冻结编码、现有网络及身份；不改输入','data_checks.json；verification/runtime.json'],
        ['训练目标','surrogate/budget_champion_ranking.py: budget_loss/objective','新预算排名近似损失原先没有实现，属于本轮新增功能','稳定张量排序保留梯度；16 个无标签哈希列表','tests/test_budget_champion_ranking.py；verification/scalar_inputs.json；independent.json'],
        ['独立初始化','analyse/budget_champion_run.py: prepare/worker','旧回放的初始50及已训练共享模型不符合新协议；不是旧实验缺陷','封存未训练权重；C/BR独立复制；初始20单独拟合缩放','initial_weights/；440个检查点；runtime.json'],
        ['离线边界','原 stage2 partition.json；prepare；worker guard','每折必须隐藏另一侧架构成绩，不能用历史性能参数','复用原五折；只下发当前折训练标签，300步后封存分数','folds.json；offline/*/predictions.json；9项检查'],
        ['顺序回放','worker；run；旧 score_candidates/scaler_from_initial','旧入口不包含本轮剩余预算比例、直接头排名和初始20恢复协议','新增薄控制器；复用旧查询保存、缩放、模型、账本工具','replay/*/batches/*/selection.json、reveal.json'],
        ['隐藏成绩与禁止训练','worker guard；labels_for_controller；_label_arrays','标签仅允许原A专用三种子验证均值；小列表冠军不是未知池冠军','方法各自接收已选成绩；拒读完整A、其他轨迹及旧模型；不启用GP/GNN/VAE','隐藏标签扰动检查；ledger/；runtime.json'],
        ['恢复与计费','analyse/budget_champion_engine.py: fit；旧 Ledger/pause_once','新增损失需恢复其列表步号、Adam及已揭示顺序','逐步先记尝试、后记成功；实际中断后恢复','invocations.jsonl；recovery.jsonl；runs.jsonl'],
        ['科学含义','协议第3、8节','属于方法假设：小列表冠军入预算不能保证全池冠军命中','固定参数完整执行；不依据结果扩展实验','30条离线、15条回放；analysis/complete_tables.md'],
    ]
    report=f'''# 预算排名训练：本轮完成报告

已在服务器完成代码审查、最小实现、9项针对性检查、30条离线训练、15条历史回放、实际恢复检查和独立复算。**本设置下没有显示出总体选优收益。** 两组离线前10命中均为6/15；回放C命中1/5、BR命中0/5、随机命中2/5。BR没有提高任何一个起点的最终最好成绩，两个起点反而更低；两个起点达到相同成绩更早，属于查询次数减少。

## 主要结果

'''+table(['方法','离线冠军前10命中','回放冠军命中','回放平均最终最好','平均曲线高度（查询1–50）'],[[m, '—' if m=='RANDOM' else f"{s['offline'][m]['hit10']}/15",f"{v['champion_hits']}/5",pct(v['mean_final_best']),pct(v['mean_curve'])] for m,v in s['replay'].items()])+'''
池内冠军的原三种子验证均值为 81.5333%。本轮五个初始20均未命中，所以五个起点都保留在有刷新空间的分析中。

'''+table(['起点','C最终','BR最终','随机最终','BR相对C的实际变化'],[[seed,pct(next(r['final_best'] for r in rep if r['seed']==seed and r['method']=='C_HEAD')),pct(next(r['final_best'] for r in rep if r['seed']==seed and r['method']=='BR_HEAD')),pct(next(r['final_best'] for r in rep if r['seed']==seed and r['method']=='RANDOM')),desc] for seed,desc in zip(range(2026091600,2026091605),['低0.4个百分点；未命中冠军','最终成绩和曲线相同','同为80.5333%；从第33次提前到第24次达到','同为80.5333%；从第49次提前到第47次达到','低1个百分点；C第28次命中，BR到50次仍未命中'])])+'''
随机在起点603的第48次、604的第12次命中冠军。BR相对C的最终成绩差均值为−0.2800个百分点、中位数0，胜/平/负为0/3/2；曲线平均高度差为−0.07253个百分点，胜/平/负为2/1/2。BR相对随机的最终成绩差为−0.49333个百分点，胜/平/负为1/1/3。五条预定随机顺序不能证明随机普遍更优。

## 原五折结果

每折先汇总三个配对模型种子。五折训练/留出数为141/38、143/36、143/36、145/34、144/35。

'''+table(['折','C前10命中/3','BR前10命中/3','C冠军平均名次','BR冠军平均名次','两组前10最好差距/百分点'],[[r['fold'],r['methods']['C_HEAD']['hit10'],r['methods']['BR_HEAD']['hit10'],f"{r['methods']['C_HEAD']['mean_champion_rank']:.3f}",f"{r['methods']['BR_HEAD']['mean_champion_rank']:.3f}",f"{r['methods']['C_HEAD']['mean_gap10_pp']:.6f}"] for r in fold])+f'''
15个配对中的主命中和前10最好成绩全部相同。平均冠军差距均为1.73333个百分点。完整逐种子、辅助K=1/5/20、MAE、相关性和正确的随机无放回理论概率见 [完整结果表](analysis/complete_tables.md)。这些是五折内重复种子，不能当作15个独立数据集；各折冠军也不是整个A池冠军。

## 损失与冠军排名

仅新增“预算排名近似损失”：从当前已知集合确定性抽16个小列表，让其中至少一个最高真实均值候选的预测分数超过第k高非冠军。正式损失保持梯度，选择继续使用同一个预测头。固定间隔0.1和权重0.1；没有改输入、网络、起点或训练步数。

BR的40个训练事件中，平均每个列表约束触发比例（按训练步加权）为{trigger:.4%}；事件首步排名损失均值{loss0:.6f}，末步均值{losslast:.6f}。这些列表随步变化，因此首末均值只作诊断，不是固定列表上的收敛证明。80个C/BR训练事件均验证排名梯度连接到映射和预测头；C只记录该项，权重为0。排名约束在训练中实际起作用；其触发和梯度连接并未转化为本轮总体选优收益，不能推广为所有排名训练无效。

全体起点的每批冠军名次与预测变化见 [冠军排名记录](analysis/champion_rankings.md)；完整150批数据见 `analysis/champion_traces.json`，曲线见 [最好成绩曲线](analysis/best_so_far.png)。冠军身份仅在完成后的分析中加入。列表冠军不等于未知池冠军；这项近似损失没有全池预算命中保证，也不能把本轮负结果归咎于编码、映射或单个模块的唯一原因。

## 实际环境、代码审查与验证

服务器 `{env['host']}`，项目 `{env['project']}`，分支 `{env['branch']}`，完整HEAD `{env['HEAD']}`。Python `{env['python_version']}`，路径 `{env['python']}`；torch {env['dependencies']['torch']}、numpy {env['dependencies']['numpy']}。实际计算为单进程单线程CPU，GPU未参与。既有未提交修改均保留。

协议身份 `{p['identity']}`。附件原文和SHA记录在 `input_originals/Prompt.md` 与 `protocol.json`；本轮起始任务文档为 `baseline/Stage_Tasks.md`，SHA256 `3e72d0c63557c8142ee4a8a7262245bfdc7b9d4d6d664c8b8fd9134a11cbfe8b`。新原文与项目既有资料分别保存，不用聊天摘要替代依据。

'''+table(['要求','代码位置','发现的问题/性质','处理方式','验证证据'],audit)+f'''
实际检查了20,000步列表身份、440个检查点和130次候选坐标/全体评分；保存分数与相应模型重算逐位一致。一个训练中断在第1步保存后发生，恢复了当次Adam、模型、随机状态、列表步号及缩放；另一个在首批已选未揭示时发生，选点文件与恢复后查询顺序一致。完成后重入新增更新和查询均为0。

独立标准库程序从原三种子成绩、保存分数和选择ID重算30条离线与15条回放、765个曲线点、全部配对差和冠军排名，{ind['checks']}项数值/身份比较通过，最大指标差{ind['max_metric_error']:.3g}。另以Python标量核对{ind['scalar_real_lists']}个真实小列表及{ind['scalar_synthetic_cases']}个合成边界案例，最大损失差{ind['max_scalar_error']:.3g}。证据为 `verification/runtime.json`、`verification/independent.json` 和 `checks_complete.json`。工程执行通过与方法没有总体收益是两个独立结论。

## 成本与边界

正式更新尝试/有效均为20,000：离线9,000，回放11,000；失败、废弃、未确认更新均为0，未动用额外4,000修复额度。两次预定中断没有丢弃或重复训练步。两方法各10,000有效步，合成检查仅2次优化更新。历史新增逻辑查询750次；初始逻辑标签300条，按起点共享为100条，不等于全局唯一候选数。新增GNN/VAE/GP训练均为0。

首轮总墙钟{timing['first_run_wall_seconds']:.3f}秒：训练计算{timing['training_compute_seconds']:.3f}秒、评分{timing['scoring_seconds']:.3f}秒、查询IO {timing['oracle_io_seconds']:.3f}秒、训练保存与状态维护{timing['training_save_state_seconds']:.3f}秒，其余进程启动、选择保存及运行内核验{timing['other_startup_selection_save_verification_seconds']:.3f}秒。运行外只读核验{runtime['seconds']:.3f}秒，独立复算{ind['seconds']:.3f}秒；审查、实现和报告制作未混入算法时间。分方法耗时在 `analysis/timing.json`。有效训练预算未增加，排名项增加的运算及其诊断已计入实际耗时；C也计算排名诊断以便公平比较，但该项不进入C更新目标。

A为179个候选、96个架构，严格沿用同配方三个种子的验证均值。候选ID保留架构与完整HP，不按架构合并查询。同一历史标签在不同方法中各计自己的逻辑成本，真实部署获得这些标签仍需对应GNN训练成本。本轮没有解析B或测试成绩。VAE文件哈希未变，继承的限制仍在：原无监督预训练的精确图成员无法由旧记录完全恢复，不能声称已排除该来源问题。

A已被反复用于开发，当前冠军只是三个历史种子的观测均值冠军。此结果不构成B池独立确认、不保证未知池表现，也不作未执行的显著性结论。旧640次账本、19项缺口及旧负结论全部保留。本轮到此停止，不自动启动下一版或真实搜索。

## 已验证入口与交付

以下命令均从 `/home/fangxy/D-VAE` 执行，实际解释器为 `/home/fangxy/.conda/envs/nas/bin/python`：

```bash
/home/fangxy/.conda/envs/nas/bin/python analyse/budget_champion_run.py --prepare
/home/fangxy/.conda/envs/nas/bin/python -m pytest -q tests/test_budget_champion_ranking.py
/home/fangxy/.conda/envs/nas/bin/python analyse/budget_champion_run.py --run
/home/fangxy/.conda/envs/nas/bin/python analyse/budget_champion_report.py
/home/fangxy/.conda/envs/nas/bin/python analyse/budget_champion_verify.py
/home/fangxy/.conda/envs/nas/bin/python analyse/budget_champion_recompute.py --output results/posthoc/budget-champion-ranking-v1
/home/fangxy/.conda/envs/nas/bin/python analyse/budget_champion_write_report.py
```

`--run` 同时是恢复入口；当前完成状态下重入只核对成果，不新增更新或查询。`--prepare` 核对已封存协议，不重建起点。训练相关源码在 `source_snapshot/`，新增分析代码在 `analysis_source_snapshot/`；本轮差异为 `changes.patch`，旧成果保存检查为 `verification/preservation.json`。结果根目录为 `{o}`，任务文档只追加第19节；未提交、推送或覆盖旧成果。
'''
    (o/'report.md').write_text(report)
    doc=ROOT/'docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md';before=(o/'baseline/Stage_Tasks.md').read_bytes();old=doc.read_bytes()
    appendix=f'''

## 19. 2026-09-16 预算排名近似损失开发实验（budget-champion-ranking-v1）

本节是新增开发记录；保留前文0–6阶段定义、旧640次账本、19项缺口和历次结论，不回写为旧实验原先已具备的功能。不开展B池确认、不增加GNN/VAE/GP训练。

- **已验证事实**：实际服务器 ismc36，项目 `/home/fangxy/D-VAE`，分支 `{env['branch']}`，HEAD `{env['HEAD']}`；原A为179候选/96架构，原五折、冻结VAE与HP顺序核对通过。五个新起点严格取旧共同初始50原顺序的前20；全部封存后才下发标签。
- **本轮已实现**：`surrogate/budget_champion_ranking.py` 的确定性16小列表预算排名近似；`analyse/budget_champion_engine.py` 的分项记录、列表步号与Adam恢复；`analyse/budget_champion_run.py` 的独立未训练初始化、A专用标签隔离、固定30条离线及15条初始20回放。复用现有身份、冻结输入、网络、缩放、预测头和账本，旧共享模块未改。
- **已执行实验**：五折×3模型种子×C/BR，各300步；五起点×C/BR/RANDOM，20初始+50新增，10/20/30/40后各200步。总尝试/有效更新20000，新增逻辑查询750，合成优化2步，GNN/VAE/GP训练0/0/0。两次预定真实恢复均无重复计费，完成后重入0更新/0查询。
- **已验证结论**：离线前10命中C与BR均6/15；五折分布均0/3、3/3、0/3、0/3、3/3。回放冠军命中C 1/5、BR 0/5、随机2/5；平均最终最好为80.6667%、80.3867%、80.8800%。BR相对C最终差−0.28个百分点，胜/平/负0/3/2。602与603起点更早达到相同成绩；600和604最终更低，601相同。本设置没有显示总体选优收益，不推广为所有排名训练无效。
- **验证证据**：9项针对性检查；20000步列表身份、440检查点、130批全体评分复核；独立程序重算30离线/15回放，指标最大差{ind['max_metric_error']:.3g}，两类真实中断及完成重入检查通过。代码、报告、曲线、逐折逐起点全表、预算及状态在 `results/posthoc/budget-champion-ranking-v1/`。
- **已验证命令**：`python analyse/budget_champion_run.py --prepare`；`python analyse/budget_champion_run.py --run`（含恢复）；`python analyse/budget_champion_report.py`；`python analyse/budget_champion_verify.py`；`python analyse/budget_champion_recompute.py --output results/posthoc/budget-champion-ranking-v1`。解释器为 `/home/fangxy/.conda/envs/nas/bin/python`。
- **方法假设与待定事项**：小列表冠军、比例预算不等于未知池全局冠军命中概率；A重复开发及原无监督预训练图成员来源限制继续保留。是否改变损失/数据设计、开展独立确认，由后续新协议决定；本轮不加种子、不调参、不启动下一版。阶段2–6既有待办及停止条件保持原文；本轮工程无未解决阻塞，不代表研究目标通过。

详细审查表、所有原始记录和证据路径见 `results/posthoc/budget-champion-ranking-v1/report.md`。附件原文保存于该目录 `input_originals/Prompt.md`，协议身份 `{p['identity']}`。本次修订以前版SHA256 `3e72d0c63557c8142ee4a8a7262245bfdc7b9d4d6d664c8b8fd9134a11cbfe8b` 为前缀，仅追加本节。
'''
    wanted=before+appendix.encode();assert old==before or old==wanted,'unexpected concurrent task-document change'
    doc.write_bytes(wanted);(o/'Stage_Tasks.updated.md').write_bytes(wanted)
    write(o/'verification/document_append.json',dict(before_sha=file_hash(o/'baseline/Stage_Tasks.md'),after_sha=file_hash(doc),before_bytes=len(before),after_bytes=len(wanted),prefix_unchanged=wanted.startswith(before)),immutable=False)
    print(o/'report.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/posthoc/budget-champion-ranking-v1');make(p.parse_args().output)
