from pathlib import Path
import argparse,csv,json,statistics,hashlib,datetime
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
read=lambda name:json.loads((root/name).read_text())
s=read('scoring/summary.json');seal=read('predictions_manifest.json');protocol=read('protocol.json')
ledger=read('resource_ledger.json');matrix=read('audit/matrix_integrity.json')
cost_summary=read('scoring/cost_summary.json')
assert matrix['pass'] and s['phase2_status']=='COMPLETE_VALID'
rows=list(csv.DictReader((root/'scoring/unit_metrics.csv').open()));costs=list(csv.DictReader((root/'scoring/cost_table.csv').open())) if (root/'scoring/cost_table.csv').exists() else []
def num(v,d=4):return '未定义' if v is None else f'{v:.{d}f}'
def metric(n,method,name):return s['per_n'][str(n)]['methods'][method]['metrics'][name]['mean']
def line(table):return '\n'.join('| '+' | '.join(map(str,r))+' |' for r in table)
main=[['已知条数','DKL','继续回归 REG','FROZEN','RAW16','随机5选期望','REG−DKL'],['---']*7]
for n in [20,50]:
 d=s['per_n'][str(n)]
 main.append([n,*[num(metric(n,m,'regret5_pp')) for m in ['DKL','REG','FROZEN','RAW16']],num(d['random5_expected_regret5_pp']),num(d['paired_regret5_improvement_pp']['REG']['mean'])])
pairs=[['n','比较基线','折0','折1','折2','折3','折4','五折均值','中位数','改善/持平/退步'],['---']*10]
for n in [20,50]:
 for comp in ['REG','FROZEN','RAW16']:
  d=s['per_n'][str(n)]['paired_regret5_improvement_pp'][comp]
  pairs.append([n,comp,*[num(d['fold_values_pp'].get(str(f))) for f in range(5)],num(d['mean']),num(d['median']),f"{d['improved_folds']}/{d['tied_folds']}/{d['worse_folds']}"])
secondary=[['n','方法','top1差距(pp)','MAE(pp)','RMSE(pp)','Spearman','90%覆盖率','90%宽度(pp)','95%覆盖率','95%宽度(pp)'],['---']*10]
for n in [20,50]:
 for m in ['FROZEN','REG','DKL','RAW16']:
  secondary.append([n,m,*[num(metric(n,m,k)) for k in ['top1_regret_pp','mae_pp','rmse_pp','spearman']],num(100*metric(n,m,'coverage90_fraction'),2)+'%',num(metric(n,m,'interval90_width_pp')),num(100*metric(n,m,'coverage95_fraction'),2)+'%',num(metric(n,m,'interval95_width_pp'))])
signals='\n'.join(f"- **{n}条：{s['scientific_signal_n'+str(n)]}。** 相对REG平均改善 {num(s['per_n'][str(n)]['paired_regret5_improvement_pp']['REG']['mean'])} 个百分点，{s['per_n'][str(n)]['paired_regret5_improvement_pp']['REG']['improved_folds']}/5折为正；相对RAW16 {num(s['per_n'][str(n)]['paired_regret5_improvement_pp']['RAW16']['mean'])} 个百分点；相对随机5选 {num(s['per_n'][str(n)]['random5_minus_DKL_regret5_pp'])} 个百分点。" for n in [20,50])
cost_table=[['n','FROZEN独立成本(s)','REG独立成本(s)','DKL独立成本(s)','RAW16独立成本(s)'],['---']*5]
for n in [20,50]:
 cost_table.append([n,*[num(cost_summary['per_n_method_standalone_cost'][str(n)][m]['mean'],3) for m in ['FROZEN','REG','DKL','RAW16']]])
variance_notes=[]
for n in [20,50]:
 reg_mae=metric(n,'REG','mae_pp');dkl_mae=metric(n,'DKL','mae_pp');imp=s['per_n'][str(n)]['paired_regret5_improvement_pp']['REG']['mean']
 relation='改善' if reg_mae>dkl_mae+1e-9 else '退步' if reg_mae<dkl_mae-1e-9 else '持平'
 choice='改善' if imp>1e-9 else '退步' if imp< -1e-9 else '持平'
 variance_notes.append(f"{n}条时，DKL相对REG的MAE{relation}（{num(reg_mae)}→{num(dkl_mae)} pp），regret5平均{choice}；两项各自报告，不互相替代。")
unit_ranges=[]
for n in [20,50]:
 vals=[float(r['regret5_pp']) for r in rows if int(r['n'])==n and r['method']=='DKL']
 unit_ranges.append(f"n={n} 的30个DKL单元 regret5范围 {num(min(vals))}–{num(max(vals))} pp，标准差 {num(statistics.stdev(vals))} pp（仅描述本协议内变动，不是独立确认或显著性检验）。")
readme=f'''# DKL阶段2：未见架构的固定池5选诊断

{signals}

本轮执行状态 **{s['phase2_status']}**：{s['endpoint_count']}/200个方法终点、60个共同初始化、344,000次正式优化器更新完成。工程有效与科学结论分开报告。

A池已反复用于研究开发；“未见架构”仅指当前拟合的训练子集没有该架构。这是固定查询折一次选5个的离线诊断，不是独立确认，也不是50次真实搜索。不能据此声称真实搜索收益或统计显著。

## 主要结果：前5选优差距

下面均为**百分点（pp）**，越小越好。每单元按GP均值选前5，按候选ID字典序破同分；先平均3个模型种子、再平均2个抽样种子、最后5折等权。RAW16每折/子集只跑一次。

{line(main)}

改善值定义为“基线regret5 − DKL regret5”，正值有利于DKL。不能把pp解释为相对百分比。

{line(pairs)}

分类严格采用运行前冻结的if/elif/else规则；“4/5折”是方向一致性描述，不是统计检验。完整逐模型/抽样种子结果保存在 `scoring/unit_metrics.csv`、`sample_seed_metrics.csv`、`fold_metrics.csv` 和配对表中。

## 误差、排序与校准

{line(secondary)}

{' '.join(variance_notes)}

映射方法的区间明显偏窄：DKL名义90%区间实际覆盖率在n20/n50分别只有 {100*metric(20,'DKL','coverage90_fraction'):.2f}% / {100*metric(50,'DKL','coverage90_fraction'):.2f}%，而RAW16分别为 {100*metric(20,'RAW16','coverage90_fraction'):.2f}% / {100*metric(50,'RAW16','coverage90_fraction'):.2f}%，代价是更宽的区间。这表明当前DKL不确定性在未见架构上校准不足；n50的top5收益不能解释成不确定性已经可靠。n50的DKL top1差距也比REG更大（{metric(50,'DKL','top1_regret_pp'):.4f} 对 {metric(50,'REG','top1_regret_pp'):.4f} pp），结论限于预定的top5指标。

潜在函数方差与观测均值方差分开保存。覆盖率采用 `y_scale² × (latent_variance_std + learned_extra_variance_std) + query_SE²`；额外噪声已是方差，不再平方，SE已是三种子均值标准误，不再次除3。查询SE仅由封存后的评分器读取。覆盖率及区间宽度是事后诊断，不表示实际搜索提前知道新候选SE；宽区间带来的高覆盖也不自动代表模型有效。

{' '.join(unit_ranges)}

模型种子波动与样本种子波动分别保留在 `sample_seed_metrics.csv` 的各指标 `_model_seed_sd` 列、`fold_metrics.csv` 的 `_sample_seed_sd` 列。训练折相互重叠，60个共同初始化不是60次独立确认。

## 固定设计、成本与训练诊断

5个架构折、2个抽样种子(20260922/20260923)、20/50条、3个模型种子(0/1/2)，共60共同状态及180映射终点；另20个RAW16终点。原分组查询候选数38/36/36/34/35；训练池141/143/143/145/144。20严格为50的前缀子集，不按成绩或SE筛选。

每个共同状态从新的随机C主体开始，回归1000步后冻结映射拟合GP300步。FROZEN再GP1000；REG回归1000后GP1000；DKL仅联合NLL1000，无事后GP收尾；RAW16绕过映射、常数初始化GP300+1000。三个映射方法均以自己的共同成本加分支和预测成本核算，实际执行则复用共同状态一次。模型/预测头/GP参数量为2328/9/4；DKL预测头冻结。

所有方法CPU/float64/1线程；Adam学习率mapping/head=.001、GP=.01，每块前80%原LR/后20%十分之一、联合global norm裁剪5、无weight decay/prior/added loss。单块保持一个Adam，不以十个100步块替代1000步。每100步记录训练诊断，仅固定最后一步用于评分，没有早停、调参、最佳checkpoint选择或追加训练。

实际正式训练 {ledger['training_seconds']:.3f} 秒；预测及分叉验证 {ledger['prediction_seconds']:.3f} 秒；另验证 {ledger.get('verification_seconds',0):.3f} 秒（{ledger.get('verification_updates',0)}次合成更新）。资源合计 {ledger['seconds']:.3f}/7200 秒，真实实验墙钟 {seal['resources']['experiment_wall_seconds']:.3f} 秒。首个正式n20/n50上下文用于估时并留在正式矩阵，没有额外真实试参。独立使用成本见 `scoring/cost_table.csv`/`cost_summary.json`；不同方法同更新数不代表等算力。

下表为每个单元独立使用时的平均拟合与预测成本；三个映射方法各自都包含完整共同训练成本。

{line(cost_table)}

本次共享共同初始化后的终点拟合与预测成本为 {cost_summary['completed_endpoint_fit_predict_seconds_with_shared_common_cost_once']:.3f} 秒；资源账本还包含 {ledger['training_seconds']+ledger['prediction_seconds']-cost_summary['completed_endpoint_fit_predict_seconds_with_shared_common_cost_once']:.3f} 秒共同状态分叉预测核验，因此总拟合与预测为 {ledger['training_seconds']+ledger['prediction_seconds']:.3f} 秒。没有数值失败、正式重跑或缺失单元。

训练诊断在 `training_traces.csv`，各块累计更新/耗时在 `block_summaries.csv`，资源账本在 `resource_ledger.json`。损失明确标为更新前loss，GP/表示诊断与state_after对应更新后状态；不以训练NLL下降证明未知选优改善，也没有把固定1000/300步解释成已证明收敛。终点核变化和约束距离保留供判断预算是否可能不足，未据此事后延长本轮。

独立末段诊断发现：DKL第900至1000步训练核相对变化，中位数在n20/n50分别为0.804%/1.861%，最大2.352%/3.668%；两个规模均有30/30单元的训练NLL仍下降。终末仍有可见变化，尤其n50，固定预算可能不足以让训练充分稳定。所有GP约束距离为正，7160条查询预测无非有限值或负潜在方差。详见 `audit/final_training_diagnostics.md`；这些训练诊断不替代留出选优和校准结论。

## 实现与证据

- 复用阶段01通过的同一C主体和精确GP数学核心，未复用任何阶段01已训练权重。
- 旧G/S使用GP预测均值误差指导，指导期间GP参数冻结；本次DKL用纯数据负边际似然联合更新映射与GP。旧运行提交信息不完整，本次未重做历史审计或声称完整复现旧实验。
- `protocol.json` SHA256：`{hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()}`。分组、20/50子集、60上下文/200终点清单、全部训练/评分源码及环境在释放本轮标签前冻结。
- 当前每次拟合只能收到其20/50条x/y/SE及无标签query x/ID；scaler仅拟合该20/50条。源A池标签hash固定，不能换成绩。
- 全部200预测和状态先封存为 `predictions_manifest.json`，独立评分器之后才取得 `scoring/labels_A.json`。预测文件没有查询真值或SE，评分器在读标签前验证封存、每个预测/终点文件hash、并拒绝INVALID/BLOCKED。
- 新增RAW16/n50、独立NLL/方差单位、查询标签/SE毒化、随机5选枚举、三层汇总/方向、块内Adam/LR、跨LR边界恢复与预算中断检查通过。评分器21项合成测试通过。
- 冻结前静态审查补齐了恢复后整块成本累计、无效封存拒绝和终点hash核验；针对性验证后再冻结，未接触真实查询评分，也没有真实训练重跑。
- 最终矩阵独立核验见 `audit/matrix_integrity.json`。语法和差异检查见 `verification/preflight_summary.json`。未运行全仓测试或重复阶段01的570步工程预检。
- 本地交付快照已仅用标准库复算评分，所有CSV、随机参考、成本汇总和除时间来源字段外的科学汇总在1e-12数值容差内一致，见 `audit/offline_scoring_reproduction.json`；跨Python版本出现末位舍入差异，不影响结论。这一步没有训练或重新预测。
- 子代理使用独立公式复核全部200个选中ID集合、指标/校准、三层汇总与成本，并对五折所有随机5选组合枚举核对，结论一致，见 `audit/final_scoring_review.json` / `.md`。

## 限制与下一阶段建议

本轮固定一个配方，没有内层调参；20/50分别报告，不择优隐藏。反复开发A池带来的研究层面适应风险仍存在；所有fold训练集重叠，不做朴素t检验、bootstrap或显著性宣称。随机5选只是同一查询折的精确机会参考，不算额外独立重复。

本轮n20未见收益、n50仅有混合信号，尚不足以宣称DKL总体更会选。建议下一步先讨论预先固定的小规模诊断性回放，保留REG与RAW16对照，区分序贯选点与一次静态5选的行为；本轮结果不用于回头调参或补跑。**本任务没有启动阶段3**，`next_stage_authorized=false`，等待用户决定方案。

没有读取B/test，没有训练或前向运行GNN/VAE，没有下载数据集、使用GPU、改动主线/阶段01或commit/push。

服务器目录：`/home/fangxy/D-VAE/artifacts/dkl-stage2-v1-20260922T080157Z/`。仅评分复算与按原协议重跑命令见 `REPRODUCE.md`。
'''
(root/'README.md').write_text(readme)
status={k:s[k] for k in ['phase2_status','scientific_signal_n20','scientific_signal_n50','next_stage_authorized','endpoint_count']}
status.update(formal_optimizer_updates=seal['resources']['formal_optimizer_updates'],resources=seal['resources'],B_test_access=False,GNN_VAE_training=False)
(root/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(status,ensure_ascii=False))
