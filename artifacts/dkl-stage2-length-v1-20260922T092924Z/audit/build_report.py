from pathlib import Path
import argparse,csv,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
read=lambda x:json.loads((r/x).read_text())
s=read('scoring/summary.json');st=read('stability_summary/summary.json');seal=read('predictions_manifest.json');budget=read('resource_ledger.json')
assert s['execution_status']==st['execution_status']==seal['execution_status']=='COMPLETE_VALID'
assert read('audit/final_matrix_review.json')['pass'] and read('audit/final_scoring_review.json')['status']=='PASS'
def f(x,d=4):return '未定义' if x is None else f'{x:.{d}f}'
def metric(n,j,k):return s['per_n'][str(n)]['checkpoints'][str(j)]['metrics'][k]['mean']
def pair(n,a,b):return s['per_n'][str(n)]['comparisons'][f'{a}_to_{b}']
def checkpoint(n,j):return st['per_n'][str(n)]['checkpoints'][str(j)]
def table(rows):return '\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)
def dirs(d):return f"{d['improved']}/{d['tied']}/{d['worse']}"
lead=[]
for n in [20,50]:
 d=pair(n,1000,4000);c=checkpoint(n,4000);p=s['per_n'][str(n)]['checkpoints']
 lead.append(f"- **n={n}：** 1000→4000的regret5平均改善 **{f(d['metrics']['regret5_improvement_pp']['mean'])} pp**，30条轨迹改善/持平/退步为 **{dirs(d['trajectory_regret5_directions'])}**；gain5平均变化 **{f(d['metrics']['gain5_improvement_pp']['mean'])} pp**。4000步实用稳定 **{c['practical_stability_counts']['stable']}/30**；打破训练集已知最佳的轨迹由 **{p['1000']['broke_training_best_trajectories']}/30** 变为 **{p['4000']['broke_training_best_trajectories']}/30**。")
main=[['n','联合步数','regret5(pp)↓','gain5(pp)↑','打破训练最佳','查询有提升空间','查询提升上限(pp)','随机gain5(pp)'],['---']*8]
for n in [20,50]:
 for j in [1000,2000,4000]:
  d=s['per_n'][str(n)]['checkpoints'][str(j)]
  main.append([n,j,f(metric(n,j,'regret5_pp')),f(metric(n,j,'gain5_pp')),f"{d['broke_training_best_trajectories']}/30",f"{d['query_has_improvement_room_trajectories']}/30",f(metric(n,j,'query_gain_upper_bound_pp')),f(metric(n,j,'random_expected_gain5_pp'))])
pt=[['n','步数变化','regret5改善(pp)','regret改善/平/退','gain5变化(pp)','gain改善/平/退','top5集合变化'],['---']*7]
foldtable=[['n','步数变化','折0','折1','折2','折3','折4','均值','中位数'],['---']*9]
for n in [20,50]:
 for x,y in [(1000,2000),(1000,4000),(2000,4000)]:
  d=pair(n,x,y)
  pt.append([n,f'{x}→{y}',f(d['metrics']['regret5_improvement_pp']['mean']),dirs(d['trajectory_regret5_directions']),f(d['metrics']['gain5_improvement_pp']['mean']),dirs(d['trajectory_gain5_directions']),f"{d['top5_set_changed_trajectories']}/30"])
  foldtable.append([n,f'{x}→{y}',*[f(d['fold_regret5_improvement_pp'][str(k)]) for k in range(5)],f(d['metrics']['regret5_improvement_pp']['mean']),f(d['metrics']['regret5_improvement_pp']['median'])])
stable=[['n','步数','全部4项稳定','目标稳定','核稳定','均值稳定','两种SD稳定','top5集合稳定','top1稳定'],['---']*9]
for n in [20,50]:
 for j in [2000,4000]:
  d=checkpoint(n,j)
  vals=[d['practical_stability_counts']['stable'],*[d['criteria_counts'][k]['stable'] for k in ['objective_small','similarity_stable','mean_prediction_stable','uncertainty_stable']],d['top5_selection_counts']['stable'],d['top1_selection_counts']['stable']]
  stable.append([n,j,*[f'{x}/30' for x in vals]])
secondary=[['n','步数','top1差距(pp)↓','MAE(pp)↓','RMSE(pp)↓','Spearman','90%覆盖','90%宽度(pp)','95%覆盖','95%宽度(pp)'],['---']*10]
for n in [20,50]:
 for j in [1000,2000,4000]:
  secondary.append([n,j,*[f(metric(n,j,k)) for k in ['top1_regret_pp','mae_pp','rmse_pp','spearman']],f(100*metric(n,j,'coverage90_fraction'),2)+'%',f(metric(n,j,'interval90_width_pp')),f(100*metric(n,j,'coverage95_fraction'),2)+'%',f(metric(n,j,'interval95_width_pp'))])
control=[['n','原固定方法（未续训）','regret5(pp)','gain5(pp)'],['---']*4]
for n in [20,50]:
 for m in ['REG','FROZEN','RAW16']:
  d=s['fixed_controls']['methods'][m][str(n)]['metrics']
  control.append([n,m,f(d['regret5_pp']['mean']),f(d['gain5_pp']['mean'])])
windows=list(csv.DictReader((r/'stability_summary/diagnostic_points.csv').open()))
changes=[]
for n in [20,50]:
 lookup={(row['context_id'],int(row['total_joint_step'])):row for row in windows if int(row['n'])==n}
 ids={cid for cid,j in lookup}
 delta=[float(lookup[cid,4000]['nll_after_update'])-float(lookup[cid,1000]['nll_after_update']) for cid in sorted(ids)]
 changes.append(f"n={n}：更新后训练NLL的1000→4000均值变化 {f(sum(delta)/len(delta),7)}，{sum(x<0 for x in delta)}/30条下降；DKL均值选出的top5集合有{pair(n,1000,4000)['top5_set_changed_trajectories']}/30条改变。")
spreadtable=[['n','步数','最后3窗口最大查询均值漂移(pp)：中位数/95分位/最大','合并核相对变化(%)：中位数/95分位/最大'],['---']*4]
for n in [20,50]:
 for j in [2000,4000]:
  d=checkpoint(n,j)['trajectory_worst_of_last_three_metric_spread']
  spreadtable.append([n,j,' / '.join(f(100*d['query_mean_max_abs_change_raw'][k]) for k in ['median','q95','max']),' / '.join(f(100*d['combined_kernel_relative_change'][k]) for k in ['median','q95','max'])])
interpret=[]
for n in [20,50]:
 d=pair(n,1000,4000);imp=d['metrics']['regret5_improvement_pp']['mean'];g=d['metrics']['gain5_improvement_pp']['mean'];stab=checkpoint(n,4000)['practical_stability_counts']['stable']
 direction='改善' if imp>1e-9 else '退步' if imp< -1e-9 else '持平'
 interpret.append(f"n={n}在加长到4000步后，regret5平均{direction}，但具体变化必须结合上述{dirs(d['trajectory_regret5_directions'])}条轨迹及逐折分布判断；gain5平均变化为{f(g)} pp。{'全部30条达到本轮实用稳定标准' if stab==30 else f'仍有{30-stab}条未达到本轮实用稳定标准'}。")
readme=f'''# DKL训练长度与预测稳定性诊断

**加长本轮低学习率尾段没有改善选优，也没有增加打破当前最佳的收益。** n20的30条轨迹regret5/gain5全部不变；n50有2条regret5退步、28条持平，没有改善轨迹。原阶段2结论没有因这次加长而转为更强的正向证据。

{chr(10).join(lead)}

执行状态 **COMPLETE_VALID**：60/60条原DKL轨迹连续续训完成，新增 **180,000** 次正式更新，保存 **120** 个2000/4000新终点；加上原1000步共180个科学评估单元。本轮没有改变原阶段2报告及其“n20未见收益、n50局部/混合信号”的历史分类。

这是A池原结果已经曝光后的开发性补充。训练长度敏感性、实用稳定与方法有效性分开判断；稳定不表示数学收敛，也不表示选得正确。4000步结果不能与未续训的REG宣称等预算优胜。

## 选优与打破当前最佳

单位为百分点（pp），不是相对百分比。regret5越小越好；gain5=max(0,所选5个中的最好成绩−训练子集已知最好成绩)，越大越好。查询提升上限仅针对当前查询折，不代表整个A池或搜索空间。

{table(main)}

随机5选regret5五折等权期望为 {f(metric(20,1000,'random_expected_regret5_pp'))} pp；随机gain5按当前训练子集的已知最佳分别精确计算。相同参考没有计作额外独立重复。

改善方向：regret5为早期减晚期；gain5为晚期减早期。判同分用未四舍五入的1e−9 pp容差。

{table(pt)}

逐折regret5改善如下；正值有利于加长训练：

{table(foldtable)}

先平均同折同抽样种子的3个模型种子，再平均2个抽样种子，最后5折等权。完整候选名单、最好候选ID/成绩、逐轨迹、两种种子分解与配对表保存在 `scoring/`；不把30条轨迹或180个时刻当独立确认，不做显著性检验。

## 哪些轨迹达到实用稳定

连续最后三个100步窗口必须同时满足：NLL相对变化≤0.001；训练核与训练＋查询合并核相对变化均≤1%；训练/查询均值最大变化均≤0.1 pp；查询潜在SD和模型SD最大变化均≤0.1 pp。四项分别报告，不隐藏失败项。

{table(stable)}

1000步缺少相同口径的历史预测窗口，稳定性为“历史信息不足”，没有回填或反向重训。top5采用集合稳定，内部排名改变另报；top1单独统计。未稳定轨迹ID完整列在 `UNSTABLE_TRAJECTORIES.md` 及 `stability_summary/summary.json`。

{table(spreadtable)}

这里先取每条轨迹最后3窗口中的最大漂移，再对30条汇总；核为相对百分比，均值为准确率百分点，两者不混用。完整8项漂移的范围、25/50/75/90/95分位数与最大值见 `stability_summary/summary.json`，1860时刻与1800窗口保留在CSV。

{' '.join(changes)}

NLL是在同一更新后状态重新计算；最近一步梯度明确属于更新前量。训练NLL下降不能替代留出选优收益，也没有因为某条达到稳定而提前停止或因不稳定继续到8000步。

## 误差与区间校准

{table(secondary)}

训练诊断中的latent SD与model SD不含query_SE；只在独立评分器中按 `y_scale²×(latent_variance_std+extra_variance_std)+query_SE²` 计算观测均值覆盖，额外噪声已是方差，SE不再除3。潜在原方差保留，负舍入仅在开平方时截为0。覆盖与宽度共同解释；预测稳定仍可能严重过度自信，不能默认方差已经适用于EI/KG。

n50的top1差距由8.8333降至8.5533 pp，但top5 regret5、MAE/RMSE和覆盖率退步；单项top1改善不能替代本轮主要选优指标或gain5。两个规模的gain5均低于各自随机5选期望，且未随追加训练增加。

## 原阶段2固定预算参照，未续训

{table(control)}

这些结果由原封存预测只读复算。REG/FROZEN/RAW16没有追加更新，原共同训练也没有重新运行。它们是背景参照，不是与4000步DKL配平的对照；若后续值得跟进，需要另立协议配平表示/GP更新预算，本任务未执行。

## 协议、验证与资源

- 新协议SHA256：`{hashlib.sha256((r/'protocol.json').read_bytes()).hexdigest()}`；源清单、60条上下文、代码和环境在新增正式更新前冻结。
- 每条从原 `checkpoints/<context>_DKL_joint.pt` 的完整Adam恢复，保留样本/顺序/scaler/x/y/known、实际各参数step和moments。先验证原均值/方差及数学状态，再新增更新。未用预测专用state替代训练checkpoint。
- 追加1001—4000步始终沿用映射1e−4/GP1e−3；2000保存沿同一Adam继续。原scheduler仍标planned1000，尾段单独记录fixed_endpoint_lr；线性头冻结，无目标或网络改动。
- 数值设置保持CPU/float64/1线程、jitter1e−8且只尝试一次、联合global norm裁剪5、纯数据每样本NLL。诊断使用独立数学副本，RNG/模式/cache/训练输入/Adam隔离。
- 合成验证发现GPyTorch默认Softplus变换模块共享导致主实例子模块mode被切换；冻结前已将副本共享模块分离，保留严格断言。恢复与跨2000保存测试、修复后的诊断路径/毒化/稳定性单位测试、13项评分测试通过。失败、定位和复测共1071次合成更新，均计入预算，未发生正式重跑。
- 所有新预测先统一封存，之后才释放独立A评分标签。1000直接引用原封存预测；其恢复重算仅用于一致性验证；仅1000/2000/4000评分，未挑最佳时刻。

本轮正式训练 {budget['training_seconds']:.3f} 秒，诊断/基线预测 {budget['prediction_seconds']:.3f} 秒，合成及恢复验证 {budget['verification_seconds']:.3f} 秒；合计 **{budget['seconds']:.3f}/1200秒**，实验墙钟 **{seal['resources']['experiment_wall_seconds']:.3f}秒**。原阶段2的344000次更新和耗时没有重复计入。各2000/4000成本为截至该点累计，不相加；全部实际成本以资源账本为准。磁盘写入、只读审计、复算与打包不混入拟合预算。

账本沿用的静态兼容字段 `diagnostics_charged_as_training=true` 是阶段2 Budget类遗留说明，不用于计时或预算计算；本轮诊断实际明确计入prediction类别，以上三项已独立对账。原封存账本不改写，说明见 `audit/resource_accounting_note.json`。

本地交付快照已仅复算评分，科学数值在1e−12容差内一致；全部来源、诊断JSON和核文件的传输hash均匹配。完整稳定性在服务器已独立复算通过；本机默认Python没有NumPy，未重复本地稳定性计算，也未为此安装依赖。见 `audit/local_reproduction.json`。

独立状态/步数/Adam审计见 `audit/final_matrix_review.json`；独立指标/方向/随机参考/校准复核见 `audit/final_scoring_review.json`。无正式数值失败或缺失单元，完整源检查点、尾段checkpoint、恢复进度和核/预测均已保留。

## 解释与停止点

{' '.join(interpret)}

这些结果不支持把原阶段2的弱选优收益仅归因于联合1000步太短：在本次实际低学习率续训范围内，没有轨迹获得更好的regret5或gain5。n20选优对长度较稳健；n50对长度有负向敏感性。4000步仍有核和预测漂移，说明未达到所定实用稳定标准；Adam/日程核验已经通过，不能把未稳定解释为恢复失败，也不自动继续到8000步。建议停止单纯增加这条尾段的步数，下一轮若继续研究，先独立设计目标/容量/约束或更新预算匹配的对照；保持一次只改一个因素，并继续报告起始最佳与查询提升空间。区间依旧严重失准，不宜默认其可用于EI/KG探索。结论仅限本次固定尾段，不证明全局最优或真实搜索收益。阶段3、网络/目标修改和搜索回放均未执行，`next_stage_authorized=false`。

未读取B/test，未加载VAE前向，未训练GNN/VAE，未下载数据集、使用GPU或commit/push；原阶段2及现有脏工作区未改。复算和从包内源checkpoint重新续训见 `REPRODUCE.md`。

服务器目录：`/home/fangxy/D-VAE/artifacts/dkl-stage2-length-v1-20260922T092924Z/`。

只读状态核验用时 {read('audit/final_matrix_review.json')['audit_wall_seconds']:.3f} 秒，未计入续训预算。证据包生成及逐文件回读校验的墙钟另存外层 `PACKAGING.json`，不加入训练用时。
'''
(r/'README.md').write_text(readme)
lines=['# 未达到实用稳定标准的轨迹','', '以下按预注册最后三个100步窗口判定；ID来源于原60条固定工作清单。']
for n in [20,50]:
 for j in [2000,4000]:
  d=checkpoint(n,j);lines+=['',f'## n={n}，总联合步{j}', '',f"稳定 {d['practical_stability_counts']['stable']}/30；以下 {len(d['unstable_trajectory_ids'])} 条未稳定：",'']
  lines += ['- `'+x+'`' for x in d['unstable_trajectory_ids']] or ['无。']
(r/'UNSTABLE_TRAJECTORIES.md').write_text('\n'.join(lines)+'\n')
status={'execution_status':s['execution_status'],'completed_trajectories':seal['completed_trajectories'],'new_formal_updates':seal['new_formal_updates'],'new_endpoint_count':seal['new_endpoint_count'],'scientific_endpoint_count':s['scientific_endpoint_count'],'stability_counts':{str(n):{str(j):checkpoint(n,j)['practical_stability_counts'] for j in [2000,4000]} for n in [20,50]},'resources':seal['resources'],'next_stage_authorized':False,'B_test_read':False,'GNN_VAE_trained':False}
(r/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(status,ensure_ascii=False))
