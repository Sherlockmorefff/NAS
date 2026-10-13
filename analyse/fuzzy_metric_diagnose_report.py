#!/usr/bin/env python3
"""Read-only report and artifact checks for the fixed diagnosis; never fits."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from analyse.fuzzy_metric_diagnose import read,write,load,file_hash,tensor_hash,tail


def mean_range(values):
    return {'mean':float(np.mean(values)),'min':float(min(values)),'max':float(max(values))}


def verify(out):
    protocol=read(out/'protocol.json');old=ROOT/protocol['config']['old_output']
    changed=[p for p,h in read(out/'old_results_hashes.json').items() if not (old/p).exists() or file_hash(old/p)!=h]
    if changed:raise RuntimeError(f'original results changed: {changed}')
    done=[];baseline=[];fixed=[];noise=[];clock=[]
    for p in sorted((out/'tasks').glob('*')):
        row=read(p/'completed.json')
        for name,h in row['artifacts'].items():
            if file_hash(p/name)!=h:raise RuntimeError('diagnosis artifact changed')
        done.append(p.name)
        if p.name.endswith('-gp'):
            access=read(p/'evaluation_access.json')
            for s in (20,100):
                pred=p/f'G{s}_predictions_before_reveal.json'
                if file_hash(pred)!=access['prediction_hashes'][f'G{s}']:raise RuntimeError('prediction hash before reveal differs')
                if pred.stat().st_mtime>access['time']:raise RuntimeError('predictions follow query access')
                bundle=read(pred);assert bundle['query_ids']==protocol['inner']['query_ids']
                assert bundle['train_ids']==protocol['inner']['train_ids']
                assert not set(bundle['query_ids']) & set(protocol['outer_holdout_ids_excluded_from_new_comparisons'])
                for r in bundle['predictions']:
                    assert r['observation_variance']==r['latent_variance']+r['pooled_training_SE_squared']+r['additional_noise_accuracy_squared']
                noise.append({'trajectory':p.name,'GP_step':s,'components':bundle['gp_parameters']})
                clock.append({'trajectory':p.name,'snapshot':s,'prediction_mtime':pred.stat().st_mtime,'query_access_time':access['time']})
            a,b=[read(p/f'G{s}_predictions_before_reveal.json') for s in (20,100)]
            assert a['standardization']==b['standardization']
            for x,y in zip(a['predictions'],b['predictions']):
                for k in ['head_mean','knn_mean','neighbor_ids','neighbor_distances']:
                    assert x[k]==y[k],(p,k)
            fixed.append(p.name)
            if (p/'baseline_reuse_check.json').exists():
                row=read(p/'baseline_reuse_check.json')
                assert row['gp_state_bitwise_equal'] and row['neighbors_equal'] and row['loss_prefix_equal']
                assert row['max_prediction_difference']<=1e-12
                baseline.append(row)
        else:
            a,b=[load(p/f'M{s}.pt') for s in (100,300)]
            assert a['standardization']==b['standardization']
            assert {float(v['step']) for v in a['optimizer']['state'].values()}=={100.}
            assert {float(v['step']) for v in b['optimizer']['state'].values()}=={300.}
    inv=read(out/'invocation02.json');assert inv['before']==inv['after']
    cost=read(out/'costs.json')
    assert cost['mapping']['actual_completed_optimizer_steps']==1200
    assert cost['gp']['actual_completed_optimizer_steps']==1200
    assert cost['mapping']['attempt_starts']==6 and cost['gp']['attempt_starts']==12
    assert len(done)==18 and len(baseline)==6
    assert read(out/'optional_trigger.json')['execute'] is False
    first=out/'tasks/regression-s0-mapping/attempt01/started.json'
    assert read(out/'saved_diagnostics/completed.json')['time']<read(first)['time']
    # The first mapping and corresponding GPs were reused, not rerun as probes.
    first_end=max(read(out/'tasks'/t/'completed.json')['time'] for t in
        ['regression-s0-mapping','regression-s0-M100-gp','regression-s0-M300-gp'])
    other_starts=[read(p)['time'] for p in (out/'tasks').glob('*/attempt*/started.json') if not p.parent.parent.name.startswith('regression-s0-')]
    assert first_end<min(other_starts)
    checks={'old_result_files_unchanged':len(read(out/'old_results_hashes.json')),'completed_trajectories':len(done),
        'old_M100_GP20_baselines_bitwise_reproduced':len(baseline),'GP_extensions_preserve_heads_and_neighbors':len(fixed),
        'inner_ids_only_and_all_scalers_equal':True,'prediction_before_query_access':clock,
        'new_GP_initializations':{p.parent.name:read(p) for p in (out/'tasks').glob('*/initial_gp.json')},
        'noise_components':noise,'resume_completed_no_new_attempts_or_steps':True,
        'first_real_task_and_GPs_finished_before_other_tasks':True,'saved_analysis_precedes_new_fits':True,
        'optional_weight_not_triggered':True,'old_missing19_unchanged':True,'new_GNN_trainings':0}
    write(out/'verification.json',checks)
    return checks


def figures(out,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    methods=['regression','N2'];colors={'regression':'#777777','N2':'#0077bb'}
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for method in methods:
        for seed in (0,1,2):
            hs=read(out/'tasks'/f'{method}-s{seed}-mapping'/'trajectory.json')['history']
            for ax,k in zip(axes[0],['regression','neighborhood','total']):
                ax.plot([r['step'] for r in hs],[r[k] for r in hs],color=colors[method],alpha=.6)
                ax.axvline(100,color='black',ls=':',lw=.6);ax.set_title(k);ax.set_xlabel('mapping step')
            for ms,style in [(100,'--'),(300,'-')]:
                hs=read(out/'tasks'/f'{method}-s{seed}-M{ms}-gp'/'trajectory.json')['history']
                for ax,k in zip(axes[1],['negative_marginal_log_likelihood','lengthscale','outputscale_normalized']):
                    vals=[r[k] if k in r else r['parameters_after_update'][k] for r in hs]
                    ax.plot([r['step'] for r in hs],vals,color=colors[method],ls=style,alpha=.6)
                    ax.axvline(20,color='black',ls=':',lw=.6);ax.set_title('GP '+k);ax.set_xlabel('GP step')
    fig.suptitle('Fixed fold0 inner split; gray=regression, blue=N2; GP dashed=M100, solid=M300')
    fig.tight_layout();fig.savefig(folder/'new_trajectories.png',dpi=150);fig.savefig(folder/'new_trajectories.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,k,title in zip(axes,['head_mae','knn_mae','gp_mae'],['Prediction head','Five-neighbor mean','GP']):
        for method in methods:
            rows=comparison['methods'][method]['per_seed']
            for seed in (0,1,2):
                ys=[next(r[k] for r in rows if r['seed']==seed and r['condition']==c)*100 for c in ['baseline','long_gp','long_mapping']]
                ax.plot(range(3),ys,marker='o',color=colors[method],alpha=.6)
        ax.set_xticks(range(3),['M100/G20','M100/G100','M300/G100'],rotation=15)
        ax.set_title(title);ax.set_ylabel('Holdout MAE (percentage points)');ax.grid(alpha=.2)
    fig.suptitle('Paired model seeds; gray=regression, blue=N2; 29 fixed holdout candidates')
    fig.tight_layout();fig.savefig(folder/'three_predictor_errors.png',dpi=150);fig.savefig(folder/'three_predictor_errors.pdf');plt.close(fig)


def report(out):
    start=time.monotonic();torch.set_num_threads(1)
    comparison=read(out/'comparison.json');cost=read(out/'costs.json');checks=verify(out)
    grouped=defaultdict(list)
    for row in comparison['paired_per_seed']:
        key=(row['comparison'],row.get('method',''),row.get('condition',''))
        grouped[key].append(row)
    paired=[]
    for key,rows in grouped.items():
        keys=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k!='seed']
        paired.append({'comparison':key[0],'method':key[1],'condition':key[2],
            'three_seed_difference_summary':{k:mean_range([r[k] for r in rows]) for k in keys}})
    write(out/'paired_mean_range.json',paired)
    newtails=[]
    for p in (out/'tasks').glob('*/trajectory.json'):
        row=read(p)
        if p.parent.name.endswith('-mapping'):
            newtails.append({'task':p.parent.name,'last20':tail(row['history'],['regression','neighborhood','scale','total'],20)})
        else:newtails.append({'task':p.parent.name,'last20':tail(row['history'],['negative_marginal_log_likelihood'],20)})
    write(out/'new_training_tails.json',newtails)
    figures(out,comparison)
    env=read(out/'environment.json');workspace=read(out/'workspace_before.json');proto=read(out/'protocol.json')
    doc=out/'input_originals/Codex_Server_Stage2_Diagnosis_Prompt.md'
    s=comparison['methods'];old=read(out/'saved_diagnostics/training_tails_and_curves.json')
    lines=['# 阶段 2 后独立诊断：固定步数比较与旧结果分析','',
        '日期：2026-09-10。状态：**本轮授权诊断完成；没有形成正式通过结论或最终确认配置。** 原阶段2五折负结果、640次映射账本和缺19项的状态保持原样。阶段3–6未启动。','',
        '本轮最明确的结果是：增加映射训练改善了近邻和预测头；只把 GP 从20步加到100步却使留出预测变差。N1/N2 在旧数据上的实际约束差很小。不能把所有问题解释为“训练步数不足”，也不能据此宣布 N2 有稳定收益。','',
        '## 1. 实际位置、版本与数据边界','',
        f"- 主机 `{workspace['host']}`；项目/Git根 `{workspace['git_root']}`；分支 `{workspace['branch']}`；完整 HEAD `{workspace['head']}`。",
        f"- Python `{env['executable']}`，{env['python']}；torch {env['packages']['torch']}、GPyTorch {env['packages']['gpytorch']}、BoTorch {env['packages']['botorch']}。CUDA可用，两张RTX4500 Ada；本轮实际为串行CPU、单线程。",
        f"- 用户原文 `/Users/sherlockmore/Downloads/Codex_Server_Stage2_Diagnosis_Prompt.md`，SHA-256 `{file_hash(doc)}`；服务器原文副本在本结果目录 `input_originals/`。",
        '- 完整阅读方案、最新任务文档、阶段0/1报告和原阶段2报告，版本副本及SHA见 `input_originals/`、`workspace_before.json`。既有未提交修改作为起点，没有拉取、切分支、提交或推送。','',
        '原A为179候选/96架构，B为157候选/96架构。旧结果诊断只使用A的既有评价记录；新比较严格使用原fold0内部112个训练候选、29个留出候选，外层38个留出候选没有进入新比较。只读原 `A_validation.json` 白名单，没有打开全局成绩库或导出/解析B成绩。目标始终为相同配方下训练种子0/1/2的验证准确率均值，没有测试成绩。','',
        '新比较使用原fold0已选N2：$\\varepsilon=0.02$、$\\lambda_f=0.01$；模型种子0/1/2。$\\lambda_e=0.001$、映射Adam学习率0.001、GP Adam学习率0.05、GP种子0均保持一致；无学习率调度、无早停。全部输入与目标标准化只拟合112个训练候选。原fold0-refit检查点没有用来续训。','',
        '6个原内部检查点均合法：训练/留出ID、标准化、初始权重、第100步模型和Adam状态一致，重算表示逐位相同。因此只续训100→300，没有重建0→100。新第100步GP重新初始化，20步状态与原状态逐位一致，预测和近邻也一致；这6个旧成功模型没有再次收费。证据为 `protocol.json/reuse` 和6份 `baseline_reuse_check.json`。','',
        '## 2. 先分析旧结果：已经保存的证据','',
        '在任何新优化开始前，已保存557个成功任务的逐步损失及尾段变化、69个选中模型的训练/留出诊断、45组固定折/固定阈值的三档权重比较、36组同一表示上的N1/N2对比与36次无更新梯度检查。目录为 `saved_diagnostics/`，完成时间早于首个新真实任务。','',
        '### 2.1 训练尾段与训练/留出差距','',
        '下表只描述原60个外层重训的尾段，按方法平均15个模型；不是重新计算的五折主指标。映射列是第81–85与96–100步的平均标准化回归损失；GP列是第11–15与16–20步的平均负对数似然。完整20步尾段、拟合斜率、回归/邻域/大小/总损失均保存在JSON及图中。','',
        '| 方法 | 映射尾段前5 → 后5 | GP尾段前5 → 后5 |','|---|---:|---:|']
    for m in ['regression','N0','N1','N2']:
        rows=[r for r in old if r['stage']=='refit' and r['family']!='final_A' and r['setting']['method']==m]
        a=[float(np.mean([r['mapping_last20']['regression'][k] for r in rows])) for k in ['first5_mean','last5_mean']]
        b=[float(np.mean([r['GP_last10']['negative_marginal_log_likelihood'][k] for r in rows])) for k in ['first5_mean','last5_mean']]
        lines.append(f'| {m} | {a[0]:.4f} → {a[1]:.4f} | {b[0]:.4f} → {b[1]:.4f} |')
    lines += ['', '二者都还在下降，故“有限梯度”不代表已经收敛。原日志只保存GP第20步检查点，没有中途长度尺度、输出尺度和噪声参数；这些缺失没有伪造或通过重跑旧五折补齐。原第20步实际参数见 `gp_parameters_at20.json`；新授权轨迹记录了0–100步参数。','',
        '训练集检索排除了自身，但允许同架构其他HP作为参考；留出集架构完全未见。按原外层模型平均，regression训练/留出预测头误差约6.716/7.398个百分点，N2为7.522/8.114；N2训练近邻差本身也未优于regression（6.558对5.944个百分点）。所以旧负结果不能简单归结为“训练很好、只在未见架构失败”。逐模型差距和同架构参考比例见 `train_holdout_spaces.json`。','',
        '### 2.2 权重、截断与局部改善','',
        '各权重比较都锁定同一折、方法、阈值和种子，不混合不同参数。以原fold0的N2、阈值0.02为例：','',
        '| 权重 | 训练回归损失 | 训练邻域损失 | 留出近邻差，百分点 | GP误差，百分点 |','|---:|---:|---:|---:|---:|']
    group=next(g for g in read(out/'saved_diagnostics/inner_grid_fixed_fold_epsilon.json') if g['fold']=='fold0' and g['method']=='N2' and g['epsilon']==.02)
    for weight in ['0.01','0.1','1.0']:
        v=group['by_weight'][weight]
        lines.append(f"| {weight} | {v['final_regression_loss']['mean']:.5f} | {v['final_neighborhood_loss']['mean']:.5f} | {100*v['main_neighborhood_metric']['mean']:.4f} | {100*v['gp_mae']['mean']:.4f} |")
    lines += ['', '增大权重降低了它直接优化的邻域损失，同时损害回归拟合，留出收益并不随权重单调增加。45组完整表及逐种子差值均保留，不能从这一示例推广到所有配置。','',
        'N1/N2在相同训练标签、阈值、噪声权重和同一保存表示上比较：截断删除的加权目标相似度总量约占N1的0.42%–0.55%。逐点目标差、删除质量、同坐标邻域损失及加权总损失差见 `N1_N2_same_embedding.json`。这比“非零目标对数量”更直接地说明截断为何几乎不改变原结果；它不证明两方法数学等价。','',
        'fold0原N2的共享映射参数上，已乘权重的邻域梯度范数约0.0127–0.0141，回归梯度为0.118–0.143；二者方向余弦约−0.124至−0.032。存在轻度方向冲突，但该单点局部证据不说明邻域项压倒回归，更不能证明全局因果。检查没有优化器更新。共同缩放不改变近邻排序，因此没有把表示范数当作根因。','',
        '高分前10%包含18个候选、15个架构。在N1/N2中，各种子有11/12/12个高分查询改善，但最大的5个查询占该子组正改善量约71.6%/68.9%/82.2%；仍有7/6/6个高分查询变差。其余90%平均近邻差反而变差。逐候选、架构和种子记录见 `improvement_concentration.json`。局部收益是真实描述，但不能翻转原五折筛选结论。','',
        '## 3. 固定步数比较：只用原fold0内部划分','',
        '每格是3个模型种子、同一29个留出候选。误差及区间宽度以下用百分点，近邻指标保持先对每查询5个参考求平均差、再取查询中位数、最后平均种子。训练回归损失是训练集标准化目标上的均方误差。预测头已还原原始单位；5近邻预测是已知训练成绩的均值。','',
        '| 方法 | 映射/GP步数 | 训练回归损失 | 近邻差 | 预测头误差 | 5近邻均值误差 | GP误差 |','|---|---|---:|---:|---:|---:|---:|']
    for m in ['regression','N2']:
        for condition,label in [('baseline','100/20'),('long_gp','100/100'),('long_mapping','300/100')]:
            row=s[m]['summary'][condition]
            lines.append(f"| {m} | {label} | {row['training_regression_loss']['mean']:.5f} | "+' | '.join(f"{row[k]['mean']*100:.4f}" for k in ['main_neighborhood_metric','head_mae','knn_mae','gp_mae'])+' |')
    lines += ['', '| 方法 | 映射/GP步数 | 错近邻比例，0.006 / 0.012 | 90%覆盖 / 宽度 | 95%覆盖 / 宽度 |','|---|---|---:|---:|---:|']
    for m in ['regression','N2']:
        for c,label in [('baseline','100/20'),('long_gp','100/100'),('long_mapping','300/100')]:
            r=s[m]['summary'][c]
            lines.append(f"| {m} | {label} | {100*r['wrong_0.006']['mean']:.2f}% / {100*r['wrong_0.012']['mean']:.2f}% | {100*r['coverage90']['mean']:.2f}% / {100*r['width90']['mean']:.2f} | {100*r['coverage95']['mean']:.2f}% / {100*r['width95']['mean']:.2f} |")
    lines += ['', '### 配对差值：平均值 [三种子最小值，最大值]','',
        '误差差值单位为百分点，负数为改善；同一比较内按模型种子配对，范围不是置信区间。完整全部指标见 `paired_mean_range.json`。','',
        '| 方法 / 比较 | 近邻差变化 | 预测头误差变化 | 5近邻误差变化 | GP误差变化 |','|---|---:|---:|---:|---:|']
    for row in paired:
        if row['method'] not in ['regression','N2']:continue
        vals=row['three_seed_difference_summary']
        cells=[]
        for k in ['main_neighborhood_metric','head_mae','knn_mae','gp_mae']:
            v=vals[k];cells.append(f"{v['mean']*100:+.4f} [{v['min']*100:+.4f}, {v['max']*100:+.4f}]")
        lines.append('| '+row['method']+' / '+row['comparison']+' | '+' | '.join(cells)+' |')
    lines += ['', '仅延长GP时，12组快照中的预测头、近邻ID、距离和近邻均值均完全不变；两方法的GP误差却在每个种子上变大。训练GP目标继续下降，但留出误差和区间宽度同时增加，这提示训练目标与本留出集预测质量不一致。这里没有将其直接命名为已经证明的过拟合，也不能仅凭一个划分决定新的核或噪声设置。','',
        '增加映射到300步时，两个方法的训练回归、预测头和近邻主指标均改善；但regression的GP误差继续增大，N2的GP误差相对100/100有所下降，仍差于其原100/20。当前300/100下N2的近邻与GP优于regression，预测头和5近邻均值误差仍劣于regression。这是局部诊断线索，不能替代原五折证据或证明截断的额外价值。','',
        'GP长度尺度、输出尺度、附加噪声和训练平均SE²在每步保存。以M100三种子均值为例：regression长度尺度约0.308→0.116、输出尺度1.307→1.928；N2长度尺度0.342→0.202、输出尺度1.219→1.479。GP目标变小的同时模型尺度明显改变，观测区间变宽；这些参数是训练后的实际值，不是默认配置值。完整每步曲线及第20/100步参数见轨迹与 `verification.json/noise_components`。','',
        '观测方差始终为 $\\sigma^2_{\\mathrm{latent}}+\\operatorname{mean}_{\\mathrm{train}}(SE^2)+\\sigma^2_{\\mathrm{learned}}$，没有重复加噪声或再除以3。该内部分割训练平均SE²约0.00137594，来自训练集；预测不使用留出SE或重复成绩。区间针对有限三种子均值观测，不是无限训练期望。','',
        '### 较小权重：按预定条件跳过','',
        '300步后，N2训练回归损失0.142688，小于regression的0.146893；N2的GP误差0.082479，也小于regression的0.098664。预定规则要求“N2训练回归严格更差，且GP误差不更低”，两个条件均不成立。因此没有训练0.001权重，没有尝试第三档、改变阈值或另选划分。触发数值及规则在 `optional_trigger.json`。','',
        '## 4. 实际费用、故障与验证','',
        '| 项目 | 本轮真实历史数据诊断 |','|---|---:|',
        f"| 唯一映射轨迹 / 实际启动 / 成功 | 6 / 6 / 6 |",
        '| 唯一GP轨迹 / 实际启动 / 成功 | 12 / 12 / 12 |',
        '| 映射实际新增优化步数 | 1200（6 × 200） |',
        '| GP实际优化步数 | 1200（12 × 100） |',
        '| 真实任务失败 / 重试 | 0 / 0 |',
        '| 新增GNN训练 / VAE训练 / 最终测试 | 0 / 0 / 0 |',
        f"| 映射耗时合计 / GP耗时合计 | {cost['mapping']['seconds_successful_trajectory_sum']:.3f}秒 / {cost['gp']['seconds_successful_trajectory_sum']:.3f}秒 |",
        f"| 首次比较入口内部耗时（含保存和汇总） | {read(out/'invocation01.json')['elapsed_seconds']:.3f}秒 |",'',
        '本轮与原阶段2合计：映射启动646次（原640含83次失败＋本轮6），GP启动569次（原557＋本轮12）。旧原账本文件没有变化，19项仍未补跑。这些次数与合成工程检查分列，历史成绩读取不算新增GNN训练。','',
        '工程检查共9项有效通过：首轮7项通过、2项失败，修复后单独重跑2项均通过（2.88秒和2.79秒）。修复一是续训加载Adam状态时先复制，避免内存别名影响同一旧状态的再次加载；二是均值断言采用1e−15浮点容差。没有改旧核心源码或科学损失。两轮合成检查实际新增映射12步、GP14步，与真实诊断1200/1200分别计数。','',
        '旧结果只读汇总还有1次保存失败：NumPy整数计数不能直接写JSON；转为普通整数后复用已落盘的6张诊断表完成汇总，没有重复模型拟合或梯度检查。36次无更新梯度检查已计数，首次梯度专用耗时因该保存失败未记录，明确为缺失，不填0；修复后只读收尾用时2.182秒。原失败日志和修复前源码/协议均保留在 `input_originals/`、`execution_amendment_report_serialization.json`。','',
        f"真实产物核验通过：{checks['old_result_files_unchanged']}个原结果文件保持原字节身份；6份M100/G20原状态逐位复现；12条GP延长轨迹的头与近邻不变；预测全部早于新评价访问；完成后恢复的尝试次数、优化步数与费用完全不变。首个真实映射及对应两个GP完成后才启动其他任务。语法检查与 `git diff --check` 已执行，最终文档差异另记交付清单。",'',
        '## 5. 运行命令与交付位置','',
        f"结果根：`{out}`。日志根：`{ROOT/'logs'/out.name}`。",'',
        '```bash','cd /home/fangxy/D-VAE',
        'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_diagnose.py --audit',
        'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_diagnose.py --run',
        '/home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_diagnose_report.py --write','```','',
        '前两条及完成后再次运行已验证；报告入口只读保存结果，不调用拟合。检查的准确节点和日志在 `engineering_checks.json`。新增入口/配置/检查保存在既有 analyse、configs、tests 结构。', '',
        '- `protocol.json`：独立设置、旧内部划分、复用检查点、输入与源码身份；修复前协议单独保留。',
        '- `saved_diagnostics/`：旧损失曲线、末段变化、训练/留出差距、同折同阈值权重表、N1/N2同表示差异、局部改善分布及无更新梯度。',
        '- `tasks/<method>-s<seed>-mapping/`：M100/M300、Adam、逐步损失、来源、恢复进度；`*-M<step>-gp/`：G20/G100、三类预测、评价、每步参数及噪声。',
        '- `comparison.json`、`paired_mean_range.json`、`optional_trigger.json`：逐种子配对、均值范围及停止规则。',
        '- `attempts.jsonl`、`costs.json`、`invocation01/02.json`、`verification.json`：真实费用、恢复和边界验证。',
        '- `figures/`：真实训练曲线与三类预测误差图（PNG/PDF）；本报告及更新任务文档位于项目 docs。','',
        '## 6. 结论与唯一下一步','',
        '支持把“映射100步可能不足”和“GP适配仍有问题”作为后续研究问题；不支持直接把100步GP设为更优默认，不支持继续降低权重，也没有依据改变核心公式。数学实现未发现实质错误，旧结果的主要现象可由实际约束差很小、回归与邻域的权衡、以及GP训练目标与留出预测不一致解释一部分；各自因果贡献尚不清楚。','',
        '**下一步最必要的行动：另立一个小规模、按架构留出的GP适配验证方案，固定表示，并以现有预测头和5近邻均值作为误差参照。** 先确认预测与区间可靠性，再讨论搜索。本轮不启动该实验。单一内部划分的3个模型种子不能替代原五折评价，原负结果和缺19项均保留，阶段3–6继续未启动。','']
    text='\n'.join(lines)
    (out/'diagnosis_report.md').write_text(text)
    report_path=ROOT/'docs/Fuzzy_Metric_Stage2_Diagnosis_20260910.md'
    if report_path.exists() and report_path.read_text()!=text:raise RuntimeError('existing diagnosis report differs; preserve it before replacement')
    report_path.write_text(text)
    write(out/'report_verification.json',{'report_hash':file_hash(report_path),'read_only_report_seconds':time.monotonic()-start,
        'paired_groups':len(paired),'scientific_fits_started':0,'checks_passed':True})
    print(str(report_path),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args()
    if not a.write:p.error('use --write to save the read-only report')
    report(ROOT/'results/posthoc/fuzzy-metric-stage2-diagnosis-20260910-v1')
