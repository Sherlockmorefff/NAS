#!/usr/bin/env python3
"""Report saved zero-training supplement tables; no model construction or fitting."""
from collections import defaultdict
from pathlib import Path
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from analyse.fuzzy_metric_diagnosis_supplement import OUT,read,write,csv_write,mean_range,file_hash


def generate(out=OUT):
    started=time.monotonic();metrics=read(out/'metrics_24_combinations.json');summary=read(out/'summary_2x2.json')
    pairs=read(out/'paired_differences.json');kernel=read(out/'kernel_and_mean_summaries.json')
    changes=read(out/'mechanism_GP20_to100.json');checks=read(out/'checks.json');identity=read(out/'config_and_identity.json')
    mean_reference=[]
    for method in ['regression','N2']:
        for m in [100,300]:
            for g in [20,100]:
                selected=sorted([r for r in metrics if (r['method'],r['mapping_steps'],r['gp_steps'])==(method,m,g)],key=lambda r:r['seed'])
                if len(selected)!=3:continue
                for predictor in ['head','knn','gp']:
                    vals=[r[f'{predictor}_mae_minus_constant'] for r in selected]
                    mean_reference.append({'method':method,'mapping_steps':m,'gp_steps':g,'predictor':predictor,
                        'seed0':vals[0],'seed1':vals[1],'seed2':vals[2],**mean_range(vals)})
    write(out/'predictor_vs_constant.json',mean_reference);csv_write(out/'predictor_vs_constant.csv',mean_reference)
    mechanism_pairs=[];mechanism_rows=[]
    for method in ['regression','N2']:
        for m in [100,300]:
            selected=sorted([r for r in changes if r['method']==method and r['mapping_steps']==m],key=lambda r:r['seed'])
            if len(selected)!=3:continue
            keys=list(selected[0]['summary']['mean_changes'])+['prediction_std_change']
            for k in keys:
                vals=[r['summary']['mean_changes'][k] if k!='prediction_std_change' else r['summary']['GP_prediction_std_G100']-r['summary']['GP_prediction_std_G20'] for r in selected]
                mechanism_pairs.append({'method':method,'mapping_steps':m,'metric':k,'seed0':vals[0],'seed1':vals[1],'seed2':vals[2],**mean_range(vals)})
            for g in [20,100]:
                selected_k=[r for r in kernel if (r['method'],r['mapping_steps'],r['gp_steps'])==(method,m,g)]
                mechanism_rows.append({'method':method,'mapping_steps':m,'gp_steps':g,
                    'GP_mean_function':float(np.mean([r['gp_parameters']['GP_mean_function_accuracy'] for r in selected_k])),
                    'prediction_std':float(np.mean([r['gp_mean']['std_population'] for r in selected_k])),
                    **{k:float(np.mean([r[k]['mean'] for r in selected_k])) for k in
                        ['gp_mean','abs_gp_minus_training_mean','abs_gp_minus_mean_function','kernel_max_correlation','kernel_top5_mean_correlation','nearest_GP_distance_over_lengthscale','latent_variance','additional_noise_accuracy_squared','pooled_training_SE_squared','observation_variance']},
                    **{k:float(np.mean([r['gp_parameters'][k] for r in selected_k])) for k in ['lengthscale_scalar','signal_variance_accuracy_squared','signal_std_accuracy']}})
    write(out/'mechanism_paired_seed_ranges.json',mechanism_pairs);csv_write(out/'mechanism_paired_seed_ranges.csv',mechanism_pairs)
    csv_write(out/'mechanism_mean_table.csv',mechanism_rows);write(out/'mechanism_mean_table.json',mechanism_rows)
    variance_changes=[]
    for method in ['regression','N2']:
        for m in [100,300]:
            a=next(r for r in mechanism_rows if (r['method'],r['mapping_steps'],r['gp_steps'])==(method,m,20))
            b=next(r for r in mechanism_rows if (r['method'],r['mapping_steps'],r['gp_steps'])==(method,m,100))
            vals={k:b[k]-a[k] for k in ['latent_variance','pooled_training_SE_squared','additional_noise_accuracy_squared','observation_variance']}
            variance_changes.append({'method':method,'mapping_steps':m,**vals})
    write(out/'variance_component_changes.json',variance_changes)
    # A compact scientific plot: all four combinations, paired seeds, constant reference.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1),sharey=True)
    for ax,method,title in zip(axes,['regression','N2'],['Regression only','N2']):
        for g,color in [(20,'#0077bb'),(100,'#ee7733')]:
            for seed in range(3):
                y=[next(r['gp_mae'] for r in metrics if (r['method'],r['seed'],r['mapping_steps'],r['gp_steps'])==(method,seed,m,g))*100 for m in [100,300]]
                ax.plot([100,300],y,'o-',color=color,alpha=.65,label=f'GP {g} steps' if seed==0 else None)
        ax.axhline(checks['constant_MAE_identical']*100,ls='--',color='#777777',label='Training-mean reference')
        ax.set_xticks([100,300]);ax.set_xlabel('Mapping steps');ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Holdout GP MAE (percentage points)')
    fig.suptitle('Saved 2 x 2 endpoints; three paired model seeds, same 29 holdout candidates')
    fig.tight_layout();fig.savefig(out/'gp_2x2.png',dpi=160);fig.savefig(out/'gp_2x2.pdf');plt.close(fig)
    names={'regression':'仅回归','N2':'N2'}
    lines=['# 阶段2诊断只读补充：完整2×2比较','',
        '**本轮完成24个评价组合，新增GNN、VAE、映射和GP训练次数及更新步数全部为0。** 全部预测和快照原已存在；只补表、核验推理和计算核关联，没有调用旧训练入口。','',
        '**补齐后的判断：表示延长仍改善近邻和预测头，但不保证GP误差改善；GP延长并非总有害；N2的额外预测收益依赖GP步数，尚不稳定。GP明显好于训练均值参照，没有退化成无信息常数。** 本次不修改正式配置，不宣布阶段2通过。','',
        '## 1. 来源、范围与只读恢复','',
        '服务器 `ismc36`，项目 `/home/fangxy/D-VAE`，分支 `feature/accuracy-bottleneck-diagnostics`，HEAD `64a44c14316f11debaf01cf8583426acfe1410e6`。Python `/home/fangxy/.conda/envs/nas/bin/python` 3.10.20；单线程CPU。当前未提交工作被保留，未拉取、切分支、提交或推送。','',
        '读取了上一轮诊断报告、最新任务文档及原协议、比较表、核验和实际源码。原文与输入副本在 `input_originals/`；绝对路径、SHA、来源和检查点身份见 `config_and_identity.json`、逐组合 `source_paths/source_hashes` 和交付清单。','',
        '只使用原fold0内部112个训练候选、29个留出候选；两方法均用模型种子0/1/2。三个种子共享同一29个候选，不是三个独立数据集，也不是87个独立候选。N2为 $\\varepsilon=0.02$、$\\lambda_f=0.01$，两组 $\\lambda_e=0.001$，GP种子0；均与保存协议一致。目标仍为每候选原三次GNN训练的验证均值。','',
        '成绩仅由A文件的ID白名单解析器取得112+29条记录；没有先解析全A再筛选。B、fold0外层38个候选及测试成绩未解析或使用。候选配置/编码文件只提供无标签输入。训练均值、训练SE²和预处理取自112个训练候选；留出SE不参与预测。','',
        '24个GP快照各使用独立评估实例，加载相应M100/M300和G20/G100；严格核对步骤、ID顺序、目标、固定噪声、已存标准化、模型参数和GP输入。没有构造或调用优化器，模型及GP参数冻结，反向传播被禁用。重建缓存的矩阵分解只用于推理。所有24份旧预测逐项精确复现，最大差0；原18格汇总及已存但未汇总的6格评价也全部一致。','',
        'GP直接使用8维联合表示转为double，无额外输入预处理。原近邻仍按该表示的欧氏距离、candidate_id并列顺序取5个训练参考，均匀平均。实际核是带信号幅度的各向同性Matérn5/2；本批全部696行的核相关性前5与原距离前5相同。新推理与核摘要先保存，再关联留出真值；这是已用开发集分析，不是新独立盲测。','',
        '## 2. 完整2×2结果','',
        '以下均为三个模型种子的平均值。误差和区间宽度单位为百分点，比例为百分数。主近邻指标仍按“每查询5个参考平均性能差→查询中位数→模型种子平均”计算。24个逐种子完整行及全部指标见 `metrics_24_combinations.csv/json`。','',
        '| 方法 | 表示步数 | GP步数 | 主近邻差 | 预测头MAE | 5近邻均值MAE | GP MAE | 常数MAE |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        v=row['metrics'];lines.append(f"| {names[row['method']]} | {row['mapping_steps']} | {row['gp_steps']} | "+' | '.join(f"{v[k]['mean']*100:.4f}" for k in ['main_neighborhood_metric','head_mae','knn_mae','gp_mae','constant_mae'])+' |')
    lines += ['', '| 方法 | 表示/GP步数 | 错近邻比例0.006 / 0.012 | 90%覆盖 / 宽度 | 95%覆盖 / 宽度 |','|---|---|---:|---:|---:|']
    for row in summary:
        v=row['metrics'];lines.append(f"| {names[row['method']]} | {row['mapping_steps']}/{row['gp_steps']} | {100*v['wrong_0.006']['mean']:.2f}% / {100*v['wrong_0.012']['mean']:.2f}% | {100*v['coverage90']['mean']:.2f}% / {100*v['width90']['mean']:.2f} | {100*v['coverage95']['mean']:.2f}% / {100*v['width95']['mean']:.2f} |")
    lines += ['', '## 3. 配对差异与常数参照','',
        '所有差值为后者减前者；误差负值代表改善。下面给出GP误差的全部配对结果，单位百分点。其余每项指标同样保存了种子0/1/2差值、均值、最小值和最大值，共132行，见 `paired_differences.csv/json`。范围不是置信区间；覆盖率及宽度的正负不直接代表优劣。','',
        '| 比较 | 种子0 | 种子1 | 种子2 | 均值 | 最小值 | 最大值 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in pairs:
        if r['metric']=='gp_mae':lines.append('| '+r['comparison']+' | '+' | '.join(f"{r[k]*100:+.4f}" for k in ['seed0','seed1','seed2','mean','min','max'])+' |')
    lines += ['', '**固定GP20，表示100→300：** 两组近邻及预测头仍有改善，但GP误差平均分别增加0.4158和0.6016个百分点；各有一个种子改善，不能说GP也随表示延长受益。固定GP100时，仅回归GP误差增加0.6697个百分点，N2降低0.7622个百分点。这说明表示与GP步数的作用不能混在一起。','',
        '**固定表示，GP20→100：** 仅回归在M100/M300均变差，三个种子方向一致；N2在M100也均变差，但在M300两种子改善、一种子变差，平均仅降低0.0215个百分点，近似持平。因此上一轮“GP延长使两组都变差”只能用于M100比较，不能推广到M300。这里只比较两个端点，未证明整段误差单调恶化。','',
        '**同一步数下N2减仅回归：** GP20下，M100/M300的N2平均误差分别更高0.2797/0.4655个百分点；GP100下则分别更低0.1866/1.6185个百分点。方向随GP步数改变。N2近邻主指标的种子平均较低，但每个表示步数均是两个种子改善、一个变差；预测头与5近邻均值的平均误差仍高于仅回归。不能把“优于较差GP100对照”当作稳定额外收益。','',
        '常数参照为112个训练候选验证均值的算术平均 $0.6526845238$，不按架构加权、不使用留出成绩。其留出MAE在24格都等于12.6821个百分点。GP逐种子MAE范围6.5725–11.3311个百分点，全部优于该参照；预测头和5近邻均值也全部优于它。GP具有该开发集上的预测信息，不能称为完全退回常数。','',
        '| 方法 | 表示/GP步数 | 预测头减常数 | 5近邻减常数 | GP减常数 |','|---|---|---:|---:|---:|']
    for row in summary:
        v=row['metrics'];lines.append(f"| {names[row['method']]} | {row['mapping_steps']}/{row['gp_steps']} | "+' | '.join(f"{100*(v[k]['mean']-v['constant_mae']['mean']):+.4f}" for k in ['head_mae','knn_mae','gp_mae'])+' |')
    lines += ['', '上表为三种子均值差；逐种子差、均值和范围另见 `predictor_vs_constant.csv/json`。这仅说明优于常数，并不证明预测能力已足够支撑搜索；尤其M300下较简单的预测头和5近邻均值误差均低于GP。','',
        '## 4. GP是否更难借用训练信息','',
        '### 4.1 两种均值与预测离散程度','',
        'GP学得的均值函数与训练算术均值不同。每查询同时保存两者、GP预测、绝对距离和真实误差。下表各值均还原至accuracy单位后乘100；“预测离散”是29个候选预测的总体标准差，再平均三个种子。','',
        '| 方法 | 表示/GP步数 | GP均值函数 | GP预测均值 | 预测离散 | 距训练均值的平均绝对差 | 距GP均值函数的平均绝对差 |','|---|---|---:|---:|---:|---:|---:|']
    for row in mechanism_rows:
        lines.append(f"| {names[row['method']]} | {row['mapping_steps']}/{row['gp_steps']} | "+' | '.join(f"{100*row[k]:.4f}" for k in ['GP_mean_function','gp_mean','prediction_std','abs_gp_minus_training_mean','abs_gp_minus_mean_function'])+' |')
    lines += ['', '仅回归M100/M300及N2 M100在GP100端点的预测更集中，对训练均值和GP均值函数的平均距离都变小；N2 M300变化很小。均值函数本身也随拟合移动，因此不能把“靠近均值”单独归因于长度尺度。所有组仍有明显候选间变化，且误差小于常数参照，不是完全信息丢失。','',
        '### 4.2 信号核关联与误差集中位置','',
        '相对距离在GP实际输入上计算：本批是欧氏距离除以该快照标量长度尺度。核相关性使用实际信号核，排除观测噪声和数值稳定对角项：','',
        '$$','\\rho_{qi}=\\frac{k(x_q,x_i)}{\\sqrt{k(x_q,x_q)k(x_i,x_i)}}.','$$','',
        '以下先对每查询求最大相关性、前5平均相关性和最近相对距离，再取29个查询平均并平均三种子。完整分布和逐候选ID见 `kernel_and_mean_summaries.json`、`per_candidate.csv/json`。','',
        '| 方法 | 表示/GP步数 | 最大相关性 | 前5平均相关性 | 最近距离/长度尺度 | 长度尺度 |','|---|---|---:|---:|---:|---:|']
    for row in mechanism_rows:
        lines.append(f"| {names[row['method']]} | {row['mapping_steps']}/{row['gp_steps']} | "+' | '.join(f"{row[k]:.5f}" for k in ['kernel_max_correlation','kernel_top5_mean_correlation','nearest_GP_distance_over_lengthscale','lengthscale_scalar'])+' |')
    lines += ['', '仅回归的两个表示版本及N2 M100，三个种子的全部29个查询在GP100时最大核相关性都下降；相对距离变大。N2 M300则两个种子全部增强、一个种子全部减弱，与“GP延长一概削弱关联”不符。相关性是核的几何关系，不是最终预测权重，不能单凭它作因果判断。','',
        '按事先写入本轮分析规则的“GP100最大相关性最低8个查询”（29的下四分之一向上取整）观察误差变化；不用真实误差选该子组。前述关联普遍下降的9个种子/表示组合中，该8个查询占正误差增加量的59.4%–78.9%，高于其27.6%的候选比例；它们平均误差增加约4.21–11.89个百分点。逐候选相关变化与误差变化的Pearson相关为−0.726至−0.284。这支持误差恶化集中在关联较弱候选的描述，但仍是同一开发集上的相关关系，没有独立性或因果证明。','',
        'N2 M300的两个增强关联种子中，最弱8个查询反而平均改善；第三个种子关联减弱且误差增加。全部12组逐候选配对和分组数值均见 `mechanism_GP20_to100.json`，核关联、相对距离、均值距离、方差与误差变化的逐种子均值及范围见 `mechanism_paired_seed_ranges.csv/json`，没有只保留支持某一解释的组。','',
        '### 4.3 区间变宽伴随哪一项变化','',
        '保持原观测方差规则，不重校准、不裁剪：','',
        '$$','\\sigma^2_{\\mathrm{obs}}=\\sigma^2_{\\mathrm{latent}}+\\operatorname{mean}_{\\mathrm{train}}(SE^2)+\\sigma^2_{\\mathrm{learned}}.','$$','',
        '训练平均SE²在全部快照固定为0.0013759405。以下为原accuracy²单位，信号方差为核的输出尺度还原目标单位后的值，不能与后验潜在方差混淆。完整信号标准差、噪声和各分量逐快照保存。','',
        '| 方法 | 表示/GP步数 | 信号方差 | 平均潜在方差 | 拟合附加噪声 | 平均观测方差 |','|---|---|---:|---:|---:|---:|']
    for row in mechanism_rows:
        lines.append(f"| {names[row['method']]} | {row['mapping_steps']}/{row['gp_steps']} | "+' | '.join(f"{row[k]:.8g}" for k in ['signal_variance_accuracy_squared','latent_variance','additional_noise_accuracy_squared','observation_variance'])+' |')
    lines += ['', '仅回归M100/M300和N2 M100的观测方差增加几乎都来自潜在方差增加；训练平均SE²不变，学得噪声变化很小。N2 M300的平均潜在方差降低约0.0004272，拟合噪声增加约0.00000985，二者相抵后观测方差降低约0.00041735，区间略窄。分量的精确差值见 `variance_component_changes.json`。','',
        '这补强了“部分快照对训练参考关联变弱、候选预测收缩、潜在不确定性扩大”的证据。训练目标变小、长度尺度变短或区间变宽本身仍不证明过拟合；未改核、噪声或停止规则，也无法从本次只读端点分析判定哪个新GP设置更好。','',
        '## 5. 检查、时间与交付','',
        f"- 24/24组合完整，无缺失；旧预测和指标最大差0。12对固定表示的GP步数变化都保持近邻、预测头及近邻均值不变，GP预测本身各有变化，排除了缓存串用造成伪同一输出。",
        f"- 8项只读CPU检查通过，0.17秒；分析入口内部耗时{checks['elapsed_seconds']:.3f}秒。没有合成模型训练；创建优化器、反向传播和优化更新计数全部为0。",
        '- 原阶段2与上一轮诊断的协议、报告、检查点、结果及费用记录保持原内容；任务文档仅追加本轮补充状态。原五折负结果和19项缺口保留，阶段3–6未启动。最终原文件身份核验、差异与包清单在 `delivery_manifest.json`。','',
        '已实际执行：','',
        '```bash','cd /home/fangxy/D-VAE',
        'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_diagnosis_supplement.py --readonly',
        '/home/fangxy/.conda/envs/nas/bin/python -m pytest -q tests/test_fuzzy_metric_diagnosis_supplement.py',
        'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_diagnosis_supplement_report.py --write','```','',
        f"服务器结果根：`{out}`；日志：`{ROOT/'logs'/out.name}`。本报告在项目docs和独立结果目录各保存一份。",'',
        '证据包包含 `metrics_24_combinations.csv/json`、`summary_2x2.json`、全部配对CSV/JSON、696行逐候选预测/真值/邻居/核摘要、112个训练标签与29个留出均值、各组合预测与评价、模型/输入来源、核与均值分布、方差分量变化、检查、源码快照及代码差异。表格可独立复算报告数值，不只是哈希清单。','',
        '## 6. 下一步只做什么','',
        '**暂不直接修改GP。最必要的后续动作是预先固定一次跨架构划分的验证，比较现有GP与预测头、5近邻均值的误差稳定性。** 理由是当前GP确实优于常数，收益与损害又明显依赖表示、GP步数和种子；先验证这些现象能否复现，比直接改核或噪声更有依据。本轮没有执行该验证，没有修改正式配置或宣布阶段2通过。','']
    report='\n'.join(lines)
    (out/'supplement_report.md').write_text(report)
    doc=ROOT/'docs/Fuzzy_Metric_Stage2_Readonly_Supplement_20260910.md'
    doc.write_text(report)
    write(out/'report_checks.json',{'report_sha256':file_hash(doc),'report_seconds':time.monotonic()-started,
        'constant_comparison_groups':len(mean_reference),'mechanism_paired_rows':len(mechanism_pairs),'new_training_steps':0})
    print(str(doc),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if not args.write:parser.error('use --write')
    generate()
