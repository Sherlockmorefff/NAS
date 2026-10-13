最后更新日期：2026-08-13
当前状态：[已验证] 协议、预算与现有 artifact 已核对；方法间因 low-fidelity 成本与 final-evaluation 协议差异需分层比较。

# 初始化方法实验

## 目的与命名

[已验证] 本页比较的是贝叶斯优化开始前如何覆盖联合搜索空间，以及初始化如何影响早期效率、surrogate 质量和最终结果。联合空间术语以 [项目总览](../project_overview.md#联合搜索空间与术语边界) 为准。

| 论文简称 | artifact/config 名称 | 初始化机制 |
| --- | --- | --- |
| S0 | `global_schur` | 在 768 个 LHS 候选的完整 16 维坐标上做全局 Schur 选择 |
| G100 | `gmm_exp100` | K=4 普通对角 GMM 分簇，簇内 TED shortlist，20-epoch low-fidelity 打分后提升 100 个候选 |
| G150 | `gmm_exp150` | 与 G100 相同，但提升 150 个候选 |
| K6-LF | `fixedk6_gmm_schur_lowfid` | 在 12 维架构坐标上做固定 K=6 GMM，使用共享 RBF-Schur 选择与 20-epoch low-fidelity，提升 100 个候选 |

证据：[正式方法配置](../../configs/cross_dataset_methods.json)、[GMM–TED 协议](../../docs/wgmm_clustered_ted.md)、[K6 Schur 实现](../../initialization_gmm_schur.py)。

## 机制细节

### S0：global Schur

[已验证] S0 从固定候选池选择 50 个 full-fidelity seed，随后执行 250 次 qLogEI。它没有 low-fidelity 阶段；其 300 次 full evaluation 既是训练预算，也是直接成本轴。

### G100 / G150：clustered TED + low fidelity

[已验证] 名称中的 `wgmm_ted_lowfid` 是历史实现名；正式配置的 `wgmm_source=gmm_fit_pool` 明确表示普通对角 GMM、候选等权，而不是加权 GMM。K 固定为 4，hard assignment，quota 为 0.5 equal / 0.5 mass 的 hybrid。TED 在 Exact-GP 固定域归一化后的完整 `z_search` 上使用 RBF 核，lengthscale 4.0、regularization 0.1，每簇 shortlist 上限 100。

[已验证] shortlist 候选以 20 epochs、无 early stopping 的 low-fidelity 评估；提升分数权重为 low-fidelity 0.50、GP mean 0.25、GP std 0.25。每簇 shortlist 上限是 100，但实际候选总数随 search seed 的簇容量变化，最多为 400；这些评估不计入 300 次 full 预算。

### K6-LF：architecture clustering + Schur + low fidelity

[已验证] K6-LF 在 12 维 `z_arch` 上拟合 K=6 普通对角 GMM，使用 5 次 restart；配额同样采用 hybrid。它在完整 `z_search` 上使用共享 RBF 核做簇内 Schur 选择，筛出 200 个 low-fidelity 候选并提升 100 个。200 × 20 / 150 = 26.67 次 equivalent-full evaluations。

## 预算与公平性

| 方法 | full seed | LF 候选 × epochs | promoted full | qLogEI full | full 总数 | LF 等价 full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 | 50 | 0 | 0 | 250 | 300 | 0.00 |
| G100 | 50 | 每簇至多 100 × 20 | 100 | 150 | 300 | 按 run 的实际 LF epochs 计算 |
| G150 | 50 | 每簇至多 100 × 20 | 150 | 100 | 300 | 按 run 的实际 LF epochs 计算 |
| K6-LF | 50 | 200 × 20 | 100 | 150 | 300 | 26.67 |

[已验证] Cora seeds 0–4 的 G100/G150 实际 LF 候选数分别为 207、272、264、356、238，对应 27.60–47.47 equivalent-full，均值 35.65。历史跨数据集 seeds 5–9 分别为 360、311、338、259、400，对应 34.53–53.33，均值 44.48；同一 search seed 的候选池使该序列跨四个数据集一致。

[已验证] 所有方法的 full-evaluation 数相同并不意味着总计算成本相同。best-so-far 图必须同时给出 full-evaluation 横轴和 `full + low_fidelity_actual_epochs / 150` 的 equivalent-full 横轴，并另列 wall-clock；不得只在 300-full 轴上声称同成本。

## 已完成矩阵

| 证据族 | 数据集 | 方法 | search seeds | 每 seed full 历史 | final evaluation |
| --- | --- | --- | --- | ---: | --- |
| Cora seed-fair | Cora | S0/G100/G150 | 0–4 | 300 | Top-10，30 replicates，300 epochs，patience 80，15/15 complete |
| K6 独立协议 | Cora | K6-LF | 0–4 | 300 | Top-7，30 replicates，150 epochs，patience 40，5/5 complete |
| 历史跨数据集 | CiteSeer/PubMed/DBLP/Flickr | S0/G100/G150 | 5–9 | 300 | 未找到匹配 artifact |

原始证据：[Cora 搜索归档](../../legacy_artifacts/pre_20260813/results/)、[Cora Top-10 final evaluation](../../legacy_artifacts/pre_20260813/results/final_eval_topk10_r30_seedfair_base20260724/)、[K6 搜索](../../legacy_artifacts/pre_20260813/results/fixedk6_gmm_schur_lowfid_pilot_20260728_024642/)、[K6 Top-7 final evaluation](../../legacy_artifacts/pre_20260813/results/fixedk6_top7_searchseeds0_4_final30_20260728/)、[跨数据集矩阵](../../legacy_artifacts/pre_20260813/results/formal_search_20260731_8831cd0a5337/)。汇总见 [结果表](../results/tables.md)。

## 混杂因素与判定规则

- [已验证] G100/G150 同时改变 promoted 数量与剩余 qLogEI 次数，不能把差异只归因于“更好的聚类”。
- [已验证] K6-LF 与 G100 不只差 K：聚类空间、核选择方法、shortlist 数和 final-evaluation 训练协议均不同，因此 K6 表独立呈现。
- [已验证] 搜索比较以 5 个独立 search seeds 为单位，样本标准差使用 `ddof=1`，95% CI 使用 df=4 的 t 区间。
- [已验证] final evaluation 先在每个 search seed 的候选中按 final validation 选择，再跨 5 个 search seeds 汇总 test；30 个 replicate 不作为 30 个独立搜索。
- [计划] 要隔离 K、clustering space 与 selector 的影响，需要在同一预算和同一 final-evaluation 协议下做逐因素 ablation。

## 可支持与不可支持的结论

[已验证] 现有证据可回答初始化方法在固定 300-full 预算下的搜索 validation、prequential surrogate 指标和运行成本差异，也可回答 Cora 各自协议内的 final test。

[未知] 现有证据不能给出跨数据集 final-test 泛化结论，也不能把 K6 与 Top-10 Cora 表作无条件横向排名。
