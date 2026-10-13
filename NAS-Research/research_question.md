最后更新日期：2026-08-13
当前状态：[已实现未充分验证] 研究问题可由现有矩阵部分检验，但机制隔离、KG 与跨数据集 final test 尚不完整。

# 研究问题与可证伪贡献链

## 总问题

[假设] 在固定联合 NAS+HPO 搜索域中，结构化且考虑 low-fidelity 的初始化，是否能以可接受的额外成本改善早期覆盖和 surrogate 学习，从而提高 BO 的样本效率、最终 validation/test 表现与跨 seed 稳定性？

[已验证] “固定联合域”具体指 12D architecture D-VAE latent 加 4D direct HP coordinates；术语、方法映射和评估层次见 [项目总览](project_overview.md)。

## 四个独立可检验维度

### RQ1：初始化覆盖与早期效率

[假设] G100/G150/K6-LF 的分簇与 low-fidelity promotion 能比 S0 更早找到高 validation 候选。

判定指标：best-so-far AUC、固定预算点的 best validation、evaluations-to-target、候选结构/HP 多样性。必须同时使用 full-evaluation 与 equivalent-full 成本轴，并报告 wall time。若改善仅存在于忽略 LF 成本的横轴上，则不能称为成本效率提升。

### RQ2：GP surrogate quality

[假设] 更覆盖搜索空间的初始化能降低 prequential error、改善 95% interval coverage 与 ranking quality，并在 top region 保持可用精度。

判定指标：prequential MAE/RMSE、95% coverage、Spearman、top-region MAE，以及随 full budget 变化的曲线。单一 seed 或单一最后窗口不构成充分机制证据。

### RQ3：最终 validation 与 test

[假设] 初始化带来的早期优势能持续到 300-full search，并通过 validation-only 的候选选择转化为 test 改善。

判定指标：每个 search seed 的 search-best validation；随后对 Top-K 候选做固定 replicates，按 final validation 选中候选，再以 search seed 为单位汇总 test。test 不参与选择、tie-breaking 或重跑决定。

### RQ4：跨 seed 稳定性

[假设] 结构化初始化降低方法对 search seed 的敏感性。

判定指标：跨 5 search seeds 的样本 SD、t-based 95% CI、失败/invalid 率以及候选选择一致性。较高均值与较高方差应分别报告，不能用单个最好 seed 代表方法。

## 证据矩阵

| 问题 | Cora S0/G100/G150 | Cora K6-LF | 四数据集历史 search | 仍缺少 |
| --- | --- | --- | --- | --- |
| RQ1 早期效率 | histories 可计算 | histories 可计算，但协议独立 | histories 可计算 | 统一曲线与等价成本图 |
| RQ2 surrogate | run summaries/GP 记录可用 | run summary 可用 | 60-run summaries 可用 | 机制型逐因素 ablation |
| RQ3 最终性能 | Top-10 × 30 complete | Top-7 × 30 complete，协议不同 | 只有 search validation | 跨数据集 final eval |
| RQ4 稳定性 | 5 search seeds | 5 search seeds | 每格 5 search seeds | 更多 seeds 或预注册 power 分析 |

## 贡献陈述的放行标准

### C1：初始化覆盖与样本效率

[计划] 只有当预注册的 early-budget/AUC 指标跨 seeds 改善，并在 equivalent-full 或 wall-clock 轴上仍成立，才写成“提高样本效率”。否则只写“改变了早期搜索轨迹”。

### C2：surrogate 改善

[计划] 只有当 error、coverage 与 ranking 指标形成一致方向的证据链，且不是单个 dataset/seed 的偶然结果，才写成“改善 surrogate quality”。

### C3：最终性能

[计划] 只有 validation-only selection 后的 method-level test 分布支持改善，才写成“改善最终性能”。当前 Cora 主表均值区间重叠，不能宣布确定优势；K6 因协议不同独立报告。

### C4：跨数据集稳健性

[计划] 只有补齐匹配的 final evaluation，并在多个数据集上得到一致方向结果，才写成“跨数据集泛化”。当前四数据集 search validation 不满足该条件。

## 当前结果对研究问题的约束

[已验证] Cora search validation 的 S0/G100/G150 均值为 .8196/.8204/.8192，95% CI 高度重叠。Cora validation-selected final test 的均值为 .80802/.80480/.80482，也不能支持结构化初始化已改善最终 test 的确定结论。完整统计见 [结果表](results/tables.md)。

[已验证] 历史跨数据集 search validation 中，S0 在四个数据集的均值均高于 G100/G150。这与“分簇初始化普遍提高最终 search validation”的强假设不一致，但尚不能回答 early efficiency、机制或 final test。

[未知] KG 未实现，不能纳入贡献比较。对照设计见 [KG 实验页](experiments/KG.md)。

## 论文主张模板

[计划] 在证据补齐前，Introduction 只使用条件式表达：

> 我们评估结构化初始化是否在固定 full-evaluation 预算下改善联合 GNN NAS+HPO 的覆盖、surrogate 学习和最终表现，并显式核算 low-fidelity 成本。现有 artifact 支持对 search 与 Cora final evaluation 的审计；跨数据集 test 结论仍待补齐。

[已验证] 该模板不把设计动机提前写成结果，也不以单个最好值宣布贡献。
