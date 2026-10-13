---
last_updated: 2026-08-14
status: living-draft
claims: [C001, C002, C003, C004, C005, C006, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

最后更新日期：2026-08-13
当前状态：[已实现未充分验证] 形成可核实的条件式 Introduction 底稿；最终贡献措辞取决于尚未补齐的判定证据。

# Introduction（论文底稿）

## Motivation

Graph neural network performance depends jointly on architectural choices and training hyperparameters. Searching these decisions together is expensive because each observation requires training and validating a concrete GNN, while the mixed structural/continuous domain makes it difficult to obtain an informative initial design for Bayesian optimization.

[已验证] 本项目将架构表示为 12 维 D-VAE latent，并与 4 个直接归一化 HP coordinates 联合搜索。它比较全局 Schur 初始化与分簇、TED/Schur 选择和 low-fidelity promotion 的初始化策略，再以 scratch Exact GP 和 qLogEI 执行在线搜索。完整术语见 [项目总览](../project_overview.md)。

## Gap

[假设] 结构化初始化可能更均衡地覆盖联合域，并通过早期训练信号减少昂贵 full evaluations 的浪费。然而，仅固定 full-evaluation 数会遗漏 low-fidelity 成本；仅报告单个 best validation 会混淆早期效率、surrogate quality、最终 test 和跨 seed 稳定性。

[已验证] 当前 repository evidence 还揭示两个常见报告风险：完整 16D 向量并非全部由 VAE decoder 生成；跨数据集 search validation 也不能替代 final test。这些边界决定本文将成本、统计单位与 selection protocol 显式写入方法和实验。

## Research question

We ask whether structured, low-fidelity-assisted initialization improves (i) early coverage and cost-aware sample efficiency, (ii) GP predictive and ranking quality, (iii) final validation-selected test performance, and (iv) robustness across independent search seeds in joint GNN NAS and HPO.

[已验证] 四个维度相互独立：某方法可能更早找到高 validation 候选，却不改善 surrogate calibration 或 final test。因此本文不以单个最好值作为贡献判定。预注册式判定链见 [研究问题](../research_question.md)。

## Conditional contributions

在相应证据满足前，贡献按以下条件式表达：

1. [已验证] 我们建立了一个可审计的联合 GNN NAS+HPO 实验管线，明确区分 12D architecture D-VAE latent 与 4D direct HP coordinates，并使用候选、数据集、split、checkpoint 和 seed identity 追踪评估。
2. [已验证] 我们实现并比较 global Schur、K=4 clustered TED low-fidelity 和 K=6 clustered Schur low-fidelity 初始化，在相同 300-full 预算之外显式报告 low-fidelity equivalent-full 与 wall-clock 成本。
3. [已验证] 我们以独立 search seed 为方法统计单位，并通过 final-validation-only candidate selection 防止 test leakage；Cora 主协议与 K6 独立协议分开报告。
4. [已实现未充分验证] 现有 Cora 与四数据集 search artifacts 支持对覆盖、surrogate 和稳定性的系统分析；是否形成性能贡献取决于 cost-aware 曲线和预设判定指标。
5. [未知] 跨数据集 final-test 泛化与 KG 对照尚无充分 artifact，因此不作为当前确定贡献。

## Current evidence snapshot

[已验证] 在 Cora 的 5 个 search seeds 上，S0/G100/G150 的 300-full best validation 均值接近，分别为 .8196/.8204/.8192；validation-selected final-test 均值为 .80802/.80480/.80482，区间重叠。历史四数据集 search matrix 中 S0 的 search-validation 均值均最高，但没有匹配的 final test。详表见 [结果表](../results/tables.md)。

[已验证] 因此当前 Introduction 不应声称 clustered initialization 已取得普遍 accuracy improvement。更稳健的现阶段表述是：本文构建并审计了结构化初始化的成本感知评估，现有结果揭示方法效果依赖数据集、指标与评估层次。

## Paper roadmap

[计划] Method 将描述联合坐标、初始化、Exact GP、qLogEI 与成本核算；Experiments 将把 Cora search、Cora final evaluation、K6 独立协议和跨数据集 search 分成四类证据；Discussion 将聚焦混杂因素、失败模式与跨数据集 final-evaluation 缺口。

参考：[Method 底稿](method.md)、[Experiments 底稿](experiments.md)、[文献边界](../papers/NAS.md)。

## Paragraph traceability and overclaim check

| 论证环节 | Claims | Literature / Evidence | 当前状态 | 缺失证据 | 过度表述检查 |
| --- | --- | --- | --- | --- | --- |
| 现有问题 | C001/C009 | 实现与 E001–E003 | [已验证]（限项目成本事实） | BO initialization 领域文献 | 不写成“所有 GNN NAS 都由初始化主导” |
| 文献缺口 | C012 | D-VAE、TED | [未知] | joint NAS+HPO、BO initialization、multi-fidelity 原始文献 | 不使用 first/novel/previous work ignores |
| 研究问题 | C001–C006 | [`paper_contract.md`](paper_contract.md) | [假设] | cost-aware、coverage、LF quality 与 final-test 证据 | 保持问句/条件式 |
| 方法直觉 | C004/C005 | TED + 已实现 selector | [已实现未充分验证] | matched mechanism ablation | 不把组件设计写成机制已成立 |
| 证据设计 | C001/C002/C003/C006/C007/C009/C011 | E001/E002/E003 | [已验证]（设计与现有范围） | 跨数据集 final eval、paired statistics | 明确 historical search / Cora final / K6 独立协议 |
| 条件式贡献 | C001/C002/C003/C006/C012 | Claim registry | [实验进行中] | 见各 Claim missing evidence | 摘要和贡献不得比结果强 |

当前可写：问题、实现、证据层次、Cora 与 historical search 的描述性/negative result。当前禁止：普遍性能优势、机制因果性、跨数据集 test 泛化和首创性。
