---
last_updated: 2026-08-14
status: skeleton
claims: [C004, C005, C008, C012]
evidence: []
literature: [D-VAE, TED]
---

# Related Work（证据约束骨架）

本节只综合 [`../papers/`](../papers/) 中已核实的文献卡；代码 API 或常识不能替代引用。当前文献库不足，因此以下除 D-VAE 与 TED 外均为待阅读结构。

## Architecture representation and GNN NAS

[已验证] Zhang et al. 的 D-VAE 将 DAG 表示为 variational-autoencoder latent；本项目使用 D-VAE 风格的 architecture encoder/decoder，但不提出 D-VAE，也不能把完整 16D joint coordinate 都归于该工作。书目信息与项目关系见 [`../papers/NAS.md`](../papers/NAS.md)。

[计划] TODO：核实 graph NAS、latent-space NAS、predictor/surrogate-based NAS 的原始论文，并逐篇记录搜索空间、预算、数据集、统计单位及与本项目的区别。

## Joint NAS + HPO

[未知] 当前仓库没有可核实的 joint NAS+HPO 文献卡。TODO：优先补最直接的联合搜索方法，判断 C012 的创新性主张是否应削弱。没有该审计前，不使用“首次”“新颖联合方法”等措辞。

## Bayesian optimization and initialization

[已验证] Yu, Bi, and Tresp (2006) 提出 Transductive Experimental Design；本项目 G100/G150 使用 TED-inspired clustered selector，但按自己的固定域、RBF kernel 和 quota 实现，不能称逐字复现。书目信息见 [`../papers/BO.md`](../papers/BO.md)。

[计划] TODO：核实 GP BO、initial design / BO initialization、Expected Improvement / log-EI 与 qLogEI 的原始论文和软件引用，区分理论算法、数值稳定变体和 BoTorch 实现。

## Diversity/coverage-based design

[未知] 当前没有独立文献卡支持“结构化覆盖是相关工作缺口”。TODO：核实 DPP/Schur/space-filling/clustered design 的最直接工作，并区分 coverage heuristic 与可证明 experimental design。

## Multi-fidelity optimization

[未知] 当前没有已核实 multi-fidelity BO 文献卡。TODO：补 LF screening、successive promotion/multi-fidelity BO 的原始工作，核对成本定义和 fidelity correlation 假设。

## Knowledge Gradient

[未知] 仓库没有 KG 文献资产、实现或结果。TODO：在 KG 被纳入实验前核实原始 acquisition、当前可用 API、batch/noisy variant 和成本；现在只可引用 [`../experiments/KG.md`](../experiments/KG.md) 作为内部计划，不能作为文献或 baseline 证据。

## GNN operations and datasets

[未知] GCN、GAT、GraphSAGE、GIN、GCNII 及数据集原始引用尚未核实；待办清单见 [`../papers/GNN.md`](../papers/GNN.md)。在补齐前，Method 可描述代码事实，Related Work 不补猜测的作者或年份。

当前可写：D-VAE 和 TED 的准确关系。当前禁止：完整综述、首创性判断、未经文献卡核实的作者/年份/结论。
