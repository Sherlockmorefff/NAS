---
last_updated: 2026-08-14
status: current-audit
claims: [C004, C005, C008, C012]
evidence: []
literature: [D-VAE, TED]
---

# Citation audit

## 已核实引用

| 引用 | 已核实来源 | 可支持的具体表述 | 不可支持 |
| --- | --- | --- | --- |
| Zhang et al., “D-VAE: A Variational Autoencoder for Directed Acyclic Graphs,” NeurIPS 2019, arXiv:1904.11088 | [`../../papers/NAS.md`](../../papers/NAS.md) 引用仓库 README | D-VAE 是 DAG representation；本项目 architecture encoder/decoder 的文献背景 | 完整 16D 都是 D-VAE latent；本项目提出 D-VAE；D-VAE 对当前任务性能有效 |
| Yu, Bi, Tresp, “Active Learning via Transductive Experimental Design,” ICML 2006, DOI 10.1145/1143844.1143980 | [`../../papers/BO.md`](../../papers/BO.md) 引用仓库协议文档 | TED 的原始相关工作；G100/G150 是 TED-inspired selector | 当前实现是原论文逐字复现；TED 导致性能改善 |

## 待核实引用

- GNN NAS、joint NAS+HPO、latent/predictor-based NAS 的最直接工作。
- GP Bayesian optimization、initial design / BO initialization、EI/log-EI/qLogEI 理论与软件引用。
- Diversity/coverage/Schur/DPP/space-filling design 的直接工作。
- Multi-fidelity optimization、LF screening/promotion 的直接工作。
- Knowledge Gradient 的原始及 batch/noisy variant 文献。
- GCN、GAT、GraphSAGE、GIN、GCNII 及 Cora/CiteSeer/PubMed/DBLP/Flickr 数据集原始引用、版本和许可。
- BOOM-Explorer 完整作者、年份、正式条目与 pinned version。

## 可能误引/过度引用风险

- 用 D-VAE 引用支持 4D `h_direct` 或整个 joint search method。
- 用 TED 原论文支持本项目具体 GMM、quota、RBF lengthscale 或 promotion score；这些是实现事实。
- 用 BoTorch API 文档替代 qLogEI/KG 理论来源。
- 从 GNN operation 的 PyG class 名推断原始论文或版本。
- 把内部 [`../../experiments/KG.md`](../../experiments/KG.md) 当作已发表文献或完成 baseline。

## 文献支持的当前正文范围

- `related_work.md` 的 D-VAE 与 TED 两段可保留；其余小节是 TODO 骨架。
- Introduction 的领域 gap 与 novelty 仍需文献支持，C012 保持 `[未知]`。
- Method 中代码/配置事实可写，但必须与 literature claim 分开。

当前审计结论：[未知] 文献覆盖远不足以放行首创性或完整 related work；未发现已核实条目的明显书目信息冲突。
