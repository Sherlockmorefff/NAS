最后更新日期：2026-08-13
当前状态：[未知] 仓库内没有可核实的 GNN 原始论文资产；本页仅登记实现映射与引用 TODO。

# GNN 文献笔记

## 当前搜索操作

[已验证] 架构搜索空间包含 GCN、GAT、GraphSAGE、GIN、Identity 和 GCNII，最多 5 个内部操作节点。该事实来自 [nas_space.py](../../nas_space.py) 与 [models.py](../../models.py)，不能替代算法原始论文引用。

| 操作 | 实现名称 | 文献状态 |
| --- | --- | --- |
| GCN | `GCNConv` | [计划] TODO：核对原始论文与版本 |
| GAT | `GATConv` | [计划] TODO：核对原始论文与版本 |
| GraphSAGE | `SAGEConv` | [计划] TODO：核对原始论文与版本 |
| GIN | `GINConv` | [计划] TODO：核对原始论文与版本 |
| Identity | `Identity` | [已验证] 搜索空间操作，无独立模型引用要求 |
| GCNII | GCNII operation | [计划] TODO：核对原始论文与实现语义 |

## 数据集文献

[已验证] 代码使用 PyG Planetoid 的 Cora/CiteSeer/PubMed、CitationFull DBLP、Flickr，以及被资源 gate 阻断的 OGB ogbn-arxiv。数据处理事实见 [dataset_utils.py](../../dataset_utils.py)。

- [计划] TODO：补齐每个数据集原始出处、版本、许可与 split 文献。
- [计划] TODO：区分 Planetoid public split、DBLP deterministic constructed split、Flickr official masks 与 OGB official split。
- [计划] TODO：在 Experiments 表中报告节点/边/特征/类别统计时，从实际 dataset manifest 读取，而非凭记忆填写。

## 写作边界

[未知] 当前仓库没有 `.bib`、论文 PDF 或 GNN 阅读笔记。本轮不联网补全，因而不能把本页当作完整 related work。后续新增引用时应附原始来源、与搜索操作的具体关系和实现差异。
