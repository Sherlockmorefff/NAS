最后更新日期：2026-08-13
当前状态：[已验证] 当前实现、正式配置、checkpoint 形状与主要 artifact 已建立统一术语映射。

# 项目总览

## 项目目标

[已验证] 本项目研究在图神经网络 Neural Architecture Search（NAS）与超参数优化（HPO）的联合空间中，初始化策略与贝叶斯优化如何影响候选覆盖、surrogate 质量、样本效率和最终性能。当前主实验使用 Cora，并保留 CiteSeer、PubMed、DBLP、Flickr 的历史正式 search matrix。

[假设] 研究动机是：结构化初始化可能比不分簇的全局选择更早覆盖多样而有潜力的区域，进而改善 GP 的早期预测和 BO 效率。该机制尚不能仅凭单个 best value 视为成立；判定链见 [研究问题](research_question.md)。

## 联合搜索空间与术语边界

[已验证] 正式 `global4` 搜索向量为 16 维：

| 坐标 | 维度 | 语义 | Phase4 实际解码路径 |
| --- | ---: | --- | --- |
| `z_arch` | 12 | 架构 D-VAE 的连续 latent | 由 D-VAE decoder 解码为 DAG |
| `h_direct` | 4 | 归一化 HP 搜索坐标 | `decode_hp_by_mode()` 直接映射为 `lr/dropout/hidden_dim/l2` |

[已验证] checkpoint 内确有独立 HP VAE，其 latent 为 4 维；但 Phase4 evaluation 没有把 `h_direct` 送入 `hp_vae.decode()`。因此论文应称 12 维部分为“architecture VAE latent”，称后 4 维为“direct normalized HP coordinates”，不能把完整 16 维统称为 VAE latent。

证据：[bo_phase4.py](../bo_phase4.py)、[eval_utils.py](../eval_utils.py)、[hp_modes.py](../hp_modes.py)、[JointSpaceVAE 定义](../nas_space.py)、[正式 checkpoint](../checkpoints/joint_model_pipeline_global4_best.pth) 与 [reference manifest](../artifacts/reference/reference_manifest.json)。

[已验证] 四个 HP 坐标映射到：log10 learning rate `[-4, -1.5]`、dropout `[0.1, 0.6]`、hidden dimension `{16,32,64,128,256,512}`、L2 `{1e-5,5e-5,1e-4,5e-4,1e-3}`。

## 架构表示与搜索空间

[已验证] 架构表示模型是标准 VAE 家族中的 D-VAE，不是 VQ-VAE。正式 checkpoint 的权重形状对应 12 维架构 latent；HP VAE 的 encoder latent 为 4 维。

[已验证] 架构 DAG 最多包含 5 个内部操作节点，另有 start/end 节点。可选操作为 GCN、GAT、GraphSAGE、GIN、Identity 与 GCNII。具体图编码/解码与约束见 [models.py](../models.py) 和 [nas_space.py](../nas_space.py)。

## 正式 BO 管线

```text
固定 checkpoint 与候选域
  -> 初始化选择与 full/low-fidelity 评估
  -> scratch Exact GP（固定域归一化）
  -> qLogEI 候选选择
  -> full GNN evaluation
  -> history / run summary / provenance
  -> Top-K final evaluation（若存在）
```

[已验证] 正式 S0/G100/G150 配置使用 scratch Exact GP、fixed-domain normalization、qLogEI，以及 warm refit（每 5 次更新、20 steps）。随机 acquisition 基线已实现。DKL GP 有代码和测试，但不属于正式配置；Knowledge Gradient 未确认实现。

证据：[accuracy_gp.py](../surrogate/accuracy_gp.py)、[dkl_accuracy_gp.py](../surrogate/dkl_accuracy_gp.py)、[bo_phase4.py](../bo_phase4.py)、[正式方法配置](../configs/cross_dataset_methods.json)。

## 初始化方法映射

| 简称 | 正式 label | 核心预算 |
| --- | --- | --- |
| S0 | `global_schur` | 50 seed + 250 qLogEI = 300 full |
| G100 | `gmm_exp100` | 50 seed + 100 promoted + 150 qLogEI = 300 full |
| G150 | `gmm_exp150` | 50 seed + 150 promoted + 100 qLogEI = 300 full |
| K6-LF | `fixedk6_gmm_schur_lowfid` | 50 seed + 200 LF/20 epochs + 100 promoted + 150 qLogEI = 300 full |

[已验证] G100/G150 还执行每簇至多 100 个 20-epoch low-fidelity evaluations；实际总数随 search seed 的簇容量变化，且不包含在 300 full 中。完整实测成本和公平性约束见 [初始化实验](experiments/initialization.md)。

## 数据与评估层次

[已验证] 论文必须区分四类证据：Cora search、Cora S0/G100/G150 final evaluation、K6 独立 final-evaluation 协议、四数据集历史 search。它们的候选选择单位、训练 epochs 与可支持结论不同，详见 [论文实验章节底稿](paper/experiments.md) 与 [结果表](results/tables.md)。

| 层次 | 指标来源 | 可以回答 | 不能替代 |
| --- | --- | --- | --- |
| search validation | 单次搜索 history | BO 发现质量、样本效率 | final test |
| surrogate diagnostics | run summary / GP CSV | calibration 与 ranking 质量 | 下游准确率 |
| candidate final evaluation | 30 training replicates | 候选的 validation/test 分布 | 30 个独立搜索 |
| method-level final test | 5 个 search-seed 选中候选 | 方法稳定性与 test 汇总 | 跨数据集泛化 |

## 仓库模块与证据地图

| 模块 | 主要文件 | 职责 |
| --- | --- | --- |
| 搜索空间/解码 | [nas_space.py](../nas_space.py)、[hp_modes.py](../hp_modes.py)、[eval_utils.py](../eval_utils.py) | 联合坐标、条件参数与 evaluation 解码 |
| Phase4 搜索 | [bo_phase4.py](../bo_phase4.py) | 候选池、GP、acquisition、history |
| 初始化 | [initialization_gmm_schur.py](../initialization_gmm_schur.py)、[initialization_wgmm_ted.py](../initialization_wgmm_ted.py) | global/clustered selector 与 LF promotion |
| Surrogate | [surrogate/accuracy_gp.py](../surrogate/accuracy_gp.py)、[surrogate/dkl_accuracy_gp.py](../surrogate/dkl_accuracy_gp.py) | Exact GP 与非正式 DKL 实现 |
| 数据/矩阵 | [dataset_utils.py](../dataset_utils.py)、[cross_dataset_runner.py](../cross_dataset_runner.py)、[formal_matrix.py](../formal_matrix.py) | 数据身份、split、资源 gate、矩阵运行 |
| Final evaluation | [final_eval.py](../final_eval.py) | Top-K、replicate seeds、validation selection |
| 当前引用资产 | [artifacts/reference/](../artifacts/reference/) | checkpoint manifest、预检与决策 |
| 历史原始证据 | [legacy_artifacts/pre_20260813/](../legacy_artifacts/pre_20260813/) | 迁移前 results/logs 的只读归档 |

## Artifact 与 provenance 规则

[已验证] 迁移后的历史 artifact 可作为正式证据，但引用必须保留 `legacy_artifacts/pre_20260813/` 路径，并声明其旧 commit、runtime source manifest 和 dirty-worktree provenance。当前 HEAD 的资源 gate 不会反向删除历史结果；历史完成也不会自动证明当前 HEAD 可重跑。

[已验证] candidate fingerprint、dataset/split/checkpoint/source identity、固定训练与 evaluation seeds、history/provenance/resume metadata 是实验归属与复核所需机制，论文数字必须能够回到原始 JSON/CSV。

## 当前证据边界

- [已验证] Cora S0/G100/G150 与 K6-LF 均有 5 个完整 search seeds 和对应 final evaluation。
- [已验证] 四个跨数据集的 S0/G100/G150 共 60 个历史搜索完整。
- [未知] 没有与跨数据集 60-run 矩阵匹配的 final-test artifact。
- [未知] 没有 KG 实现、配置或结果。
- [已验证] 仓库内可核实的论文资产有限；不联网补全作者或结论，见 [文献目录](papers/NAS.md)。
