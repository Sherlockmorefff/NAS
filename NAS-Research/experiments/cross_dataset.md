最后更新日期：2026-08-13
当前状态：[已验证] 历史 60-run search 矩阵完整；当前 HEAD 的重新启动 gate 与历史状态冲突；跨数据集 final evaluation 缺失。

# 跨数据集实验

## 数据集与 split

| 数据集 | loader/split | 特征处理 | 当前角色 |
| --- | --- | --- | --- |
| Cora/CiteSeer/PubMed | PyG Planetoid 公共 split | `NormalizeFeatures` | Cora 为历史参考；CiteSeer/PubMed 为 formal |
| DBLP | PyG CitationFull DBLP；seed 0 的 20/class train、500 val、1000 test | `NormalizeFeatures` | formal |
| Flickr | 官方 features 与 masks | 不额外归一化 | formal，但当前资源 gate 未放行 |
| ogbn-arxiv | OGB 官方 split，图转无向 | 由 loader 处理 | 因资源不足 blocked，非当前 formal matrix |

证据：[dataset_utils.py](../../dataset_utils.py)、[跨数据集协议](../../docs/cross_dataset_protocol.md)、[正式方法配置](../../configs/cross_dataset_methods.json)。

## 三种必须分开的状态

### 1. 历史 60-run 搜索已完成

[已验证] 归档包含 CiteSeer、PubMed、DBLP、Flickr × S0/G100/G150 × search seeds 5–9，共 60 个 `run_summary.json` 和 60 个 `history_final.json`。每条 history 长度为 300，停止原因为 `fixed_iteration_complete`。共 18,000 条 full-evaluation 记录，其中 3 条 invalid。

[已验证] 这些是“历史正式 artifact”：provenance 指向旧 commit `d73c849…`、同一 runtime source manifest，并记录 dirty worktree。迁移清单声明 9,017 个文件、3,359,970,058 bytes 被保留且内容未重写。它们不能被描述为由当前 HEAD 原样复现。

证据：[历史矩阵](../../legacy_artifacts/pre_20260813/results/formal_search_20260731_8831cd0a5337/)、[迁移 JSON](../../legacy_artifacts/pre_20260813/migration_manifest.json)、[迁移说明](../../docs/artifact_migration_20260813.md)。

### 2. 当前 HEAD 再启动 gate 冲突

[已验证] 当前 config 把 seeds 5–9 定义为 formal seeds，但 [training_mode_decisions.json](../../artifacts/reference/cross_dataset_v1_preflight/training_mode_decisions.json) 只授权 CiteSeer/PubMed/DBLP seeds 0–4，Flickr 只授权 seed 0；当前 runner 会拒绝 formal seed 5。

[已验证] [formal_search_seed_5_9_decision.json](../../artifacts/reference/cross_dataset_v1_preflight/formal_search_seed_5_9_decision.json) 先记录 `blocked_cuda_unavailable`。其后产生的真实 NVIDIA GPU 预检显示：CiteSeer、PubMed、DBLP 的 seeds 5–9 均 pass；Flickr seeds 5–9 各自初次与 retry1 均 OOM。这里按时间记录两个事实，不合并成单一状态。

执行状态：CiteSeer/PubMed/DBLP 的资源预检通过，但授权表尚未覆盖 formal seeds；Flickr 的最新资源预检失败。历史 Flickr 搜索仍然有效存在，但不代表当前协议重新获准。

证据：[当前预检结果](../../results/cross_dataset_v1_preflight/)、[cross_dataset_runner.py](../../cross_dataset_runner.py)、[formal_matrix.py](../../formal_matrix.py)。

### 3. 跨数据集 final evaluation 缺失

[未知] 历史矩阵下未找到与 60 个搜索一一匹配的 `final_eval_summary_global4.json` 或 final-test artifact。因此当前只能报告 search validation、surrogate 与成本，不得把 validation 结果写成 test 泛化。

## 协议与预算

[已验证] 三个方法均使用 300 次 full evaluation、scratch Exact GP、固定域归一化和 qLogEI；正式训练为 150 epochs、patience 40。预算和 low-fidelity 等价成本见 [初始化实验](initialization.md#预算与公平性)。

[已验证] candidate fingerprint 和 evaluation seed 方案跨数据集保持稳定；dataset、split、checkpoint 与 source identity 参与 run/cache attribution，避免不同数据集记录碰撞。

## 当前可回答问题

- [已验证] 可比较四个数据集上 S0/G100/G150 的五个 search-seed validation 分布、prequential surrogate 指标、invalid 数与 wall time。
- [已验证] 可检查方法排序是否跨数据集一致；现有均值显示 S0 在四个数据集的历史 search validation 均为最高，但这只是描述性结果，区间明显重叠且没有 final test。
- [未知] 不能回答跨数据集 test 泛化、最终候选稳定性或当前 HEAD 的完全复现性。

## 下一步放行条件

- [计划] 更新授权决策，使其与 formal seeds 5–9 一致，并保留资源预检 identity。
- [计划] 为 Flickr 明确采用资源可行的协议或继续 blocked；不得静默改为不同训练模式后混入同一表。
- [计划] 在冻结的 Top-K、replicate、epochs、patience 和 validation selection 规则下补齐跨数据集 final evaluation。
- [计划] 补齐后只以 search seed 为方法级推断单位；结果写入 [结果表](../results/tables.md)，而不是覆盖历史 search 表。
