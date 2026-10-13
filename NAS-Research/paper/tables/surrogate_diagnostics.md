---
last_updated: 2026-08-14
status: supplementary-candidate
claims: [C002, C007, C010]
evidence: [E003]
literature: []
---

# T04 — Surrogate diagnostics candidate

- 研究问题：RQ2。
- 数据来源：[`../../results/tables.md#surrogate-指标`](../../results/tables.md#surrogate-指标)、[E003](../evidence/E003.md) 与 60-run GP records。
- 统计单位：每个 dataset × method 的 5 个 run-summary 最后记录窗口指标之 mean。
- 当前完整度：最后窗口汇总完整；随 budget trajectory、paired effect/CI 和 multiplicity 缺失。
- 进入论文：可以作为明确标注的补充描述表；不能作为整体 surrogate superiority 主表。

| Dataset | Method | MAE | RMSE | 95% coverage | Spearman | Top-region MAE | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CiteSeer | S0 / G100 / G150 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | E003 |
| PubMed | S0 / G100 / G150 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | E003 |
| DBLP | S0 / G100 / G150 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | E003 |
| Flickr | S0 / G100 / G150 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | 见事实源 | E003 |

候选论文版暂不复制 60 个数，避免在 trajectory/pairing 审计前形成第二份数据。最终表由事实源派生并记录生成脚本 identity。

- 当前能支持：指标方向混合，没有方法跨 dataset/metric 一致支配。
- 当前不能支持：G100/G150 普遍改善 surrogate，或 surrogate 指标因果解释 final performance。
