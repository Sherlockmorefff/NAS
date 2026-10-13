---
last_updated: 2026-08-14
status: planned
claims: [C004, C005, C007]
evidence: [E001, E002]
literature: []
---

# T03 — Initialization ablation candidate

- 研究问题：RQ2，区分 coverage、candidate quality 与 surrogate mechanism。
- 数据来源：未来新 protocol；现有 E001/E002 只用于说明组件已绑定，不能填 matched ablation 数值。
- 统计单位：同 candidate pool/dataset/search seed 的 paired runs。
- 当前完整度：[计划]；没有满足控制要求的 artifact。
- 进入论文：否。

| Variant | K | Clustering features | Selector | LF shortlist | Promotion score | Full budget | Coverage | LF→full ranking | Search outcome | Evidence |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Global reference | TODO | TODO | TODO | TODO | TODO | TODO | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [WAITING-FOR-E###] |
| Matched variant | TODO | TODO | TODO | TODO | TODO | TODO | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [WAITING-FOR-E###] |

最低控制：一次只改变 global/clustered allocation、12D/16D clustering、TED/Schur、K 或 promotion score 之一；其余 pool、budget、seed、surrogate、acquisition、training 和 final protocol 固定。

- 当前能支持：现有方法包含不同组件组合。
- 当前不能支持：把 observed difference 归因于 K、clustering、TED/Schur 或 LF 任一单项。
