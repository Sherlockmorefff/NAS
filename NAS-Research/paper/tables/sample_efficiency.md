---
last_updated: 2026-08-14
status: waiting-analysis
claims: [C001, C007, C009]
evidence: [E001, E002, E003]
literature: []
---

# T02 — Sample efficiency candidate

- 研究问题：RQ1。
- 数据来源：E001–E003 链接的 `history_final.json`、`budget_summary.json`、run elapsed；派生统计必须先写入 [`../../results/tables.md`](../../results/tables.md)。
- 统计单位：dataset × method × search seed；matched seed paired comparison。
- 当前完整度：[实验进行中] 原始 histories 完整，但 AUC/threshold/equivalent-cost 表尚未验证。
- 进入论文：否；仅保留表头和判定规则。

| Dataset | Method | BSF@50 | BSF@100 | BSF@150 | BSF@200 | BSF@250 | BSF@300 | AUC (full) | AUC (eq-full) | Evaluations-to-target | Paired effect/CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [TBD-DATASET] | [TBD-METHOD] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] | [TBD-NUMBER] |

TODO：在看 test 前固定 threshold 和 AUC 定义；invalid record、LF epochs/150 换算、wall-clock hardware caveat 必须写入表注。

- 当前能支持：histories 足以开展复算；full budgets 相同而 LF total cost 不同。
- 当前不能支持：任何 sample-efficiency improvement 或 cost-saving 结论。
