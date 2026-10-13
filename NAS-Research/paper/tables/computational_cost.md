---
last_updated: 2026-08-14
status: partially-eligible
claims: [C001, C009]
evidence: [E001, E002, E003]
literature: []
---

# T05 — Computational cost candidate

- 研究问题：RQ1/RQ6。
- 数据来源：[`../../results/tables.md`](../../results/tables.md)、E001/E002/E003 的 budget summaries 和 run summaries。
- 统计单位：每 method/search seed 的实际 LF epochs、300 full evaluations 和 elapsed；wall-clock 只在相同历史环境内描述。
- 当前完整度：full/LF work units 已核实；acquisition/selector overhead 与统一硬件 GPU-hours 未完成。
- 进入论文：部分可用，必须披露限制。

| Method | Full evaluations | LF candidates × epochs | LF equivalent-full | qLogEI full | Wall-clock status | Evidence |
| --- | ---: | --- | --- | ---: | --- | --- |
| S0 | 300 | 0 | 0 | 250 | historical run summaries available | E001/E003 |
| G100 | 300 | Cora 207–356；cross-dataset 259–400 ×20 | Cora 27.60–47.47；cross-dataset 34.53–53.33 | 150 | historical run summaries available | E001/E003 |
| G150 | 300 | 与同-seed G100 shortlist 相同 ×20 | 同上 | 100 | historical run summaries available | E001/E003 |
| K6-LF | 300 | 200×20 | 26.67 | 150 | independent historical protocol | E002 |

表注：`equivalent-full = low_fidelity_actual_epochs / 150`，只是一致工作量换算；不代表实际 wall time。历史 wall time 受 GPU、并发和 source version 影响。

- 当前能支持：所有方法 full budget 相同，LF 方法总训练工作量更高。
- 当前不能支持：LF 方法总成本相同/更低、KG overhead 或跨硬件效率排名。
