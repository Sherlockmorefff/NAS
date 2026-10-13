---
last_updated: 2026-08-14
status: candidate-index
claims: [C001, C002, C003, C004, C005, C006, C007, C009, C010, C011]
evidence: [E001, E002, E003]
literature: []
---

# Paper table candidates

本目录只保存准备进入论文的表格视图。所有数值必须先存在于 [`../../results/tables.md`](../../results/tables.md)，并同时链接 evidence card 与原始 artifact。这里不得运行独立计算、修正事实源或形成第二套数字；发现差异时先停止写作并修复项目级结果/证据链。

| Table ID | 文件 | 研究问题 | Claims | 当前完整度 | 可以进入论文？ |
| --- | --- | --- | --- | --- | --- |
| T01 | [`main_results.md`](main_results.md) | RQ3/RQ4 | C003/C006/C007/C011 | Cora complete；cross-dataset search-only；K6 independent | 是，按 evidence tier 分 panel |
| T02 | [`sample_efficiency.md`](sample_efficiency.md) | RQ1 | C001/C009 | histories available，统一复算缺失 | 否 |
| T03 | [`ablation.md`](ablation.md) | RQ2 | C004/C005 | matched experiments 缺失 | 否 |
| T04 | [`surrogate_diagnostics.md`](surrogate_diagnostics.md) | RQ2 | C002/C010 | final-window summary complete；trajectory 缺失 | 可作为补充表，需明确边界 |
| T05 | [`computational_cost.md`](computational_cost.md) | RQ1/RQ6 | C001/C009 | full/LF 已核实；统一 overhead 分析缺失 | 部分可用 |
| T06 | [`dataset_statistics.md`](dataset_statistics.md) | setup/RQ4 | C010 | manifest paths 已知，统计尚未抽取 | 否 |

## 更新规则

1. 从 `results/tables.md` 选择已经验证的行，不在此重算。
2. 在表头/表注中写出统计单位、seed、CI、metric tier、protocol mismatch、invalid handling。
3. 每个数字链接一个 E-card；E-card 的 Provenance 再链接 raw artifact。
4. 更新正文前检查 Claim 的允许/禁止表述，并在 changelog 记录。
5. 若表格必须看 test 才决定方法/seed/metric，视为 selection risk，停止并交人工审核。
