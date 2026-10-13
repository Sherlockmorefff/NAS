---
last_updated: 2026-08-14
status: active-log
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

# Paper Agent changelog

## 2026-08-14 — Paper Agent 增量建设

- 修改/新增：在保留原 `introduction.md`、`method.md`、`experiments.md` 内容的基础上，建立论文合同、story、Claim registry、glossary、living sections、evidence cards、候选表图、reviews 和 reusable prompts。
- 修改原因：把已有实验/文献事实组织为“事实源 → Evidence → Claim → 最小正文修改 → 审计 → 人工确认”的可追溯流程。
- 新增证据：E001（Cora S0/G100/G150 search + final）、E002（Cora K6 独立协议）、E003（四数据集 60-run historical search）。
- Claim 状态变化：首次登记 C001–C012。C003/C006/C009 的强正向版本被标记为当前证据不支持；C002 只允许“diagnostic profile 无一致支配”；C011 限 Cora 放行。
- 撤回/弱化：不使用“结构化初始化已提高 final accuracy”“相同 full budget 等于相同总成本”“跨数据集已完成 test 泛化”“KG 已实现”等表述。
- 尚未解决：early/cost-aware/coverage 复算、paired effect size 与 multiplicity、跨数据集 final evaluation、文献覆盖、Cora 主 search 的运行 commit/protocol ID。

后续条目必须记录修改文件、证据增量、Claim disposition、被弱化结论和未决问题；仅改文风且不改变含义时也应说明没有 Claim 状态变化。
