---
last_updated: 2026-08-14
status: active-workflow
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

# Evidence-driven Paper Agent

本目录是论文视图和人工审核区，不是实验事实的第二份数据库。工作流固定为：

```text
实验或文献完成
  -> 建立/更新 evidence card
  -> 更新 claims.md
  -> 更新候选表格或图表登记
  -> 对正文生成最小修改建议
  -> 执行 evidence / consistency / citation audit
  -> 人工确认
  -> 正式写入论文并记录 changelog
```

## 事实源与论文视图

- 项目级数值事实首先写入并复核于 [`../results/tables.md`](../results/tables.md)；完整图表计划位于 [`../results/figures/README.md`](../results/figures/README.md)。
- 原始事实来自链接到的 history、JSON、CSV、SQLite、日志、配置和 provenance；单个日志不是确定性结论的充分来源。
- [`evidence/`](evidence/README.md) 只索引能够支持或反对论文主张的核实证据、协议、统计边界和 artifact 路径，不复制完整结果库。
- [`claims.md`](claims.md) 是主张放行登记表；[`tables/`](tables/README.md) 与 [`figures/`](figures/README.md) 是投稿视图，不能独立计算或保存另一套结果。
- `abstract.md` 至 `appendix.md` 是 living draft。`[TBD-*]`、`[WAITING-FOR-E###]` 和 `[UNSUPPORTED-CLAIM]` 必须保留到证据放行为止。

## 日常更新

1. 使用 [`prompts/integrate_experiment.md`](prompts/integrate_experiment.md) 或 [`prompts/integrate_literature.md`](prompts/integrate_literature.md) 做完整性检查。
2. 先更新项目级结果或文献卡，再更新 evidence card 和 Claim；不得从正文反向创造事实。
3. 只改受该证据影响的表、图登记和段落；使用 [`prompts/update_section.md`](prompts/update_section.md) 生成最小修改。
4. 人工检查 validation/test 隔离、统计单位、协议身份和禁用表述后再接受修改。
5. 在 [`changelog.md`](changelog.md) 记录 Claim 状态变化、结论弱化和未决问题。

## 每周审计

- 默认只审计、不自动改正文；使用 [`prompts/audit_paper.md`](prompts/audit_paper.md)。
- 更新 [`reviews/evidence_audit.md`](reviews/evidence_audit.md)、[`reviews/consistency_audit.md`](reviews/consistency_audit.md) 和 [`reviews/citation_audit.md`](reviews/citation_audit.md)。
- 从 [`reviews/reviewer_questions.md`](reviews/reviewer_questions.md) 选择最高风险问题，确认是否已有直接证据或必须新增实验。

## 投稿前冻结

1. 冻结 [`paper_contract.md`](paper_contract.md) 中的协议、主张最低证据和实验停止条件。
2. 给每个正文数字绑定 `results/tables.md`、evidence card 与原始 artifact；冻结所用 Git commit、protocol ID、run tag 和生成时间。
3. 确认表格、正文、附录使用同一统计单位与 selection rule；将跨协议结果分表。
4. 完成三类审计和 ICLR reviewer question 回答，由人工确认后生成投稿快照。

## 禁止事项

- 禁止从单个日志、单个 seed、单个最好值或未完成 run 写出普遍结论。
- 禁止把代码实现当成方法有效，把 search validation 当成 final test，把 30 个 training replicates 当成 30 个独立搜索。
- 禁止将 KG 计划写成已实现或已运行，将历史 search completion 写成当前 HEAD 已复现。
- 禁止为补全叙事而猜测数字、显著性、作者、年份、引用或机制解释。

当前可写：实现与协议边界、Cora 和历史跨数据集的描述性事实、现有证据不支持的强主张。当前禁止：跨数据集 test 泛化、KG 收益、成本感知 sample-efficiency 改善以及机制因果结论。
