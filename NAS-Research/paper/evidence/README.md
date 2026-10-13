---
last_updated: 2026-08-14
status: active-index
claims: [C001, C002, C003, C004, C005, C006, C007, C009, C010, C011]
evidence: [E001, E002, E003]
literature: []
---

# Evidence cards

本目录是论文主张所依赖的证据索引，不复制 [`../../results/tables.md`](../../results/tables.md) 的完整结果，也不替代原始 artifact。编号按创建顺序使用 `E001`、`E002`、`E003`；不得复用或重排已有编号。计划实验一般不建结果卡；若确需登记，必须在 frontmatter 和“实验状态”中标记 `[计划]`，且“已核实结果”只能写 TODO。

## 当前索引

| Evidence ID | 证据族 | 状态 | 主要 Claims | Artifact tier |
| --- | --- | --- | --- | --- |
| [E001](E001.md) | Cora S0/G100/G150 search +共同 final evaluation | [已验证] | C001/C002/C003/C005/C007/C009/C010/C011 | historical search + final summaries |
| [E002](E002.md) | Cora K6-LF 独立协议 | [已验证] | C001/C004/C005/C007/C009/C010/C011 | historical search + final summaries |
| [E003](E003.md) | 四数据集 S0/G100/G150 historical search | [已验证] | C001/C002/C006/C007/C009/C010 | 60-run search only |

## 模板

每张卡使用如下 frontmatter；未知字段写 `null` 或明确 TODO，不猜测：

```yaml
---
evidence_id: E###
title:
status:
created_at: YYYY-MM-DD
updated_at: YYYY-MM-DD
supports: []
contradicts: []
datasets: []
methods: []
search_seeds: []
evaluation_seeds: []
git_commit:
protocol_id:
run_tag:
---
```

正文固定包含：研究问题、实验状态、实验协议、已核实结果、统计分析、解释、Provenance、Claim 影响、Eligible paper update、Forbidden conclusion。每次更新必须先核实 completion、protocol identity、budget、seed 与 validation/test 隔离，再同步 [`../claims.md`](../claims.md) 和 [`../changelog.md`](../changelog.md)。

## 证据边界

- `git_commit` 指 artifact 记录的运行 commit，不是当前 HEAD；若 artifact 未记录则为 `null` 并保留 TODO。
- `run_tag` 只是运行家族标识，不能自动等同于正式 `protocol_id`。
- 旧绝对路径保留于原 JSON 作为 provenance；知识库链接使用迁移后的 `legacy_artifacts/pre_20260813/` 路径。
- 不从单个 `run_summary.json` 概括方法结论；方法级统计单位是独立 search seed。
