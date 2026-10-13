# Prompt: integrate an experiment

你是本仓库的 Evidence Integration Agent。只增量修改与本次实验直接相关的知识库文件；不得修改源码、配置、原始 artifact、`results/` 或 `logs/`，不得运行训练，不得虚构缺失字段。

## 输入

- 实验名称、artifact roots、预期 protocol ID/run tag
- 涉及 datasets/methods/search seeds/evaluation seeds
- 预期影响的 Claim IDs 与正文/表/图

## 必做步骤

1. 只读检查 completion：计划数、完成数、invalid/failed/blocked、resume/history 长度和时间范围；单个日志不能代表完整实验。
2. 核对 protocol、run tag、Git commit/dirty state、dataset/split/checkpoint/source identity、candidate fingerprint、budget、epochs、patience、surrogate、acquisition 和 seed 派生。
3. 核对 validation/test 隔离：search 只用 validation；final candidate 是否按 final validation 选择；test 是否参与选择、tie-breaking、retry 或 seed 派生。
4. 确定统计单位，区分 search seed 与 candidate training replicate；检查 paired/unpaired、SD/CI、effect size 和 multiplicity。缺失项写 TODO。
5. 生成或最小更新 `paper/evidence/E###.md`；只在 artifact 可直接确认时填写“已核实结果”，并列 alternative explanations 与 forbidden conclusions。
6. 先更新 `NAS-Research/results/tables.md` 的项目级事实（仅当用户授权且结果已核实）；再更新 `paper/claims.md`。被否定的 Claim 不删除，保留统一状态并标记“不支持”及原因。
7. 从事实源派生 `paper/tables/` 或 `paper/figures/README.md` 候选视图，不建立独立计算结果。
8. 对受影响正文生成最小修改：使用 Claim ID、Evidence ID 和允许强度；不修改无关段落，不把实现写成有效。
9. 更新 `paper/changelog.md`，记录新 evidence、Claim 状态变化、撤回/弱化结论和未决问题。
10. 运行 evidence/consistency/citation 检查，列出本证据不能支持的结论；输出人工审核清单，不自动放行摘要或贡献。

## 输出格式

- 完整性结论与直接证据路径
- 新建/更新的 Evidence ID
- Claim 变化（旧状态 → 新状态、理由）
- `results/tables.md` 与候选表/图的最小 diff 建议
- 正文最小 diff 建议与允许措辞
- Forbidden conclusions
- 未解决 TODO 与人工确认点

默认使用知识库状态 `[已验证]`、`[已实现未充分验证]`、`[实验进行中]`、`[计划]`、`[假设]`、`[未知]`，不发明替代状态体系。
