---
last_updated: 2026-08-14
status: living-draft
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: []
---

# Limitations

1. **搜索空间与表示依赖。** 当前 joint domain 固定为 12D architecture D-VAE latent 与 4D direct HP coordinates，候选池为 768。观察到的排序不能外推到其他 architecture representation、候选域或更大搜索空间；architecture D-VAE 的表示质量尚未消融。
2. **数据集范围与 evidence tier。** Cora 有 search 和 final evaluation；CiteSeer/PubMed/DBLP/Flickr 只有历史 search validation。没有跨数据集 final test，Flickr 当前 resource gate 又与历史完成状态不同。
3. **预算公平性。** S0/G100/G150 都有 300 full evaluations，但 LF 方法额外执行 20-epoch evaluations。Equivalent-full 是统一换算，不等同于实测 wall time；旧 wall time 还受硬件、并发和实现版本影响。
4. **机制混杂。** G100/G150/K6 同时改变 clustering feature、K、selector、shortlist、promotion 数和剩余 qLogEI 次数；现有差异不能归因于单一组件。coverage 与 LF candidate-quality 指标尚未登记。
5. **Surrogate 证据。** 当前汇总包含 prequential 最后窗口 MAE/RMSE/coverage/Spearman/top-region MAE，但没有完整学习曲线和一致方向。单项改善不能证明 surrogate 整体更好，也不能解释最终 test。
6. **Low-/full-fidelity 关系。** LF–full correlation、promotion precision 和 rank preservation 未完成统计分析；LF 可能只是增加计算而非提高候选质量。
7. **样本量与统计。** 方法级单位只有 5 个 search seeds。已有 t-CI 不能替代 paired effect size、practical threshold、power analysis 或多重比较方案，区间重叠也不能证明等价。
8. **Final-evaluation selection。** Cora final candidates 按 final validation 选择，test 不参与该选择；这降低 test-selection leakage，但不能排除所有 researcher degrees of freedom。30 replicates 不是 30 个独立搜索。
9. **协议不一致。** K6 final evaluation 使用 Top-7、150 epochs、patience 40，主 Cora 协议使用 Top-10、300 epochs、patience 80；两组 test 不能直接排名。
10. **历史 provenance。** 四数据集和 K6 artifacts 来自旧 commit，并明确记录 dirty worktree/runtime source manifest。它们是可审计历史证据，不等于当前 HEAD 原样复现；Cora 主 search 的运行 commit/protocol ID 仍无法确认。
11. **跨数据集可迁移性。** 四个 historical datasets 的 S0 mean search validation 均最高，但多个 CI 重叠且无 final test；既不能声称 clustered 方法泛化，也不能宣称 S0 普遍最优。
12. **KG 成本与状态。** KG 尚未实现/运行，其 inner-loop overhead、数值稳定性与收益未知。论文不能为了叙事把 KG 写成当前方法或完成 baseline。
13. **文献与创新性。** 当前仅核实 D-VAE 和 TED 两组直接文献，joint NAS+HPO、BO initialization、multi-fidelity 和 KG 覆盖不足；C012 的创新性边界尚未确定。
14. **尚未完成的机制验证。** early/cost-aware curves、coverage/diversity、LF promotion quality 与 matched ablation 都在等待分析或新实验，当前只能写条件式解释。

当前可写：上述真实限制及其证据边界。当前禁止：用 limitations 段落暗示尚未获得的正向结果，或把“未证明改善”写成“证明无效”。
