---
last_updated: 2026-08-14
status: living-draft
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011]
evidence: [E001, E002, E003]
literature: []
---

最后更新日期：2026-08-13
当前状态：[已实现未充分验证] Cora search/final 与四数据集 search 已完成；跨数据集 final test、KG 和统一 K6 对照仍缺失。

# Experiments（论文底稿）

## Experimental questions

[计划] 实验围绕四个问题展开：初始化是否改善早期覆盖/效率、是否改善 GP surrogate、是否改善最终 validation/test、是否降低跨 seed 方差。判定标准见 [研究问题](../research_question.md)。

## Common setup

[已验证] S0/G100/G150 共享 `global4` 16D 搜索域、同一正式 checkpoint、768-candidate pool、scratch Exact GP、fixed-domain normalization 和 qLogEI。每次 search 使用 300 个 full evaluations；正式 search training 为 150 epochs、patience 40。额外 low-fidelity 成本另计。

[已验证] search 方法统计单位是独立 search seed，样本 SD 使用 `ddof=1`，95% CI 使用 df=4 的 t 区间。final candidate 内的 30 training replicates 描述候选训练噪声；方法级推断单位仍是 5 个 search seeds。

## Evidence block A: Cora search

[已验证] S0/G100/G150 各完成 search seeds 0–4，每个 history 有 300 次 full evaluations且无 invalid。300-full 最佳 validation 均值分别为 .8196、.8204、.8192；对应 95% CI 为 [.8148,.8244]、[.8147,.8261]、[.8125,.8259]。

[已验证] 这些区间高度重叠。本证据支持“固定预算下结果接近”的描述，不支持初始化方法已有确定优势。early-budget AUC 与 equivalent-cost 曲线仍需按 [图表清单](../results/figures/README.md) 统一生成后再写机制结论。

## Evidence block B: Cora final evaluation

[已验证] S0/G100/G150 每个 search seed 的 Top-10 候选都完成 30 replicates、300 epochs、patience 80 的 final evaluation。每个 seed 按 final validation mean 选中候选后，跨 search seeds 的 test mean 为：S0 .80802、G100 .80480、G150 .80482；95% CI 分别为 [.80571,.81033]、[.79483,.81477]、[.79589,.81375]。

[已验证] test 未参与候选选择。当前结果不支持 G100/G150 改善 Cora final test 的主张；如果论文讨论差异，应同时给出宽区间与 G100/G150 较大的 seed 方差。

## Evidence block C: K6-LF independent protocol

[已验证] K6-LF 完成 seeds 0–4 的 5 个 300-full searches，search-best validation 为 .81640 ± .00261。Top-7、30 replicates、150 epochs、patience 40 的 final evaluation 全部完成；validation-selected test mean 为 .80551 ± .00518，95% CI [.79908,.81195]。

[已验证] K6 的 Top-K、epochs 与 patience 不同，故独立分表；本结果不能用于无条件宣称 K6 优于或劣于主 Cora 三方法。

## Evidence block D: cross-dataset historical search

[已验证] CiteSeer、PubMed、DBLP、Flickr × S0/G100/G150 × search seeds 5–9 的 60-run 历史矩阵完整，每个 history 为 300 full。18,000 条记录中有 3 条 invalid。四个数据集的 search-validation 均值均由 S0 最高，但多个 CI 重叠。

[已验证] 该矩阵来自旧 commit 与 dirty-worktree provenance 的历史正式 artifact，而非当前 HEAD 原样运行。最新资源预检与授权表另有冲突，详见 [跨数据集实验](../experiments/cross_dataset.md)。

[未知] 未找到匹配的跨数据集 final-test artifact，因此本节不得使用“test generalization”措辞。

## Surrogate and cost analysis

[已验证] 每个 run summary 提供 prequential window MAE/RMSE、95% coverage、Spearman 与 top-region MAE。不同指标没有给出统一支配关系，例如某方法可有较低 MAE但较差 rank correlation；论文应报告完整指标组而非挑选单项。

[已验证] G100/G150 的 low-fidelity shortlist 随 search seed 变化：Cora 为 207–356 个（27.60–47.47 equivalent-full），历史跨数据集为 259–400 个（34.53–53.33 equivalent-full）；K6 固定 200×20，即 26.67 equivalent-full。图表必须从各 run summary 读取实际 epochs，并同时使用 full-only、equivalent-full 和 wall-clock 视角。

## Tables and figures

[已验证] 可直接进入论文的审计表位于 [results/tables.md](../results/tables.md)，包含 artifact 完整度、预算、Cora search/final、K6、跨数据集 search、surrogate、invalid 与成本。任何转写必须保留表下注释中的统计单位与协议 caveat。

[计划] 优先图：best-so-far 曲线、equivalent-cost 效率、surrogate calibration/ranking、跨数据集方法差值、实验覆盖矩阵。当前知识库只登记来源，不复制或新生成图片。

## Missing experiments

- [未知] KG 未实现，不能列入主结果。
- [未知] 跨数据集 final evaluation 缺失。
- [未知] K6 与主 Cora 方法没有完全一致的 final-evaluation 协议。
- [计划] 逐因素 ablation 需要固定 K、clustering feature、selector、shortlist 和预算后执行。
- [计划] Flickr 必须先解决当前资源 gate；不能用历史完成状态代替重新放行。

## Result-language guardrails

[已验证] search validation 不写成 test；candidate replicate 不写成独立 search；历史 artifact 不写成当前 HEAD reproduction；未完成实验不写入 conclusion。Introduction 中的贡献在相应判定标准满足前保持条件式。

## RQ traceability and decision criteria

| RQ | Claims | Experiment document | Evidence | Result/table source | 当前状态 | 判定标准 |
| --- | --- | --- | --- | --- | --- | --- |
| RQ1：初始化是否改善 sample efficiency？ | C001/C009 | [`../experiments/initialization.md`](../experiments/initialization.md) | [E001](evidence/E001.md)、[E002](evidence/E002.md)、[E003](evidence/E003.md) | [`../results/tables.md`](../results/tables.md)、[`tables/sample_efficiency.md`](tables/sample_efficiency.md) | [实验进行中] | 预注册 AUC/固定 budget/evaluations-to-target 在 paired seeds 上改善，且 equivalent-full 或 wall-clock 视角不消失 |
| RQ2：改善来自 coverage、候选质量还是 surrogate quality？ | C002/C004/C005 | [`../experiments/initialization.md`](../experiments/initialization.md) | E001/E002/E003 | [`tables/surrogate_diagnostics.md`](tables/surrogate_diagnostics.md)、[`tables/ablation.md`](tables/ablation.md) | [未知]（surrogate 描述已核实） | coverage/LF–full/promotion/全程 prequential 指标与 matched ablation 形成一致证据链 |
| RQ3：是否转化为最终 validation/test？ | C003/C007/C011 | initialization experiment | E001；E002 仅独立协议 | [`tables/main_results.md`](tables/main_results.md) | [已验证] 当前不支持 G100/G150 improvement | 同一 final protocol、validation-only selection、以 search seed 为单位的 paired effect/CI |
| RQ4：是否跨数据集稳定？ | C006/C010 | [`../experiments/cross_dataset.md`](../experiments/cross_dataset.md) | E003 | main results / cross-dataset candidate figure | [已验证] search tier 不支持稳定 improvement；final test [未知] | 同协议多数据集 search + 匹配 final eval；逐数据集与总体 negative results 全披露 |
| RQ5：KG 相比 qLogEI 是否有额外收益？ | C008 | [`../experiments/KG.md`](../experiments/KG.md) | [WAITING-FOR-E005] | [TBD-NUMBER] | [未知] | 同初始化/GP/budget/seed/final protocol；主要指标、effect size、overhead、failure rate |
| RQ6：成本和适用边界是什么？ | C009/C010 | initialization + cross-dataset | E001/E002/E003 | [`tables/computational_cost.md`](tables/computational_cost.md)、[`../limitations.md`](limitations.md) | [已实现未充分验证] | full、LF equivalent、wall-clock/acquisition overhead、failure/blocked 和协议边界同时报告 |

## Explicit placeholders

- RQ1 early curves: `[WAITING-FOR-E004]`，主要结论 `[TBD-NUMBER]`。
- RQ2 coverage/LF matched mechanism: `[UNSUPPORTED-CLAIM]`，不得由 selector 实现反推。
- RQ4 cross-dataset final test: `[WAITING-FOR-E006]`。
- RQ5 KG: `[WAITING-FOR-E005]`；没有 artifact 前表格保持空白/TODO。

当前可写：E001–E003 的描述性结果与不支持边界。当前禁止：无 paired effect 的显著性、跨数据集 test、KG comparison、coverage 或 LF quality 的因果解释。
