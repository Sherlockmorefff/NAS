---
last_updated: 2026-08-14
status: draft-contract
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

# Paper contract

## 投稿目标与核心问题

- 投稿目标：ICLR。
- 核心问题：[假设] 在固定联合 GNN NAS+HPO 候选域中，结构化、low-fidelity-assisted 初始化是否以可接受的总成本改善覆盖、surrogate 学习、BO 样本效率和 validation-only selection 后的最终 test 表现？
- 暂定主线：[已实现未充分验证] 论文对多种初始化进行成本感知、分层证据评估；现有结果更支持“效果依赖数据集、指标和评估层次”，不支持普遍 accuracy improvement。

## 条件式潜在贡献

1. [已验证] 可审计的 12D architecture D-VAE latent + 4D direct HP coordinates 联合搜索与 seed/provenance 管线。
2. [已实现未充分验证] 对 global Schur、K=4 clustered TED + low fidelity 和 K=6 clustered Schur + low fidelity 的同域研究；性能贡献取决于 Claim 放行。
3. [已实现未充分验证] 同时核算 full evaluations、low-fidelity equivalent-full 和 wall-clock 的初始化评估设计；统一曲线尚未完成。
4. [已验证] Cora 协议中按 final validation 选择候选，之后才读取 test；统计单位区分 search seed 与 training replicate。

## 明确不声称

- 不声称单数据集现象是普遍规律，不把小幅均值差写成显著提升。
- 不声称“已实现”等于“有效”，不把相关性写成初始化机制的因果效果。
- 不把 search validation、final validation 和 final test 混称为 accuracy。
- 不声称 G100/G150 与 S0 总计算成本相同；它们只有 full-evaluation budget 相同。
- 不声称四数据集已有 final-test 泛化证据，不声称 KG 已实现、已运行或优于 qLogEI。
- 不声称 K6 与 S0/G100/G150 的 final test 可直接排名，因为 Top-K、epochs 和 patience 不同。

## 固定实验协议

正式 S0/G100/G150 配置以 [`../../configs/cross_dataset_methods.json`](../../configs/cross_dataset_methods.json) 和原 run provenance 为准：`global4` 16D 搜索坐标、768-candidate pool、scratch Exact GP、fixed-domain normalization、qLogEI、search 150 epochs / patience 40、300 full evaluations。S0 为 50 seed + 250 online；G100 为 50 seed + 100 promoted + 150 online；G150 为 50 seed + 150 promoted + 100 online。G100/G150 另有实际数量随 seed 变化的 20-epoch low-fidelity shortlist。

Cora 主 final evaluation 固定为 Top-10、30 replicates、300 epochs、patience 80；K6 独立协议为 Top-7、30 replicates、150 epochs、patience 40。两者必须分层报告。历史跨数据集 search seeds 是 5–9；当前配置还同时保存 development seeds 0–4，不能混用。

## 可以改变的条件

- [计划] 图表样式、caption、正文组织与不改变统计含义的展示精度。
- [计划] 在预先记录后增加 paired comparison、effect size、confidence interval 或成本分析脚本。
- [计划] 新增 KG、ablation 或跨数据集 final evaluation，但必须分配新 protocol identity，保留独立结果，不回写历史协议。

## 不得随意改变的条件

- dataset/split/checkpoint/source identity、候选池与 fingerprint、search/training/evaluation seed 派生规则。
- 方法间候选域、full budget、training epochs/patience、surrogate/refit/acquisition 参数和失败处理。
- final candidate 的 validation-only selection；test 不得用于选择、tie-breaking、retry 或 seed 派生。
- 已查看 test 后不得修改主要指标、阈值、方法筛选或汇报 seeds；任何变更必须新建协议并披露。

## Claim 最低证据

| 主张类型 | 最低放行证据 |
| --- | --- |
| sample efficiency | 至少 5 个独立 search seeds；预先指定的 AUC/固定预算点/evaluations-to-target；同时报告 full-only、equivalent-full、wall-clock；paired effect size 与不确定性 |
| surrogate improvement | prequential error、95% coverage、ranking 与 top-region 指标方向形成一致证据链；不能只取最后窗口或单数据集 |
| final performance | 每个 search seed 内 validation-only selection；跨 search seeds 汇总 test；匹配 Top-K、replicates、epochs、patience；报告 effect size 和 CI |
| cross-dataset stability | 多数据集同协议 search，并有匹配 final evaluation；逐数据集和总体方向均披露 negative results |
| mechanism/coverage | 可复现 coverage 指标以及逐因素 ablation，固定 K、clustering feature、selector、shortlist、promotion 与预算 |
| KG benefit | 与 qLogEI/random 的同初始化、同 surrogate、同预算、同 final-eval 对照，并报告 acquisition overhead 与失败率 |

## 停止扩展实验的条件

- 预注册主要 Claim 已达到上表最低证据，新增数据集/方法不会改变对应结论或消除主要 reviewer 风险。
- 连续、同协议结果显示 effect size 很小或方向不稳定，且置信区间已足以排除预先定义的实际重要改善；转向报告边界/negative result。
- 新方法只增加与核心问题无关的组合，不能隔离已有混杂或回答前三个投稿阻塞项。
- 资源成本超过预设上限或协议必须改变；记录 blocked，停止将其并入原比较。

## 当前投稿阻塞项

1. [实验进行中] 缺少统一的 early-budget、cost-aware sample-efficiency 与 coverage 分析（C001/C004/C005）。
2. [未知] 四数据集没有匹配的 final-evaluation artifact，不能形成跨数据集 test 结论（C006）。
3. [未知] 缺少预注册 paired comparison、effect size 和多重比较处理（C007）。
4. [未知] 相关工作资产不足，无法可靠界定相对 joint NAS+HPO 的创新性（C012）。
5. [未知] KG 没有实现、正式协议或 artifact（C008）；除非主线需要，不作为当前交付阻塞的最高优先级。
