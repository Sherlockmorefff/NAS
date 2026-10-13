最后更新日期：2026-08-13
当前状态：[未知] 仓库内未确认 Knowledge Gradient 实现、正式配置或实验结果；本页仅定义可执行对照设计。

# Knowledge Gradient 对照实验

## 现状边界

[已验证] 正式在线 acquisition 是 qLogEI，入口位于 [bo_phase4.py](../../bo_phase4.py)，surrogate 是 [scratch Exact GP](../../surrogate/accuracy_gp.py)。随机 acquisition 基线也已实现。DKL GP 有实现与测试，但不属于 S0/G100/G150 的正式配置。

[未知] 全仓检索没有发现 KG acquisition 的实现、配置或结果。`tests/test_experiment_paths.py` 中含 `kg` 的字符串只是协议路径格式夹具，不能作为算法证据。因此本页不得被引用为“KG 已实现”或“KG 已优于 qLogEI”。

## 最小充分对照

[计划] 保持以下项目完全相同，只替换在线 acquisition：

| 控制项 | 固定值/来源 |
| --- | --- |
| 数据集与 split | 首先 Cora；复用正式 split identity |
| checkpoint 与搜索空间 | 同一 `global4` checkpoint、12D architecture + 4D direct HP coordinates |
| 初始化 | 首选 S0；若资源允许再对 G100 重复 |
| full 预算 | 50 seed + 250 online = 300 |
| candidate/evaluation seeds | 现有 fingerprint-based 固定方案 |
| surrogate | scratch Exact GP、fixed-domain normalization、相同 refit schedule |
| acquisition optimization | 相同 candidate domain、restarts、raw samples 与约束 |
| final evaluation | 同一 Top-K、replicate 数、epochs、patience 和 validation-based selection |

[计划] 比较臂至少为 qLogEI、KG 和 random；如果 KG 实现需要 fantasies 或额外 inner optimization，应把 acquisition wall time、GPU/CPU 时间和失败率单列，不能只比较 GNN evaluation 数。

## 指标与判定标准

- [计划] 主要指标：best validation 的 full-evaluation AUC、达到预设阈值的 evaluations-to-target、300-full 最终 validation。
- [计划] 次要指标：prequential MAE/RMSE、95% coverage、Spearman、top-region MAE、acquisition overhead。
- [计划] final-test 指标：每个 search seed 先按 final validation 选候选，再以 search seed 为统计单位。
- [计划] 支持 KG 的最低标准：在预注册主要指标上跨 seeds 改善，t-based 95% CI 与效应量同时报告，且计算成本增量可接受。
- [计划] 否定或停止标准：无稳定早期效率增益、KG overhead 抵消节省、数值失败率明显增加，或只在 test-oracle 选择下显得更好。

## 实施与证据 TODO

- [计划] TODO：增加 KG acquisition adapter 和 CPU-safe 单元测试；错误时清晰失败，不回退为 qLogEI。
- [计划] TODO：冻结实验配置与 seed 清单，生成独立 protocol ID。
- [计划] TODO：先做 CPU 合成目标 smoke test，再在远端 GPU 做小预算 pilot。
- [计划] TODO：只有 artifact 包含配置、history、run summary、成本和 final evaluation 后，才更新论文结果。

## 风险

[假设] KG 的 inner-loop 计算可能显著高于 qLogEI；这是待测假设，不是当前结果。[未知] 当前代码栈下 batch/noisy KG 的具体可用 API 和数值稳定性尚未验证。
