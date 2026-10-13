最后更新日期：2026-08-13
当前状态：[计划] 已登记可追溯图源与优先图表；本轮不生成、复制或改写图片。

# 图表清单与生成约定

## 使用原则

[已验证] 论文图只能由原始 history、run summary、GP CSV/JSON 或 final-evaluation summary 生成。旧 PNG 可用于导航和视觉核对，但图中数值不得反向作为主表证据。所有新图应在 caption 中写明数据集、方法、search seeds、统计单位、成本横轴与 artifact root。

[已验证] 当前知识库不复制任何 PNG；历史图继续保留在 [legacy results](../../../legacy_artifacts/pre_20260813/results/) 中。分析入口包括 [analyze_initialization_strategies.py](../../../analyse/analyze_initialization_strategies.py)、[collect_experiment_results.py](../../../analyse/collect_experiment_results.py)、[diagnose_bo_gp_surrogate.py](../../../analyse/diagnose_bo_gp_surrogate.py) 和 [plot_gp_prediction_records.py](../../../analyse/plot_gp_prediction_records.py)。脚本输出需先复核是否保持本知识库统计口径。

## 最高优先级图

| ID | 图 | 原始输入 | 聚合/横轴 | 必须披露 |
| --- | --- | --- | --- | --- |
| F1 | Best-so-far validation | Cora 与跨数据集 `history_final.json` | 5 search seeds；full evaluations | mean、95% t CI、invalid 处理 |
| F2 | Cost-aware best-so-far | history + `budget_summary` + elapsed | equivalent-full 与 wall-clock | LF epochs/150 的换算、硬件 caveat |
| F3 | Surrogate calibration/ranking | GP prediction CSV、run summary | dataset × method × seed | MAE/RMSE、coverage、Spearman、top-region MAE |
| F4 | 跨数据集方法差值 | 60-run histories | `method - S0` paired by dataset/seed | 只标 search validation，不标 test |
| F5 | 实验证据覆盖矩阵 | run/final summaries | dataset × method × evidence tier | complete/missing/blocked 分离 |

### F1/F2：初始化效率

[计划] Cora 主图只比较 S0/G100/G150；K6-LF 用独立 panel 或线型，并注明其独立协议。full-only 横轴回答“相同 full budget”，equivalent-full 横轴回答“计入 low-fidelity epochs”；两者缺一不可。

[计划] 建议 budget checkpoints 为 50、100、150、200、250、300 full evaluations，并补充 AUC 与 evaluations-to-target。阈值需在看 test 前固定。

### F3：surrogate quality

[计划] calibration panel 绘 predictive mean versus observed validation，并给 95% interval coverage；ranking panel 绘 rolling Spearman/top-region MAE。不要把单个最后窗口的均值画成全程表现。

### F4：跨数据集差值

[计划] 在同一 dataset/search seed 内计算 G100−S0、G150−S0，以 paired dots/interval 展示。跨数据集 accuracy scale 不同，优先画差值而不是把原始 accuracy 混在单轴。

### F5：证据覆盖

[计划] 格子状态只使用：search complete、final complete、missing、current gate blocked。历史完成与当前 HEAD gate 必须显示为不同层，避免把 Flickr 历史结果误解为当前重新放行。

## 现有图资产登记

| 资产族 | 示例路径 | 用途限制 |
| --- | --- | --- |
| GP 预测/误差 | [gp scratch global4](../../../legacy_artifacts/pre_20260813/results/bo_phase4_gp_scratch_global4/) | 旧探索图，不直接代表 formal matrix |
| 独立 seed GP 图 | [independent seed0](../../../legacy_artifacts/pre_20260813/results/bo_phase4_gp_scratch_global4_independent_seed0/) | 可核对绘图样式，需重新聚合 formal seeds |
| Prequential v2 图 | [prequential seed0](../../../legacy_artifacts/pre_20260813/results/bo_phase4_gp_scratch_global4_prequential_v2_seed0/) | 单 seed 导航，不作为方法级结论 |

[已验证] 常见文件名包括 `convergence_phase4_bo_global4.png`、`gp_pred_vs_actual.png`、`gp_prediction_interval_by_step.png`、`gp_abs_error_by_step.png` 和 `gp_time_cost.png`。它们不在本目录重复保存。

## Caption 检查表

- [计划] 写明 search seed 数和 CI 算法。
- [计划] 写明 search validation 或 final test，不用泛称 accuracy。
- [计划] 写明 final candidate 的 validation-only selection。
- [计划] 写明 150/300 epochs 与 patience；K6 panel 单独标注。
- [计划] 写明 historical artifact 的旧 commit/dirty provenance。
- [计划] Flickr 图同时标注“历史 search complete”和“当前 resource gate failed”。

## 尚不生成的图

[未知] KG 无 artifact，因此不预留虚构曲线。[未知] 跨数据集 final test 缺失，因此不生成 test generalization 图。[计划] 待这些证据落盘后再登记具体 artifact 与 caption。
