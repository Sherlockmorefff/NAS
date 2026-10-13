---
last_updated: 2026-08-14
status: candidate-index
claims: [C001, C002, C004, C005, C006, C007, C009, C010]
evidence: [E001, E002, E003]
literature: []
---

# Paper figure candidates

本目录只登记准备进入论文的图，不复制或生成图像。完整项目图表清单及现有脚本位于 [`../../results/figures/README.md`](../../results/figures/README.md)。每张图必须从 E-card 链接的 JSON/CSV 生成，记录脚本 identity，并在 caption 中披露 metric tier、seed、CI、cost axis 和 protocol caveat。

| Figure ID | 图 | RQ / Claims | 横轴 / 纵轴 | 分组 | 数据来源 | 生成脚本 | 当前状态 | 预期位置 | 误导风险 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F01 | Best-so-far validation | RQ1 / C001/C007 | full evaluations / best search validation | dataset × method；5 search seeds | E001/E002/E003 histories | [`../../../analyse/analyze_initialization_strategies.py`](../../../analyse/analyze_initialization_strategies.py)，需先审计口径 | [实验进行中] | Experiments | 把 full-only 曲线称为等成本；把 search validation 称 test |
| F02 | Cost-aware sample efficiency | RQ1/RQ6 / C001/C009 | equivalent-full 或 wall-clock / best search validation | method × dataset | histories + budget summaries + elapsed | TODO verified script | [计划] | Main/Experiments | LF epochs/150 当真实时间；跨硬件 wall-clock 排名 |
| F03 | Cross-dataset paired difference | RQ4 / C006/C007/C010 | method−S0 / search validation difference | paired dataset/search seed | E003 | TODO verified script | [计划] | Experiments | CI/seed pairing缺失；误标 final test |
| F04 | Initialization coverage/diversity | RQ2 / C004 | full budget or init stage / preregistered coverage metric | S0/G100/G150/K6 panels | candidate/cluster/fingerprint records | TODO | [未知] | Analysis | 事后挑 metric；把 diversity 当 quality |
| F05 | Surrogate calibration and ranking | RQ2 / C002 | observed/predicted or budget / calibration/ranking metric | dataset × method | E003 GP CSV/run summaries | [`../../../analyse/diagnose_bo_gp_surrogate.py`](../../../analyse/diagnose_bo_gp_surrogate.py)、[`../../../analyse/plot_gp_prediction_records.py`](../../../analyse/plot_gp_prediction_records.py)，需审计 | [实现数据可用] | Experiments/Supplement | 只展示 favorable metric/window；相关性写成因果 |
| F06 | LF versus full fidelity | RQ2 / C005 | LF metric / matched full metric | method × dataset/seed | E001/E002 LF/promotion records | TODO | [计划] | Analysis | selection bias；只画 promoted candidates |
| F07 | Computational cost | RQ6 / C009 | method / LF/full/acquisition/wall-clock cost | dataset × method | E001/E002/E003 | TODO | [部分数据可用] | Experiments | 堆叠不同单位；旧硬件时间泛化 |
| F08 | Evidence coverage matrix | RQ3/RQ4/RQ5 | evidence tier / dataset×method | complete/missing/current-gate-blocked | E001/E002/E003 + audits | TODO documentation graphic | [计划] | Setup/Limitations | 历史 complete 与当前 reproducible 合并 |

## 放行规则

- F01 必须与 F02 同时出现或在 caption 明确“full-only，不代表总成本”。
- F03 只画 paired search-validation difference；没有跨数据集 final artifact 时不得出现 test label。
- F04/F06 在 metric、comparison set 和 exclusions 预注册前不生成。
- F05 报告 error、coverage、ranking 和 top-region 指标组，保留 negative/mixed results。
- F08 状态沿用知识库证据状态，另用清楚文字区分 historical completion 与 current gate；不发明结果状态。

[未知] KG 无 artifact，因此不登记虚构 KG curve；待真实 E-card 建立后再新增 figure ID。
