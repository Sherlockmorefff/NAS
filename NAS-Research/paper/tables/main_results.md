---
last_updated: 2026-08-14
status: partially-eligible
claims: [C003, C006, C007, C011]
evidence: [E001, E002, E003]
literature: []
---

# T01 — Main results candidate

- 研究问题：RQ3 最终 validation/test；RQ4 跨数据集稳定性。
- 数据来源：[`../../results/tables.md`](../../results/tables.md)；E001/E002/E003 及各卡 Provenance。
- 统计单位：每方法 5 个独立 search seeds；`mean ± sample SD`，95% CI 为 df=4 t interval。30 final replicates 不是独立 searches。
- 当前完整度：Panel A/B/C 可用；跨数据集 final-test panel 缺失。
- 进入论文：可以，但三类协议必须分 panel，不能合并排名。

## Panel A：Cora 主协议

| Method | 300-full search validation mean ± SD | Validation-selected final test mean ± SD | Evidence |
| --- | ---: | ---: | --- |
| S0 | .8196 ± .0038 | .80802 ± .00186 | [E001](../evidence/E001.md) |
| G100 | .8204 ± .0046 | .80480 ± .00803 | [E001](../evidence/E001.md) |
| G150 | .8192 ± .0054 | .80482 ± .00719 | [E001](../evidence/E001.md) |

表注：[已验证] Search 为 150 epochs/patience 40；final 为 Top-10×30、300 epochs/patience 80，并按 final validation 选候选。精确 CI 和 seed values 保留在事实源。

## Panel B：K6 独立协议

| Method | Search validation mean ± SD | Validation-selected final test mean ± SD | Evidence |
| --- | ---: | ---: | --- |
| K6-LF | .81640 ± .00261 | .80551 ± .00518 | [E002](../evidence/E002.md) |

表注：[已验证] K6 final 为 Top-7×30、150 epochs/patience 40；不得与 Panel A 作无条件 test 排名。

## Panel C：四数据集 historical search validation

| Dataset | S0 mean | G100 mean | G150 mean | Evidence tier |
| --- | ---: | ---: | ---: | --- |
| CiteSeer | .72920 | .72400 | .71400 | [E003 search validation](../evidence/E003.md) |
| PubMed | .82040 | .81280 | .81480 | [E003 search validation](../evidence/E003.md) |
| DBLP | .80680 | .80320 | .80080 | [E003 search validation](../evidence/E003.md) |
| Flickr | .522257 | .520375 | .519747 | [E003 search validation](../evidence/E003.md) |

表注：[已验证] 仅为 historical search；每格 5 seeds 5–9。完整 SD/CI/min–max 在项目级事实源。0/60 匹配 final evaluations 已确认，因此不能标为 test。

- 当前能支持：Cora 未显示 G100/G150 稳定 final-test improvement；historical search 中 S0 mean 在四数据集更高；Cora validation-only selection 已核实。
- 当前不能支持：统计等价、显著优越、跨数据集 final-test 泛化、K6 直接排名。
