---
last_updated: 2026-08-14
status: current-audit
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: []
---

# Evidence audit

## Claim → Evidence coverage

| Claim | 覆盖 | 结论/风险 |
| --- | --- | --- |
| C001 sample efficiency | E001–E003 histories | [实验进行中] 缺 early/cost-aware paired analysis |
| C002 surrogate quality | E003 final-window diagnostics | [已验证] 指标混合，不支持一致 improvement；缺 trajectory |
| C003 final performance | E001；E002 独立协议 | [已验证] Cora 不支持 G100/G150 improvement |
| C004 coverage | selector/config only | [未知] 无 coverage metric |
| C005 LF candidate quality | LF/promotion artifacts exist | [未知] 无 LF–full/matched no-LF analysis |
| C006 cross-dataset stability | E003 search only | [已验证] search tier 不支持稳定 improvement；final test 缺失 |
| C007 statistics/effect | E001/E003 mean/SD/t-CI | [未知] 无 paired effect、practical threshold、multiplicity |
| C008 KG | none | [未知] 无实现/配置/result artifact |
| C009 cost | E001–E003 budget summaries | [已验证] full budget 相同、LF 额外成本；完整 overhead 未分析 |
| C010 dependency/boundary | E001/E003 | [已验证] 限 dataset/search-seed 异质性；搜索空间外推未知 |
| C011 validation-only final selection | E001/E002 final summaries | [已验证] 限 Cora；不能说排除所有 bias |
| C012 novelty | D-VAE/TED only | [未知] 文献覆盖不足 |

## 无来源数字

- 本轮候选正文/表格中的实际结果数字均链接 [`../../results/tables.md`](../../results/tables.md) 和 E001–E003；未发现新增的独立计算数字。
- `[TBD-NUMBER]`、`[TBD-DATASET]` 和 `[WAITING-FOR-E###]` 是显式缺口，不是可投稿数字。
- Cora 主 search 的运行 Git commit/protocol ID 无来源，已保持 `null/TODO`；不得用当前 HEAD 补写。

## Incomplete / missing / blocked

- [实验进行中] early-budget、equivalent-full、wall-clock sample-efficiency 与 coverage 分析。
- [未知] 四数据集 60-run matrix 的匹配 final evaluation（0/60 已确认）。
- [未知] KG implementation/config/results。
- [计划] matched mechanism ablation、dataset statistics extraction、paired effect-size analysis。
- [已验证] Flickr historical search complete；current formal seeds latest resource preflight/retry OOM。二者不可合并。

## 统计缺口

- 未登记 paired mean difference/CI、standardized or raw effect size、practical significance threshold、power/sensitivity。
- 未登记 dataset × method × metric × budget checkpoint 的 multiplicity plan。
- Surrogate 表只有最终窗口；缺全程/rolling aggregation 和 paired uncertainty。
- Coverage、LF–full correlation、promotion precision、evaluations-to-target threshold 均未预注册。

## Validation/test 风险

- Cora final summary 记录 validation-only candidate selection，当前未发现 test 参与选择的直接证据。
- 风险仍包括：看 test 后挑 method/metric/threshold、把 30 replicates 当独立 searches、把 search validation 写成 final test。
- Cross-dataset 完全没有 matching final test；K6 final protocol 与主 Cora 不匹配。

## 当前投稿阻塞项

1. C001/C004/C005 的主要机制与成本感知分析未完成。
2. C006/C007 缺跨数据集 final evaluation 与 formal paired effect-size/multiplicity。
3. C012 的直接相关文献审计不足，创新性定位无法放行。

审计结论：[已实现未充分验证] Paper Agent 的追溯链已建立，但现有 evidence 更适合谨慎 empirical/negative-result 叙事，尚不足以放行强性能贡献。
