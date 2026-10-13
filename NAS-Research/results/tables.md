最后更新日期：2026-08-13
当前状态：[已验证] 数值由原始 JSON/CSV 重新解析；跨数据集 final test 与 KG 仍缺失。

# 结果表与统计口径

## 统计口径

[已验证] search 方法比较的独立单位为 search seed。下表中的 `mean ± SD` 使用 5 个 seeds 的样本标准差（`ddof=1`）；95% CI 为双侧 t 区间，df=4，临界值 2.776。单候选 final-evaluation 的单位是 training replicate seed，但方法级 final test 必须先在每个 search seed 内按 final validation 选候选，再跨 5 个 search seeds 汇总；禁止把 150 个 replicate 当作 150 个独立搜索。

## 实验与 artifact 完整度

| 实验族 | search artifacts | final-eval artifacts | 可支持证据 |
| --- | --- | --- | --- |
| Cora S0/G100/G150 | 15/15 complete；每个 300 full | 15/15 complete；Top-10 × 30 | search validation 与各自协议内 final test |
| Cora K6-LF | 5/5 complete；每个 300 full | 5/5 complete；Top-7 × 30 | K6 独立协议内 search/final test |
| 四数据集 S0/G100/G150 | 60/60 complete；每个 300 full | 0/60 已确认 | search validation、surrogate、成本 |
| KG | 0 | 0 | 无结果结论 |

## 方法配置与预算

| 方法 | 初始化 full | LF 成本 | promoted full | qLogEI full | full 总预算 | 训练协议 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 50 | 0 | 0 | 250 | 300 | search 150 epochs / patience 40 |
| G100 | 50 | 每簇至多 100×20；按实际 epochs 核算 | 100 | 150 | 300 | search 150 / 40 |
| G150 | 50 | 每簇至多 100×20；按实际 epochs 核算 | 150 | 100 | 300 | search 150 / 40 |
| K6-LF | 50 | 200×20 epochs = 26.67 eq-full | 100 | 150 | 300 | search 150 / 40 |

证据：[正式配置](../../configs/cross_dataset_methods.json)、[Cora run summaries](../../legacy_artifacts/pre_20260813/results/)、[K6 run summaries](../../legacy_artifacts/pre_20260813/results/fixedk6_gmm_schur_lowfid_pilot_20260728_024642/)。

[已验证] G100/G150 的实际 LF 成本随 search seed 变化：Cora seeds 0–4 的候选数为 207/272/264/356/238，equivalent-full 均值 35.65、范围 27.60–47.47；历史跨数据集 seeds 5–9 为 360/311/338/259/400，均值 44.48、范围 34.53–53.33。同 seed 的 G100/G150 shortlist 相同，但 promoted full 数不同。

## Cora search validation

指标为每个 search seed 的 300 次 full evaluation 中最佳 `val_acc`。

| 方法 | seed values | mean ± SD | 95% CI | min–max |
| --- | --- | --- | --- | --- |
| S0 | .820, .818, .814, .822, .824 | .8196 ± .0038 | [.8148, .8244] | .814–.824 |
| G100 | .820, .826, .824, .816, .816 | .8204 ± .0046 | [.8147, .8261] | .816–.826 |
| G150 | .828, .814, .816, .820, .818 | .8192 ± .0054 | [.8125, .8259] | .814–.828 |

[已验证] 三种均值接近且区间高度重叠；当前表不支持“某初始化显著更优”的表述。

## Cora final test：S0/G100/G150 共同协议

[已验证] 每个 search seed 取 Top-10 search-validation 候选，每候选 30 replicates、300 epochs、patience 80；最终候选按 replicate 的 final-validation 均值选择，test 不参与选择。

| 方法 | 每个 search seed 选中候选的 test mean | 跨 search seeds mean ± SD | 95% CI |
| --- | --- | --- | --- |
| S0 | .80910, .80847, .80670, .80560, .81023 | .80802 ± .00186 | [.80571, .81033] |
| G100 | .80457, .79830, .80807, .79657, .81650 | .80480 ± .00803 | [.79483, .81477] |
| G150 | .79933, .80650, .80267, .79910, .81650 | .80482 ± .00719 | [.79589, .81375] |

证据：[Top-10 final summaries](../../legacy_artifacts/pre_20260813/results/final_eval_topk10_r30_seedfair_base20260724/runs/)。

## Cora K6-LF 独立 final-test 表

[已验证] K6 使用 Top-7、30 replicates、150 epochs、patience 40；因训练协议不同，本表不与上表作无条件排名。

| 指标 | seed values | mean ± SD | 95% CI |
| --- | --- | --- | --- |
| search best validation | .820, .814, .818, .814, .816 | .81640 ± .00261 | [.81316, .81964] |
| validation-selected candidate test mean | .80830, .80037, .80047, .81240, .80603 | .80551 ± .00518 | [.79908, .81195] |

证据：[K6 histories](../../legacy_artifacts/pre_20260813/results/fixedk6_gmm_schur_lowfid_pilot_20260728_024642/)、[K6 final summaries](../../legacy_artifacts/pre_20260813/results/fixedk6_top7_searchseeds0_4_final30_20260728/)。

## 四数据集历史 search validation

| 数据集 | 方法 | mean ± SD | 95% CI | min–max |
| --- | --- | --- | --- | --- |
| CiteSeer | S0 | .72920 ± .01404 | [.71176, .74664] | .712–.750 |
| CiteSeer | G100 | .72400 ± .01049 | [.71098, .73702] | .706–.732 |
| CiteSeer | G150 | .71400 ± .00583 | [.70676, .72124] | .706–.722 |
| PubMed | S0 | .82040 ± .00669 | [.81209, .82871] | .816–.832 |
| PubMed | G100 | .81280 ± .00363 | [.80829, .81731] | .808–.818 |
| PubMed | G150 | .81480 ± .00179 | [.81258, .81702] | .812–.816 |
| DBLP | S0 | .80680 ± .00179 | [.80458, .80902] | .804–.808 |
| DBLP | G100 | .80320 ± .00540 | [.79649, .80991] | .798–.812 |
| DBLP | G150 | .80080 ± .00110 | [.79944, .80216] | .800–.802 |
| Flickr | S0 | .522257 ± .002328 | [.519366, .525148] | .519989–.525233 |
| Flickr | G100 | .520375 ± .000709 | [.519494, .521255] | .519720–.521468 |
| Flickr | G150 | .519747 ± .001926 | [.517356, .522138] | .517614–.522409 |

[已验证] 以上均为 search validation，不是 test。证据：[60-run histories](../../legacy_artifacts/pre_20260813/results/formal_search_20260731_8831cd0a5337/)。

## Surrogate 指标

以下为每个 dataset × method 的 5 个 run-summary prequential 指标均值；window 指最后记录窗口，不能替代整条学习曲线。

| 数据集 | 方法 | MAE | RMSE | 95% coverage | Spearman | top-region MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CiteSeer | S0 | .042176 | .058737 | .820 | .196865 | .045061 |
| CiteSeer | G100 | .035994 | .044586 | .900 | .612647 | .030677 |
| CiteSeer | G150 | .062539 | .090272 | .780 | .086061 | .036188 |
| PubMed | S0 | .017735 | .021351 | .980 | .016660 | .014088 |
| PubMed | G100 | .037094 | .060991 | .820 | -.112071 | .020285 |
| PubMed | G150 | .031142 | .043243 | .900 | -.039400 | .016160 |
| DBLP | S0 | .027991 | .042871 | .900 | .437790 | .007881 |
| DBLP | G100 | .048714 | .090284 | .880 | .270750 | .013564 |
| DBLP | G150 | .032601 | .043067 | .980 | .123969 | .013237 |
| Flickr | S0 | .013904 | .019922 | .860 | .120055 | .006842 |
| Flickr | G100 | .015651 | .021793 | .860 | .003636 | .012643 |
| Flickr | G150 | .013744 | .018463 | .940 | .183030 | .008779 |

## 无效记录与运行成本

[已验证] 18,000 条历史 full-evaluation 记录中有 3 条 invalid：CiteSeer/S0/seed5 step253、CiteSeer/G100/seed5 step214、DBLP/S0/seed9 step143；其余 17,997 条有效。run 均完成固定 300 条历史，invalid 未被静默删除。

[已验证] 18,000 条记录的 `evaluation_stage` 构成为 `initial_seed=3,000`、`initial_expand=5,000`、`online_bo=10,000`；这与三种方法的 50/100/150/250 full-budget 分解一致。

| 数据集 | S0 五 seeds 总时长 | G100 五 seeds 总时长 | G150 五 seeds 总时长 |
| --- | ---: | ---: | ---: |
| CiteSeer | 1.434 h | 1.399 h | 1.200 h |
| PubMed | 2.325 h | 2.351 h | 2.094 h |
| DBLP | 2.476 h | 2.647 h | 2.354 h |
| Flickr | 18.829 h | 17.671 h | 15.944 h |

[已验证] elapsed 来自历史 run summary，受硬件、并发与实现版本影响；它是 provenance 成本描述，不是跨硬件 benchmark。

## 缺失实验与不可支持结论

| 缺口 | 当前状态 | 不可支持的结论 |
| --- | --- | --- |
| KG 实现/结果 | [未知] 无 artifact | KG 优于或劣于 qLogEI |
| 四数据集 final evaluation | [未知] 未找到匹配 artifact | 跨数据集 test 泛化 |
| Flickr 当前 gate | [已验证] 最新预检 OOM | 当前 HEAD 可按原协议重跑 Flickr formal matrix |
| K6 与主 Cora 同协议比较 | [未知] final epochs/Top-K 不一致 | K6 与 S0/G100/G150 的直接 test 排名 |
| 独立逐因素初始化 ablation | [计划] 尚未执行 | 将差异单独归因于 K、聚类空间或 selector |

## 复算说明

[已验证] 本页数值由 `history_final.json`、`run_summary.json` 与 `final_eval_summary_global4.json` 解析，不从旧分析 Markdown 转抄。生成论文表格时应保留同一路径、字段名、统计单位和选择规则。
