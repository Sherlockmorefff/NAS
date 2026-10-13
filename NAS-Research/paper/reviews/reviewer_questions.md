---
last_updated: 2026-08-14
status: active-review-register
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

# ICLR reviewer questions

风险等级用于论文准备优先级，不是新 evidence 状态。

| ID | Reviewer question | 当前回答 | 支持证据 | 缺失证据 / 对应实验 | 风险 |
| --- | --- | --- | --- | --- | --- |
| Q01 | 相对普通 joint NAS+HPO 的创新性是什么？ | 当前只能说研究结构化初始化的成本感知、分层证据评估；不能声称 first/new method | D-VAE/TED cards；C012 | direct related work audit；joint NAS+HPO literature | 高 |
| Q02 | 为什么需要该初始化方法？ | 设计动机是覆盖与早期信息质量，但现有终点/final test 未显示普遍改善 | E001–E003 | C001/C004/C005 early/coverage/LF analyses | 高 |
| Q03 | 提升是否只是额外 LF 预算造成？ | 现有设计不能排除；full budget 相同但总成本不同 | E001/E002/E003、C009 | equivalent-full/wall-clock curves；matched no-LF ablation | 高 |
| Q04 | 方法比较是否公平？ | candidate domain、full budget、GP/qLogEI/search training 对主方法一致；LF cost 不同，K6 final protocol 不一致 | paper contract、E-cards | matched total-cost comparison；统一 K6 protocol | 高 |
| Q05 | 机制解释有直接证据吗？ | 没有；surrogate 指标混合，coverage/LF quality 未分析 | E003、C002/C004/C005 | coverage metrics、LF–full correlation、逐因素 ablation | 高 |
| Q06 | 统计可靠性如何？ | 5 search seeds，已报 sample SD/t-CI；不能由 CI overlap 判断等价 | E001/E003 | paired effect、practical threshold、power/multiplicity | 高 |
| Q07 | 结论是否跨数据集泛化？ | historical search 中无 G100/G150 稳定 improvement；没有 matching final test | E003 | cross-dataset final evaluation | 高 |
| Q08 | 是否有 test-selection bias？ | Cora final candidate 按 final validation 选，之后报告 test；不声称消除所有 researcher bias | E001/E002、C011 | 跨数据集 final protocol；analysis preregistration | 中 |
| Q09 | 为什么使用 S0、G100、G150 这些 baselines？ | S0 隔离无 LF 的 global selector；G100/G150 比较 promotion/online allocation，但并非逐因素 ablation | config、initialization experiment | random/standard initialization 文献定位；matched ablation | 高 |
| Q10 | 为什么 KG 值得加入？ | 目前只有对照设计；是否值得取决于主线和 overhead，不是已完成贡献 | KG experiment、C008 | implementation、synthetic smoke、matched GPU experiments | 中 |
| Q11 | KG 若没有收益怎么办？ | 报告 negative result、overhead 和 failure rate，不因论文叙事隐藏 | KG stop criteria | real artifact | 低（当前非主线） |
| Q12 | 计算成本是否可复现？ | full/LF work units 可追溯；historical wall time 仅作 provenance 描述 | T05、E001–E003 | same-hardware GPU-hours、selector/acquisition overhead | 高 |
| Q13 | Historical dirty worktree 是否可信？ | 每 run 保存 runtime source manifest、commit/status hashes；可审计但不等同 clean-commit reproduction | E002/E003 provenance | current HEAD reproduction or source snapshot verification | 中 |
| Q14 | 为什么 S0 historical mean 都更高仍主推 clustered 方法？ | 当前不主推 superiority；story 转为边界/negative empirical evaluation，是否成立取决于 early/cost evidence | E003、story | early paired curves、literature novelty | 高 |
| Q15 | 失败/blocked runs 是否 cherry-picked？ | 18,000 records 的 3 invalid 保留；Flickr historical complete 与 current OOM 同时报告 | E003/evidence audit | preregistered failure handling for new runs | 中 |

## 当前回答优先级

先解决 Q02/Q03/Q05（同一批 early/cost/mechanism analysis），再解决 Q06/Q07（统计与 cross-dataset final），最后用文献审计回答 Q01/Q09。KG 只有在主线仍需 acquisition 对照时进入执行优先级。
