---
last_updated: 2026-08-14
status: active-registry
claims: [C001, C002, C003, C004, C005, C006, C007, C008, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

# Claim–Evidence 登记表

状态使用知识库统一证据状态；“不支持”是 Claim disposition，不是新的证据状态。被证据削弱的 Claim 保留在表中，不删除。`最后更新` 为人工登记日期，不代表 artifact 生成日期。

| Claim ID | 主张 | 类型 | 状态 | 支持证据 | 反对证据 | 缺失证据 | 允许表述 | 禁止表述 | 目标章节 | 最后更新 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C001 | 结构化初始化提高 BO 早期 sample efficiency | 性能/效率 | [实验进行中] | [E001](evidence/E001.md)、[E002](evidence/E002.md)、[E003](evidence/E003.md) 的 histories 可复算 | 300-full 终点未显示一致优势；LF 有额外成本 | 预注册 AUC、budget checkpoints、evaluations-to-target、equivalent-full 与 paired effect size | “我们正在检验初始化是否改变成本感知早期效率” | “结构化初始化已提高 sample efficiency” | Experiments | 2026-08-14 |
| C002 | G100/G150 一致改善 surrogate quality | 机制 | [已验证]；不支持强主张 | E003 中 CiteSeer/G100 的 MAE、coverage、Spearman 较好 | E003 其他数据集/指标方向混合，G150 也无一致支配 | 全程曲线、paired comparison、机制 ablation | “不同初始化产生不同 surrogate diagnostic profile，未见一致支配” | “clustered initialization 普遍改善 surrogate” | Experiments/Method | 2026-08-14 |
| C003 | G100/G150 改善最终 validation 或 test performance | 性能 | [已验证]；不支持 | E001 中 300-full validation 均值接近 | E001 中 validation-selected final-test 均值 S0 更高且区间重叠 | 更高 power 的同协议 seeds；预注册 effect size | “当前 Cora 证据不支持确定 test improvement” | “G100/G150 显著提高 final test” | Abstract/Experiments | 2026-08-14 |
| C004 | 结构感知选择改善联合搜索空间覆盖 | 机制 | [未知] | 方法实现记录显示分簇/Schur/TED selector 存在 | 无已登记 coverage 统计 | architecture/HP diversity、cluster coverage、fingerprint coverage 与 matched ablation | “设计目标是改善覆盖” | “已改善覆盖，并因此提高性能” | Method/Experiments | 2026-08-14 |
| C005 | low-fidelity screening 提高初始候选质量 | 机制 | [未知] | E001/E002 含 LF histories 和 promotion 记录 | 无 matched no-LF 对照 | LF–full 相关、promotion precision/regret、同预算 no-LF ablation | “LF screening 用于候选 promotion” | “LF screening 已提高候选质量” | Method/Experiments | 2026-08-14 |
| C006 | G100/G150 相比 S0 的改善跨数据集稳定 | 泛化 | [已验证]；不支持当前 search 强主张 | E003 为四数据集同协议 search | E003 中 S0 的 search-validation 均值在四个数据集均更高；final test 缺失 | 匹配的跨数据集 final evaluation、paired effect size | “历史 search validation 未显示稳定改善” | “方法实现跨数据集 test 泛化” | Abstract/Experiments | 2026-08-14 |
| C007 | 方法改善具有统计意义和实际 effect size | 统计 | [未知] | E001/E003 有 5-seed 均值、SD、t-CI | CI 大量重叠；尚无正式差值检验和 effect size | paired CI/test、effect size、multiplicity plan、practical threshold | “描述性差值及不确定性如下” | “显著提升”或“实质提升” | Experiments | 2026-08-14 |
| C008 | KG 相比 qLogEI 带来额外收益 | Acquisition | [未知] | 无 | 无 KG 实现、配置或 artifact | KG adapter、同协议 qLogEI/random 对照、overhead 和 final eval | “KG 是待检验对照” | “KG 已实现/已优于 qLogEI” | Future/Experiments | 2026-08-14 |
| C009 | LF 初始化不增加明显计算成本 | 成本 | [已验证]；不支持“不增加” | E001/E003 的 full budget 均为 300 | G100/G150 另有 27.60–53.33 equivalent-full 范围；E002 为 26.67 equivalent-full | 方法级 acquisition overhead 与可比硬件 wall-clock 分析 | “full budget 相同，但 LF 带来额外训练成本” | “所有方法同成本” | Method/Experiments/Limitations | 2026-08-14 |
| C010 | 结果依赖数据集、搜索空间或 seed | 边界 | [已验证]（限 dataset/seed 描述） | E001/E003 的方法排序与方差跨 dataset/seed 变化 | 尚未比较其他搜索空间/representation | 新搜索空间、更多 seeds、representation ablation | “当前结果随 dataset 和 search seed 变化” | “已证明对任意搜索空间都依赖/不依赖” | Discussion/Limitations | 2026-08-14 |
| C011 | 独立 final evaluation 使用 validation-only selection，未用 test 选候选 | 评估严谨性 | [已验证]（限 Cora） | E001/E002 final summaries 明确 `selected_by_final_validation` | 跨数据集没有 final eval | 跨数据集同协议 final summary | “Cora final candidate 由 final validation 选择，test 随后报告” | “已排除所有 selection bias” | Method/Experiments | 2026-08-14 |
| C012 | 本文相对普通 joint NAS+HPO 的新增贡献已明确 | 创新性 | [未知] | 已核实 D-VAE、TED 与本项目实现边界 | 相关工作资产覆盖不足 | joint NAS+HPO、BO initialization、multi-fidelity、KG 的原始文献卡 | “本文研究结构化初始化的成本感知评估” | “这是首个/新颖的 joint NAS+HPO 方法” | Introduction/Related Work | 2026-08-14 |

## 摘要/贡献放行

- 可进入摘要：[已验证] 的中性协议事实；C003/C006/C009 的 negative or boundary result 只能按证据覆盖范围表述。
- 不得进入摘要：C001/C004/C005/C007/C008/C012 的确定性版本。
- 任何 Claim 升级必须先更新对应 evidence card 的统计分析、Claim 影响和 eligible paper update，再由人工审核。
