---
last_updated: 2026-08-14
status: provisional
claims: [C001, C002, C003, C004, C005, C006, C009, C010, C011, C012]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

# Paper story

## One-sentence story

[假设] 结构化初始化可能改变联合 GNN NAS+HPO 的早期搜索轨迹，但只有把额外 low-fidelity 成本、surrogate 诊断、validation-only final selection 和跨数据集 negative results 放在同一证据链中，才能判断这种改变是否构成实际收益。

## Problem → gap → research question

- Problem：[已验证] 联合架构与 HP 搜索的每个 full observation 需要训练具体 GNN，初始化会决定 GP 最早看到的区域。
- Gap：[已实现未充分验证] 现有仓库文献资产还不足以证明领域工作忽略了初始化；本项目内部明确暴露了只按 full budget、单个 best validation 或单一 surrogate 指标汇报的风险。
- Research question：[假设] 结构化且 low-fidelity-assisted 的初始化是否改善覆盖、成本感知 sample efficiency、surrogate 质量和 validation-only selection 后的 test 表现？

## Method intuition

[假设] 分簇分配先覆盖潜在不同区域，TED/Schur 在簇内减少冗余，low-fidelity promotion 再优先投入 full budget；之后由同一 scratch Exact GP 与 qLogEI 继续在线搜索。该直觉不是已证明机制，G100/G150/K6 同时改变多个因素。

## Evidence design

1. [`evidence/E001.md`](evidence/E001.md)：Cora S0/G100/G150，同一 300-full search 和共同 Top-10 final-evaluation 协议。
2. [`evidence/E002.md`](evidence/E002.md)：Cora K6-LF 的独立 Top-7/150-epoch final-evaluation 协议，只做协议内结论。
3. [`evidence/E003.md`](evidence/E003.md)：CiteSeer/PubMed/DBLP/Flickr 的 60-run 历史 search，回答 validation、surrogate 和成本，不回答 final test。
4. [实验进行中] 从 histories 生成 early-budget、equivalent-full 与 coverage 指标；[未知] 跨数据集 final evaluation。

## 当前最可能成立的叙事

[已验证] 现有证据不支持“clustered initialization 普遍提高最终 accuracy”。更可能成立的是一篇审慎的 empirical study：不同初始化改变搜索成本和部分 surrogate 指标，但其收益依赖 dataset、metric、budget axis 和 evaluation tier；在 Cora final test 与跨数据集 search validation 上，强性能主张未获支持。

## 主假设不成立时的备选叙事

[假设] 如果成本感知 early-efficiency 和机制分析也不支持改善，论文可转为 negative-results / evaluation-rigor 叙事：展示为何固定 full budget 会掩盖 low-fidelity 成本、为何 surrogate 单项改善不保证 final test、以及如何用 validation-only selection 和 evidence tiers 防止过度结论。是否适合 ICLR 仍取决于文献定位与分析完整度。

## Boundary 与 alternative explanations

- 结果可能由特定 D-VAE 表示、候选池、搜索空间或小样本 search seeds 驱动。
- G100/G150 将 clustering、TED、LF shortlist、promotion 数和剩余 qLogEI 次数绑定，差异不能归因于单一组件。
- low-fidelity 与 full-fidelity 相关性尚未形成完整证据；额外候选训练本身可能解释早期收益。
- Cora 与 K6 的 final-evaluation 协议不匹配；跨数据集只有 search validation。
- surrogate 最后窗口的单项好坏可能来自目标范围、噪声或采样轨迹，而不是更好的全局模型。

## 最薄弱证据环节

1. C001/C004/C005：未完成预注册的 early/cost-aware/coverage 复算。
2. C006/C007：没有跨数据集 final test、formal paired effect size 与多重比较处理。
3. C012：仓库内可核实文献仅覆盖 D-VAE 和 TED，创新性边界仍未知。

当前可写：问题、协议、分层证据和不支持的强主张。当前禁止：把暂定 story 写成已证明因果机制或广泛性能优势。
