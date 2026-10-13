# Latent Representation Diagnosis

最后更新：2026-08-18  
项目快照：`/home/fangxy/D-VAE`，分支 `feature/accuracy-bottleneck-diagnostics`  
证据标签：`[文献事实]`、`[代码事实]`、`[实验事实]`、`[推断]`、`[计划]`

> **结论先行。** 当前证据不支持“decoder randomness 与 latent representation 二选一”。随机 decoder 是一个真实且严重的复现性问题；但现有 Track A 已经给出了关键反事实：把 decoder 改成完全确定的 greedy/argmax 后，modal consistency 从约 0.23–0.41 升到 1.00、invalid rate 降到 0，latent–performance geometry 却没有恢复，HP-near 与 latent-local 区域的差距反而扩大。因此，**decoder repair 是必要的工程修复，但不是当前 BO 失效的充分解释；当前主导科研假设应转为 search-coordinate / representation geometry 与 performance objective 不匹配。**
>
> 但还不能把“representation 是唯一主因”写成最终结论：greedy geometry 使用的是历史 stochastic-policy architecture 的 validation 标签，属于零训练反事实，尚未对 greedy 解码出的新 architecture 重新标注。P1 的小规模 matched relabel 实验将关闭这一因果缺口。

## 1. Research Question

本报告把问题拆成三个可独立证伪的层次，而不是把所有失败统称为 “GP 不好”。

| 层次 | 假设 | 可观测结果 | 首要干预 |
| --- | --- | --- | --- |
| Decoder | 同一 `z_search` 因采样得到不同 architecture，造成标签与候选身份不稳定 | one-z→many、decode entropy、invalid、cycle drift | 固定 checkpoint，仅替换 decoding policy |
| Representation / coordinates | 即使 mapping 确定，欧氏距离仍不保持 topology 或 performance | `d_latent` 与 `d_arch`/`|Δy|` 弱相关，局部邻域不保真 | 比较 `z_search`、encoder mean、结构表示；或 performance-aware retraining |
| Surrogate | 输入几何可用，但 GP/kernel/噪声模型不能学习局部排序 | 多种合理 representation 都有信号，而同一 GP 仍系统失效 | 最后才替换 kernel、likelihood 或 surrogate |

六个论文级问题及当前状态如下。

| 问题 | 当前状态 | 还缺什么 |
| --- | --- | --- |
| Q1. stochastic decoding 是否是主要 mismatch 来源？ | **部分否定**：它很严重，但 greedy 后 geometry 未恢复 | 对 greedy architecture 做 matched relabel |
| Q2. learned latent 是否保持 topology？ | **倾向否定**：roundtrip drift 大、结构距离比 `z_search` 更有解释力 | 直接比较 `z_search`、`μ(a)` 与 architecture kNN |
| Q3. 是否保持 performance geometry？ | **否定证据较强**：global/HP-near/latent-local 的 latent rho 为负，top region 约为 0 | matched relabel 与 encoder-mean 分析 |
| Q4. reconstruction-trained latent 是否适合作为 BO 坐标？ | **目前不支持** | matched GP replay：同观察、同切分、只换 representation |
| Q5. structure/performance-aware 表示是否更好？ | **结构表示有信号，但高性能区未稳定胜出** | `z_search`/`μ(a)`/structure/hybrid 的严格 matched replay；P2 performance-aware fine-tune |
| Q6. decoder repair 是否足够恢复 latent BO？ | **现有零训练反事实显示不够** | P1 relabel 后再决定是否值得 tiny BO pilot |

关键概念必须分开：

- `z_search`：Phase4 直接在固定连续域中优化的 12D architecture 坐标；它不一定等于任何 architecture 的 encoder posterior mean。
- `z_enc = μ(a)`：给定离散 architecture 后由 encoder 得到的确定性 posterior mean。
- `h_direct`：当前 `global4` 的 4D 直接 HP 坐标；它不是 Phase4 中实际使用的 HP-VAE latent。
- `a_hist`：历史 stochastic/modal policy 实际评估并产生 validation 标签的 architecture。
- `a_greedy`：同一 `z_search` 通过 deterministic greedy 得到的 architecture。

这个区分非常重要：现有 A1 中所谓的 `latent_distance` 实际是 `z_search` 的距离，而不是 `μ(a_hist)` 的距离。

## 2. Current Evidence

### 2.1 Repository and provenance

[代码事实] 只读检查时：

- 当前分支为 `feature/accuracy-bottleneck-diagnostics`，相对远端同名分支 ahead 11。
- 检查前已有用户改动：`bo_phase4.py` 被修改，`Cora/Cora/` 与 `artifacts/packages/` 未跟踪；本调查不触碰这些内容。
- GitHub 连接确认 `Sherlockmorefff/NAS` 为当前 remote 对应的公开仓库，默认分支 `main`；本任务没有 commit、push 或创建 PR。
- 本报告之外没有修改核心代码，也没有启动 GNN 训练。

### 2.2 Encoder: `architecture → z_arch`

[代码事实] 当前 architecture encoder 是 D-VAE 的异步 DAG message passing encoder：

- [`models.py`](../models.py) 的 `DVAE.encode()` 沿拓扑顺序传播，输出 `fc1(Hg)=μ` 与 `fc2(Hg)=logvar`。
- 训练态 `reparameterize()` 使用 `μ + 0.01·σ·ε`；eval 态直接返回 `μ`。因此同一 architecture 在 `model.eval()` 下的 encoder 输出是确定的。
- [`nas_space.py`](../nas_space.py) 的 `encode_to_joint_latent()` 拼接 `μ_arch` 与 `μ_hp`，不采样。
- `JointSpaceVAE.forward()` 的训练路径会采样 architecture/HP latent；但正式 Phase4 的 `global4` 后四维是 direct normalized HP coordinates，evaluation 不走 `hp_vae.decode()`。
- 正式 BO 不是从某个 architecture encode 后再优化其 `μ`；候选池直接在有界 `z_search=[z_arch,h_direct]` 中采样/优化，然后 decode `z_arch`。

### 2.3 Decoder: `z_arch → architecture`

[代码事实] 当前 decoder 默认明确是 stochastic：

- [`models.py`](../models.py) 的 `DVAE.decode(z, stochastic=True)` 对内部 node/op 的 softmax 分布调用 NumPy `random.choice`。
- edge 由 sigmoid 概率与 `torch.rand_like` 比较后作 Bernoulli 采样。
- `stochastic=False` 时 node 用 `argmax`，edge 用阈值 `>0.5`，因此形成确定性 greedy policy。
- 最后一个 node 被强制为 END，END 会连接所有 loose ends；生成顺序保证 DAG，但不能保证有效层数、唯一性或良好 roundtrip。
- 当前代码没有 temperature、beam search、grammar-level canonicalization 或 latent projection。
- [`eval_utils.py`](../eval_utils.py) 的 `_decode_arch()` 默认执行 3 次 stochastic decode，过滤无有效层结果，再按最终 `(operations, edges)` 取众数。
- RNG 可由 `decoder_seed` 隔离并复现。Phase4 的 seed 由 search seed 与 `z_arch` 共同导出，因此固定 search seed 可复现，但相同坐标在不同 search seed 下理论上可选择不同 architecture。
- “取 3 次众数”不是 deterministic decoder；它只是有限样本下对一个高熵生成分布的估计。

### 2.4 Training objective

[代码事实] `DVAE.loss()` 是 teacher-forced node/edge reconstruction negative log-likelihood 加 `β·KL`，默认 `β=0.005`。仓库中的该 objective 没有 accuracy regression、ranking、contrastive 或 metric-learning 项。

[推断] 因此 checkpoint 被优化为“生成训练 architecture”，而不是“让欧氏距离与 Cora validation accuracy 的差异单调对应”。良好 reconstruction 即使成立，也不是 performance geometry 的充分条件。

### 2.5 GP and search interface

[代码事实] [`surrogate/accuracy_gp.py`](../surrogate/accuracy_gp.py) 将 12D `z_arch` 由 `[-z_bound,z_bound]` 映射到 `[0,1]`，HP 坐标保持 `[0,1]`，拟合 `SingleTaskGP` 并使用 qLogEI。默认 kernel 交给 BoTorch `SingleTaskGP`；conditional 模式才显式使用带 ARD 的 masked RBF。正式 observation 的标签来自一次 seed-conditioned modal stochastic decode 后的 architecture 训练结果。

[推断] 所以 GP 实际学习的是：

```text
y = performance( Decode_stochastic-modal(z_search; search_seed), h_direct, train_seed )
```

而不是一个天然确定的 `performance(z_search)`。decoder noise、off-manifold search coordinates、representation distortion 与训练噪声都会进入同一个监督关系。

### 2.6 Existing decoder and geometry experiments

[实验事实] Stage A 对 4,500 条正式 records 去重后得到 3,582 个候选；128 个代表坐标各用 10 个 decoder seeds：

| 指标 | 结果 |
| --- | ---: |
| fixed-seed reproducibility | 1.000 |
| mean modal consistency | 0.2289 |
| normalized architecture entropy | 0.8677 |
| one-z→many rate | 1.000 |
| many-z→one modal collision | 0.3281 |
| median normalized `z→a→μ(a)` L2 drift | 0.4259 |
| p90 normalized roundtrip drift | 0.5025 |

证据：`results/posthoc/cora-accuracy-stage-a-v1/latent_geometry/a1_result.json`。

[实验事实] 结构距离与 validation difference 的 Spearman 关系系统性强于 `z_search` 距离：

| Region | Pairs | latent rho | operation rho | path rho | combined rho | discrete − latent effect, 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Global | 100,000 | -0.1618 | 0.1126 | 0.1535 | 0.0589 | 0.2207 [0.2130, 0.2285] |
| HP-near | 13,693 | -0.1502 | 0.1867 | 0.2265 | 0.1203 | 0.2705 [0.2483, 0.2917] |
| Latent-local | 15,680 | -0.1518 | 0.2034 | 0.2405 | 0.1529 | 0.3047 [0.2844, 0.3241] |
| Top 20% | 50,000 | 0.0026 | 0.0235 | 0.0286 | 0.0210 | 0.0184 [0.0073, 0.0299] |
| Exact-HP, different architecture | 1,381 | 0.0765 | 0.2041 | 0.2156 | 0.1680 | 0.0915 [0.0232, 0.1586] |

[实验事实] Track A 随后比较 modal-3/5/9/15/25 与 deterministic greedy。增加随机采样次数没有让众数稳定收敛；9→15 的 independent-half agreement gain 为负。greedy 的关键结果：

| Policy | Modal consistency | Entropy | Half-mode agreement | Invalid | many-z→one collision | Median arch cycle discrepancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current modal-3 | 0.4141 | 0.8932 | 0.1016 | 0.0443 | 0.2598 | 0.2500 |
| modal-9 | 0.2378 | 0.8688 | 0.2500 | 0.0516 | 0.2797 | 0.2500 |
| modal-25 | 0.1884 | 0.8241 | 0.3125 | 0.0491 | 0.3391 | 0.2750 |
| deterministic greedy | **1.0000** | **0.0000** | **1.0000** | **0.0000** | 0.4297 | **0.2000** |

greedy 修复了 one-z→one 输出与 validity，却增加了 many-z→one collision，说明折叠并未消失。

[实验事实] 用 greedy architecture 重算结构距离后：

| Region | latent rho | greedy combined rho | discrete − latent effect | 相对 Stage A gap shrinkage |
| --- | ---: | ---: | ---: | ---: |
| HP-near | -0.1502 | 0.1334 | 0.2836 [0.2609, 0.3049] | -4.8% |
| Latent-local | -0.1518 | 0.1896 | 0.3413 [0.3218, 0.3609] | -12.0% |
| Top 20% | 0.0035 | -0.0037 | -0.0072 [-0.0182, 0.0034] | 60.8%，但两者都接近零 |

Track A 的预注册判定为 `CASE_B / LATENT_GEOMETRY_STILL_FAILS`。证据：

- `results/posthoc/cora-accuracy-decoder-b0-v1/track_a/decoder_policy_results.json`
- `results/posthoc/cora-accuracy-decoder-b0-v1/track_a/representation_verdict.json`
- `results/posthoc/cora-accuracy-decoder-b0-v1/track_a/track_a_summary.json`

[实验事实] 该分析没有新 GNN 训练。其重要限制是：validation 标签属于历史 `a_hist`，而 alternative-policy 距离属于 `a_greedy`。因此它足以反驳“只要输出确定，原有 geometry 指标自然恢复”，却不能替代对 `a_greedy` 的真实 performance relabel。

### 2.7 Existing surrogate evidence

[实验事实] A2 在 15 个 run 上判定 `A2-GLOBAL-FAIL`：global Spearman 0.2430、late-50 Kendall 0.1383、top-20 Kendall 0.1021、late-50 top-20 Spearman 0.0717 且 CI 跨 0、top-10 regret 0.0156、95% coverage 0.9309。局部 pairwise accuracy 约 0.58，只有弱信号。证据：`results/posthoc/cora-accuracy-stage-a-v1/surrogate_audit/a2_result.json`。

[实验事实] 已有 offline prefix-only replay 使用 131D architecture features + 4D HP，对 15 条 histories 保持无 future-label leakage。boosted tree/ensemble 明显改善 late-50 Kendall、NDCG 与 regret，但 top-20 Kendall 和 top-region pairwise 没有稳定优于 latent GP，因此严格 gate 为 `NO-GAIN`。这不是“结构表示完全无用”，而是“已有结构 predictor 尚未在关键高性能区形成稳定胜出”。证据：

- `results/posthoc/cora-offline-predictor-v1/offline_predictor_result.json`
- `results/posthoc/cora-offline-predictor-v1/summary.json`

[推断] 这进一步说明：不能仅凭 global/late-stage 指标修改 surrogate；必须先完成 matched representation comparison，并将 top-region 排序设为主判据。

## 3. Decoder Randomness Hypothesis

### 3.1 Why the hypothesis was plausible

[文献事实] D-VAE 与 Graph-VAE 类方法把 node type 建模为 categorical、edge 建模为 Bernoulli，生成式语义本身允许同一 `z` 对应一个 architecture 分布。原始 D-VAE 通过多次采样报告 reconstruction，而不是证明一个坐标具有唯一离散映射。

[代码事实] 当前实现完整保留了这一 stochastic generation 机制，而 BO 把每个连续坐标当作单值输入。这构成了模型语义与 surrogate 语义之间的直接冲突。

### 3.2 What current experiments already establish

[实验事实] fixed seed 复现率为 1 只说明 PRNG 可控，不说明 mapping 稳定。跨 seed 的 one-z→many 为 100%，modal consistency 约 0.229；这是 decoder distribution 高熵，而不是普通的不可复现 bug。

[实验事实] modal sample 从 3 增加到 25 没有解决 ambiguity。greedy 则把 ambiguity/invalid 彻底消除，所以当前网络已经提供一个低成本 deterministic policy，无需先实现 beam search。

[实验事实] greedy 后性能 geometry 没有整体恢复。因此“decoder randomness 是唯一或主导瓶颈”已被现有零训练反事实显著削弱。

### 3.3 Falsifiable decoder-dominant criterion

[计划] 只有同时满足以下条件，才把 decoder 称为 dominant bottleneck：

1. 相同 `z_search` 下 deterministic policy 的 consistency=1、invalid≈0；
2. 对 deterministic architecture 重新获得 matched performance 标签；
3. 相比 current policy，HP-near/latent-local 的 `corr(d_z,|Δy|)`、局部 pairwise ranking 或 Lipschitz violation 显著改善；
4. 多个预注册 region 中至少两个的 paired improvement CI > 0，并且 geometry gap 至少收缩 50%；
5. 相同 observation budget 的 tiny replay/BO recommendation quality 同向改善。

若只满足第 1 条而 3–5 不满足，则 decoder 是复现性问题而不是 performance-search 的主要问题。

## 4. Representation Mismatch Hypothesis

### 4.1 Two different mismatches

必须区分：

1. **Search-coordinate mismatch**：BO 优化的 `z_search` 可能位于 encoder posterior manifold 之外；`z_search → a → μ(a)` 发生大漂移。
2. **Encoder-objective mismatch**：即使使用 `μ(a)`，reconstruction-trained representation 也可能只保持可重建性，而不保持 Cora/HP 条件下的 performance ranking。

现有 median normalized roundtrip drift 0.426 已支持第 1 类风险，但还没有把 `μ(a)` 与 `z_search` 放入同一 matched geometry/surrogate test，所以不能直接断言第 2 类已经完全证实。

### 4.2 Why reconstruction is not enough

[文献事实] D-VAE 自身只对 computation-aware injective encoding 给出条件性理论讨论，并明确不保证 performance smoothness。NAO、Graph-VAE、SVGe fine-tuning 与 CR-LSO 都另外引入 performance predictor/loss，正说明 reconstruction geometry 不是 performance geometry 的充分条件。

[实验事实] 当前 top 20% 区域中，latent、operation、path 距离与 `|Δaccuracy|` 的 rho 都接近 0。结构表示在全局/局部有优势，但高性能窄区也存在噪声或功能等价问题。

[推断] 最可能的组合机制是：

```text
高熵 stochastic decoder
        +
off-manifold z_search / cycle drift
        +
reconstruction-only latent objective
        +
top-region accuracy noise and HP interaction
        ↓
weak local ranking for GP/BO
```

### 4.3 Current best causal conclusion

[推断] 当前可以写成：

> Decoder randomness 是已证实的 mapping defect；deterministic decoding 可以修复候选身份与 validity，但现有反事实显示它不足以恢复 search-coordinate performance geometry。representation/search-coordinate mismatch 是下一优先机制，surrogate modification 暂不应先行。

不能写成：

> Encoder 已被最终证明是唯一主因。

原因是 `z_search` 与 `μ(a)` 尚未分离，且 greedy architectures 尚未 matched relabel。

## 5. Zotero Literature Search

### 5.1 Search scope

[文献事实] Zotero Desktop local API 状态正常（Zotero 9.0.6）。NAS 根收藏夹 key 为 `9CJWSSEV`。调查不是关键词抽样，而是先枚举完整 subtree，再筛 metadata/abstract/full text。

| Collection | Key | Top-level items |
| --- | --- | ---: |
| NAS | `9CJWSSEV` | 59 |
| BO | `UEERN834` | 9 |
| Continuous–Discrete Alignment | `CS7HH69H` | 1 |
| NAS+HPO | `3YLAULIX` | 2 |
| new paper | `9AQDN3W9` | 18 |
| pre | `69D6VFE4` | 1 |
| 实验对比 | `2CMTMUP9` | 8 |
| 条件参数 | `HL5HRPKJ` | 4 |
| 聚类 | `HTHJ38MF` | 1 |

跨 collection 去重后共有 99 个 top-level items，其中 1 个是 attachment、98 个是 bibliographic items。全部 98 个条目完成 title/metadata 层筛查；对直接涉及 latent NAS、decoder、architecture encoding、performance predictor、structure-aware kernel 的 10 篇读取 Zotero indexed full text。

### 5.2 Evidence table

| Paper | Year | Zotero key | Problem addressed | Representation | Decoder | Optimization space | Relevant mechanism | Solution | Evidence strength | Relevance |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| D-VAE: A Variational Autoencoder for DAGs | 2019 | `64R8A368` | DAG continuous embedding / BO | async computation-aware graph encoder, Gaussian latent | autoregressive categorical/Bernoulli stochastic decoder | latent GP/BO | reconstruction、validity、stochastic mapping；无 performance guarantee | DAG-aware encoder/decoder + GP | 高：方法与当前代码直接同源 | 5 |
| Neural Architecture Optimization (NAO) | 2019 | `VZPZKP89` | 在连续 embedding 中沿 performance gradient 搜索 | LSTM sequence encoder | autoregressive sequence decoder，valid-token constraint | supervised latent gradient ascent | predictor loss 直接塑造 embedding；对称 architecture augmentation | joint encoder–predictor–decoder loss | 高：直接解决 objective alignment | 5 |
| Neural Architecture Optimization with Graph VAE | 2020 | `Q6LX85LZ` | graph latent NAS 的 performance/complexity 优化 | GNN VAE latent | probabilistic whole-graph decoder | latent gradient ascent | reconstruction+KL+performance+complexity joint training | performance-aware graph latent | 高：与当前 VAE 最接近 | 5 |
| Smooth Variational Graph Embeddings (SVGe) | 2021 | `F9NX59EJ` | smooth、isomorphism-aware latent NAS | two-sided variational graph embedding | canonical/two-sided graph generation；论文强调 unique/injective mapping | latent GP/BO | canonicalization、isomorphism、uniqueness/folding、performance fine-tuning | smoother graph VAE + predictor fine-tune | 高：有 matched D-VAE comparison | 5 |
| CR-LSO | 2025 | `7BZLHMWG` | performance mapping 非凸、latent gradient 容易陷局部最优 | G-VAE latent jointly regularized by ICNN | probabilistic whole-graph MLP decoder | convex latent gradient ascent | performance predictor 反向约束 latent geometry；duplicate adaptive step/noise | convex performance-aware latent | 中高：直接机制，benchmark setting 较理想化 | 5 |
| BANANAS | 2021 | `GNA3BFAK` | predictor-based sample-efficient NAS | path encoding / adjacency encoding | 无 decoder | discrete mutation + neural surrogate | architecture-specific encoding 可优于 VAE；碰撞若 performance-compatible 可接受 | path features + ensemble + mutation | 高：直接反驳“必须用 VAE” | 5 |
| NASBOT | 2018 | `TZPICARS` | 对离散 architecture 直接做 BO | OTMANN structure distance/kernel | 无 decoder | discrete evolutionary acquisition | operation/structural position distance与 performance smoothness假设 | graph-distance GP | 高：结构 kernel 的经典直接方案 | 5 |
| How Powerful are Performance Predictors in NAS? | 2021 | `LKDPU4G3` | 系统比较 31 类 predictor/encoding | adjacency、path、GCN、tree、zero-cost 等 | 通常无 decoder | predictor-guided discrete NAS | local/mutation/top region 比 global 难；encoding 与数据量强交互 | matched predictor evaluation | 高：直接指导 Diagnostic C | 4 |
| LS-CATE-SANAS | 2026 | `A3B7C8M7` | Transformer-VAE latent NAS 与 collapse | ops/degree/distance/adjacency Transformer-VAE | residual MLP decoder | latent CMA-ES + GP/UCB | 显式讨论 many-z→same、invalid、dense entanglement | richer topology encoding + surrogate-assisted EA | 中：新论文，改进同时改变多组件 | 4 |
| Continuous–Discrete Alignment Optimization | 2025 | `3JJGDBN4` | DARTS continuous/discrete discretization gap | continuous operation weights | argmax discretization | differentiable supernet | 连续优化对象与最终离散对象不一致会降性能 | 将 discretization 反馈进训练 | 中：机制类比，不是 VAE latent | 3 |

公开定位链接：D-VAE [arXiv:1904.11088](https://arxiv.org/abs/1904.11088)、NAO [arXiv:1808.07233](https://arxiv.org/abs/1808.07233)、Graph VAE [arXiv:2006.10310](https://arxiv.org/abs/2006.10310)、SVGe [arXiv:2010.04683](https://arxiv.org/abs/2010.04683)、BANANAS [AAAI DOI](https://doi.org/10.1609/aaai.v35i12.17233)、[NASBOT](https://proceedings.neurips.cc/paper/2018/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html)、Performance Predictors [arXiv:2104.01177](https://arxiv.org/abs/2104.01177)、CR-LSO [arXiv:2211.05950](https://arxiv.org/abs/2211.05950)、LS-CATE [DOI](https://doi.org/10.1016/j.neucom.2026.133116)、CDAO [DOI](https://doi.org/10.1016/j.engappai.2025.110721)。

## 6. High-Relevance Papers

以下只把 full text 中明确出现的机制写成文献事实；“如何移植”是本报告的推断。

### 6.1 D-VAE — Zotero `64R8A368`, relevance 5

- **Representation**：[文献事实] 使用沿 DAG 计算依赖的异步 message passing，把图映射到 Gaussian latent。
- **Training objective**：[文献事实] reconstruction likelihood + `α·KL`，论文实验 `α=0.005`；无 accuracy supervision。
- **Structural preservation**：[文献事实] 在 aggregation/update injective 的条件下讨论 computation encoding 的 permutation invariance/injectivity；这是模型性质条件，不等于有限 checkpoint 已保持距离。
- **Performance preservation**：[文献事实] 没有理论保证；论文靠 posterior mean 上的 GP 预测/BO 实验验证实用性。
- **Decoder**：[文献事实] node categorical sampling、edge Bernoulli sampling，是 stochastic autoregressive decoder。
- **Search**：[文献事实] posterior means 上训练 sparse GP，以 EI 在 latent 中做 BO。
- **Mapping**：[文献事实] 优化的 latent 通过生成式 decoder 回到 DAG；reconstruction 通过多次 latent/decode sampling 评估。
- **Failure mode**：[文献事实] 关注 validity/uniqueness/novelty 与 remapping accuracy，但不提供同一 `z` 唯一 architecture 的保证。
- **Transferability**：[推断] 当前 checkpoint 正是该设计族；论文不能作为当前 mapping 稳定的证据，反而要求单独审计 stochastic decoding 与 posterior-manifold coverage。

### 6.2 NAO — Zotero `VZPZKP89`, relevance 5

- **Representation**：[文献事实] architecture token sequence 经 LSTM encoder 得到连续 embedding。
- **Training objective**：[文献事实] `λ·performance prediction loss + (1−λ)·reconstruction loss`；论文 CNN/RNN 设置分别使用较高的 predictor 权重。
- **Structural preservation**：[文献事实] 使用对称 architecture data augmentation，使等价表示更接近；没有一般图距离等距保证。
- **Performance preservation**：[文献事实] performance MLP 与 encoder 联合训练，predictor loss 明确正则 encoder，避免只复制 token。
- **Decoder**：[文献事实] autoregressive sequence decoder，并限制每一步 valid token set；全文证据不足以把所有 inference 概括为随机或 greedy，不能过度外推。
- **Search**：[文献事实] 沿 performance predictor 对 embedding 的梯度上升。
- **Mapping**：[文献事实] decode 得到 architecture；若新 embedding 解码回原 architecture，则增大 step size 直到得到不同候选。
- **Failure mode**：[文献事实] 显式处理 duplicate/same-decode；报告 reconstruction Hamming distance 与 predictor pairwise accuracy。
- **Transferability**：[推断] 最直接的可移植思想是 joint performance regression，而不是复制其 sequence encoder。当前 D-VAE 可在 `μ_arch` 上接 predictor，并联合 fine-tune encoder/decoder。

### 6.3 Neural Architecture Optimization with Graph VAE — Zotero `Q6LX85LZ`, relevance 5

- **Representation**：[文献事实] GNN encoder 将 architecture graph 编码到 VAE latent。
- **Training objective**：[文献事实] node/edge reconstruction、KL、performance squared error 与 complexity squared error联合训练。
- **Structural preservation**：[文献事实] 通过 graph reconstruction 建模结构，但没有普适的 graph-distance isometry 保证。
- **Performance preservation**：[文献事实] performance predictor loss 直接参与 latent learning，是 performance-aware representation。
- **Decoder**：[文献事实] 对整个小图进行 probabilistic node/edge generation；仍是 stochastic mapping。
- **Search**：[文献事实] 在 `performance − complexity` predictor 上做 stochastic gradient ascent。
- **Mapping**：[文献事实] 优化后的 latent 由 graph decoder 映射回 architecture。
- **Failure mode**：[文献事实] 报告 validity、reconstruction、uniqueness；优化后 uniqueness 下降，表明多个 latent 会汇聚到重复 architecture。
- **Transferability**：[推断] 是 P2 的首选 objective 模板：保留 D-VAE 图模型，加入 validation-only performance head；但 decoder 确定性仍需独立处理。

### 6.4 SVGe — Zotero `F9NX59EJ`, relevance 5

- **Representation**：[文献事实] 双向/two-sided variational graph encoder，显式处理 computation graph 与 isomorphism。
- **Training objective**：[文献事实] 无监督 reconstruction+KL 预训练；随后可与 MLP performance predictor 联合 fine-tune encoder。
- **Structural preservation**：[文献事实] 通过 upper-triangular/canonical training representation 去除 isomorphic duplicates，并用 injectivity/uniqueness 讨论 latent smoothness。
- **Performance preservation**：[文献事实] 无监督阶段不保证；supervised fine-tune 才注入 accuracy 信息。
- **Decoder**：[文献事实] 论文目标是让 latent 对应唯一 canonical graph；其概率生成描述与“deterministic unique mapping”主张之间存在实现细节张力，移植前必须核对代码。
- **Search**：[文献事实] 使用 GP/EI，在 matched setup 中与 D-VAE 比较 generative model。
- **Mapping**：[文献事实] latent 经 two-sided graph decoder 回到 canonical architecture。
- **Failure mode**：[文献事实] 将低 uniqueness/多 latent 同 graph 视为 folded/non-smooth latent 的诊断；还讨论 D-VAE 的硬生成约束与跨搜索空间问题。
- **Transferability**：[推断] 当前可先移植 canonical fingerprint、isomorphism-aware dedup、uniqueness/folding diagnostics；完整替换 encoder/decoder 属于 P3。

### 6.5 CR-LSO — Zotero `7BZLHMWG`, relevance 5

- **Representation**：[文献事实] GNN-based VAE latent。
- **Training objective**：[文献事实] reconstruction+KL，同时用 ICNN performance predictor 的误差约束 latent，使 performance mapping 近似凸。
- **Structural preservation**：[文献事实] GNN aggregation提供 permutation invariance/injectivity 论述；不是严格距离保持。
- **Performance preservation**：[文献事实] ICNN 只能拟合凸函数，若 latent performance mapping 不凸则误差高；最小化 predictor loss迫使 encoder 改变 geometry。
- **Decoder**：[文献事实] MLP probabilistic whole-graph decoder，适用于小 cell graph。
- **Search**：[文献事实] 在 convex predictor 上做 gradient ascent，并交替更新高性能区域。
- **Mapping**：[文献事实] decode 后若 architecture 已存在，则增大 latent step；仍重复时增大 Gaussian perturbation。
- **Failure mode**：[文献事实] 明确讨论 similar embeddings→same architecture、invalid architecture 与 duplicate。
- **Transferability**：[推断] “performance loss塑造 latent”直接适用；但 ICNN convexity 假设对 GNN+HP/Cora 未经验证，不应作为 P2 第一选择。先做普通 performance head 的最小可证伪实验。

### 6.6 BANANAS — Zotero `GNA3BFAK`, relevance 5

- **Representation**：[文献事实] path encoding 描述 input→output operation paths，也比较 adjacency、GCN 与 VAE encoding。
- **Training objective**：[文献事实] architecture encoding 输入 neural predictor ensemble；不是 reconstruction objective。
- **Structural preservation**：[文献事实] path encoding 去除任意 node ordering；它可以 many-to-one，但论文认为其碰撞 architecture 往往有相近 validation error。
- **Performance preservation**：[文献事实] 由 supervised performance predictor 学习；经验上 path/GCN representation 在相应 benchmark 上优于 VAE predictor。
- **Decoder**：[文献事实] 无 latent decoder。
- **Search**：[文献事实] 直接对高分离散 architecture 做 mutation，再用 predictor/acquisition 排候选。
- **Mapping**：[文献事实] 始终在有效离散 architecture 中，因此没有 latent→architecture remapping。
- **Failure mode**：[文献事实] encoding collision 可以存在；关键不是严格 injective，而是 collision 是否 performance-compatible。
- **Transferability**：[推断] 当前 ops/edges/path signatures 已可构造 D 模型；这是一条不保留 VAE 的合法路线，尤其适合作为 matched surrogate 对照。

### 6.7 NASBOT — Zotero `TZPICARS`, relevance 5

- **Representation**：[文献事实] 不先压成固定 latent；用 OTMANN optimal-transport distance 比较 layer operation、mass 与从 input/output 出发的 structural position。
- **Training objective**：[文献事实] distance/kernel 不依赖 autoencoder reconstruction。
- **Structural preservation**：[文献事实] architecture topology与 operation label直接进入距离。
- **Performance preservation**：[文献事实] 假设小 OTMANN distance 倾向对应相近 validation error，并以 BO 实验检验；不是定理保证。
- **Decoder**：[文献事实] 无 decoder。
- **Search**：[文献事实] distance-based GP + acquisition，acquisition 通过离散 evolutionary mutations优化。
- **Mapping**：[文献事实] 候选始终是离散 architecture。
- **Failure mode**：[文献事实] kernel 的 PSD 性质不完全由理论保证，论文依赖经验矩阵可用性。
- **Transferability**：[推断] 不必直接实现完整 OTMANN；当前 op/edge/path distance 可先形成 literature-aligned architecture kernel，若 P0 有信号再考虑更精细的 GNN operation cost。

### 6.8 Performance Predictors — Zotero `LKDPU4G3`, relevance 4

- **Representation**：[文献事实] 系统比较 path、adjacency/one-hot、GCN、tree、zero-cost、learning-curve 等多类输入。
- **Training objective**：[文献事实] 直接拟合/rank architecture performance，而非重建 architecture。
- **Structural preservation**：[文献事实] 不要求单一表示满足所有搜索空间；encoding 与 predictor 的组合决定效果。
- **Performance preservation**：[文献事实] 通过 Pearson/Spearman/Kendall/sparse Kendall 与实际 NAS结果衡量。
- **Decoder**：[文献事实] predictor 比较通常直接作用于 architecture，无 latent decoder。
- **Search**：[文献事实] 在 uniform 与 mutation-local 数据分布上测试，并进入 predictor-guided NAS。
- **Mapping**：[文献事实] 离散候选直接特征化。
- **Failure mode**：[文献事实] local/top neighborhood 明显比 global test 难；global 指标好不能保证 NAS 有效。
- **Transferability**：[推断] Diagnostic C 必须以 prefix-only、top-region、late-stage、recommendation regret 为主；当前 offline replay 的评估思想正确，但还缺 `z_enc` 与 architecture-kernel GP 的同协议对照。

### 6.9 LS-CATE-SANAS — Zotero `A3B7C8M7`, relevance 4

- **Representation**：[文献事实] Transformer-VAE 同时编码 operation、degree、pairwise distance 与 adjacency matrix。
- **Training objective**：[文献事实] point-wise unsupervised reconstruction+KL 预训练；search 中再拟合 GP。
- **Structural preservation**：[文献事实] richer matrices旨在减少 masked/irreversible encoding 的结构信息丢失，但没有严格 distance guarantee。
- **Performance preservation**：[文献事实] 预训练 objective 未显式注入 performance；search surrogate 使用已评估候选。
- **Decoder**：[文献事实] residual fully-connected decoder；论文报告/讨论 invalid 与 architecture collapse。
- **Search**：[文献事实] CMA-ES-WM 采样 latent，GP/UCB 选择评估候选。
- **Mapping**：[文献事实] top latent candidates decode 后评估。
- **Failure mode**：[文献事实] 明确指出 multiple latents→same architecture、invalid、large-space information entanglement，并把它们列为后续问题。
- **Transferability**：[推断] 支持“当前 12D D-VAE 可能缺少结构上下文”的假设，但该方法同时改变 encoder、decoder 与 search strategy，不能作为单因果修复；适合作为 P3 方向而非 P0。

## 7. How Existing NAS Methods Solve the Problem

| Solution family | Literature mechanism | 解决什么 | 不能自动解决什么 | 对本项目的优先级 |
| --- | --- | --- | --- | --- |
| 1. Deterministic/constrained decoding | D-VAE greedy API；NAO valid-token constraints；SVGe canonical graph | one-z→many、invalid、候选身份复现 | many-z→one folding、performance geometry | 立即保留为 P1 policy |
| 2. Performance-aware representation | NAO joint predictor；Graph-VAE joint loss；SVGe fine-tune；CR-LSO regularization | reconstruction/performance objective mismatch | decoder ambiguity、标签噪声 | P2 首选 |
| 3. No generative decoder | BANANAS discrete mutation；arch2vec-like nearest valid buffer | latent→discrete ambiguity | 大空间覆盖、最近邻可扩展性 | P0 对照，P3 可替代路线 |
| 4. Structure-aware surrogate | NASBOT OTMANN；BANANAS path/adjacency | topology 信息直接进入 kernel/predictor | top-region performance noise | P0 matched replay，P3 正式搜索 |
| 5. Hybrid representation/kernel | architecture kernel + HP product/additive；multi-feature predictors | architecture 与 HP interaction | 表示维数、kernel calibration | P0/P3 |
| 6. Canonicalization/cycle control | SVGe canonical isomorphs、uniqueness；CDAO 连续–离散对齐思想 | 等价图重复、roundtrip mismatch | task-specific performance | P0 diagnostics，之后按证据实现 |

[推断] 对当前项目最合理的顺序不是“先调 GP”，而是：

```text
固定 decoder policy 和候选身份
→ 分离 z_search 与 μ(a)
→ matched representation replay
→ 小规模真实 relabel
→ 决定 performance-aware latent 还是 structure-aware discrete 路线
→ 最后才做正式 BO/surrogate 改造
```

## 8. Mapping Literature Solutions to D-VAE

### 8.1 Minimal mapping table

| Current component | Observed issue | Literature-supported intervention | Repository landing point | Gate before implementation |
| --- | --- | --- | --- | --- |
| `DVAE.decode(stochastic=True)` | high entropy、one-z→many | greedy/argmax；validity constraints；canonical fingerprint | `models.py`, `eval_utils._decode_arch` | P1 matched relabel shows geometry/recommendation gain |
| finite modal vote (3 trials) | mode estimate不稳定 | 不再靠增加 3→25；直接 deterministic or structured search | `eval_utils.py`, `bo_phase4.py` | Track A 已足以停止 sample-count tuning |
| `z_search` as GP coordinate | off-manifold/cycle drift | projection to `μ(decode(z))`、nearest valid/canonical representation | candidate preprocessing / logging | P0 比较 `z_search` vs `μ(a)` |
| reconstruction-only D-VAE | no performance objective | NAO/Graph-VAE式 performance head 联合 fine-tune | training pipeline/new experiment module | P0 matched replay显示 `μ(a)`有可塑性；P2 split无泄漏 |
| Euclidean GP on latent | structure signal丢失 | NASBOT-like architecture kernel；BANANAS path features | `surrogate/` new experimental predictor | P0 key top-region metrics pass |
| latent + direct HP | architecture–HP interaction未显式建模 | additive/product architecture+HP kernels；hybrid features | experimental surrogate | 与 current GP 同 prefix/seed/budget比较 |

### 8.2 Immediate engineering invariants

[计划] 无论最终选哪条方法，后续实现应强制记录：

- `decoder_policy`、`decoder_seed`、sample count/temperature（若有）；
- `z_search`、decoded architecture fingerprint、`z_enc=μ(a)` 与 cycle distance；
- architecture+HP fingerprint 作为 observation identity；
- invalid/duplicate reason，而不是静默过滤；
- performance label 属于哪个 decoded architecture，禁止把 `a_hist` 的标签归给 `a_greedy`；
- search seed、GNN train seed、dataset split identity。

### 8.3 What not to do first

- 不继续把 modal samples 从 25 增到更大；Track A 已显示收益不收敛。
- 不先加 temperature；它改变分布尖锐度，但不能证明 geometry 正确。
- 不先实现 beam search；当前 greedy 已能形成 consistency=1/invalid=0 的低成本干预，beam 只有在 greedy performance 明显差且 logits 中存在可利用次优序列时才值得。
- 不把 fixed seed reproducibility 当作 decoder stability。
- 不直接将历史 stochastic label 复用于新的 deterministic architecture。
- 不因 global regression 变好就替换正式 surrogate；top-region gate 必须通过。

## 9. Diagnostic Experiments

### 9.1 Common matched protocol

[计划] 所有实验共享以下不变量：

1. 同一批 architecture/candidate fingerprints；
2. 同一 HP、dataset split、search seed 与 train/evaluation seed；
3. 相同 observation prefix、train/test 时序和 refit budget；
4. architecture fingerprint 级别去重，禁止 future-label leakage；
5. test accuracy 不参与模型选择或 gate；
6. primary metrics 预注册为 top-20 Kendall、top-region pairwise、top-10 recall、NDCG@10、best-of-10 regret、late-50 Kendall；
7. global Spearman/RMSE、NLL/coverage 为 secondary diagnostics；
8. paired bootstrap 按 15 个 history/run 为重复单位，报告 CI 与 run wins；
9. decoder policy 与 performance label attribution 必须一致。

### 9.2 Diagnostic A — Decoder-only intervention

**已有部分（不重复训练）**

[实验事实] 128 个 `z`、10 seeds 的 stochastic audit，以及 modal-3/5/9/15/25/greedy policy sweep 已经回答了“随机性是否存在”“增加采样能否解决”“greedy 能否确定化”。没有必要先机械重复 N=30 才承认问题。

**缺失的因果闭环（P1）**

[计划] 构造 48 个 matched `z_search`：16 个 top、16 个 latent-local disagreement、16 个全局分层样本。只保留 `a_hist != a_greedy` 的 informative pairs，并固定同一 `h_direct`。

执行顺序：

1. 先以 `(architecture fingerprint, HP fingerprint, split, train seed)` 查询现有 cache；
2. `a_hist` 复用合法历史标签；`a_greedy` 只有 fingerprint/HP/train-seed 完全匹配时才复用；
3. 仅对 unmatched `a_greedy` 做小规模 evaluation，每个 pair 使用相同的 3 个 train seeds；
4. 比较 current vs greedy 的局部 `|Δy|/d_z`、top ranking、duplicate/invalid、recommendation regret；
5. 分别汇报 architecture policy effect 与训练噪声。

**判定：**

- greedy consistency 修复且 performance geometry gap ≥50% 收缩、paired CI>0：支持 decoder-dominant；
- consistency 修复但 geometry/recommendation 不改善：否定 decoder-dominant，支持 representation/coordinate mismatch；
- greedy architecture 本身 performance 更差：说明 deterministic policy 需要 constrained/beam/canonical alternative，但仍不能证明 latent geometry 良好。

### 9.3 Diagnostic B — Latent ↔ Architecture alignment

这是 P0 最关键的新增零训练分析。

[计划] 对同一 3,582 候选同时构造：

- `Z_search`: 历史 12D search coordinates；
- `Z_enc`: 对 `a_hist` 重新 encode 的 `μ_arch(a)`；
- `A_ops`: per-position operation one-hot；
- `A_edges`: canonical adjacency；
- `A_path`: input→output path signatures；
- `A_combined`: op/edge/path 的预注册组合距离。

主分析：

1. `Spearman(d_search,d_arch)` 与 `Spearman(d_enc,d_arch)`；
2. `Spearman(d_search,|Δy|)`、`Spearman(d_enc,|Δy|)`、`Spearman(d_arch,|Δy|)`；
3. latent kNN 与 architecture kNN 的 mean overlap，`k∈{5,10}`；
4. latent neighbors 内的 median `|Δy|`，与 random/architecture-neighbor 对照；
5. `z_search→a→μ(a)` drift 对 decode entropy、collision 与 `|Δy|` 的关系。

分层：global、exact-HP、HP-near、top 20%、每个方法/seed。architecture isomorphism 先 canonicalize；相同 architecture 不作为“结构近邻”重复计数。

**关键解释：**

- `Z_search` 差、`Z_enc` 好：主要是 off-manifold search/cycle mapping，优先 projection 或在 encoded valid set 中搜索；
- 两者 topology 都差：encoder/decoder representation 失真；
- topology 好但 performance 差：reconstruction/structure objective 与 BO objective mismatch；
- architecture neighbor 也无 performance signal：top-region 噪声、HP/dataset interaction 或结构距离不适合任务。

### 9.4 Diagnostic C — Matched surrogate comparison

[计划] 使用现有 15 条 histories 做严格 prefix-only replay：

| Model | Input / kernel | Purpose |
| --- | --- | --- |
| Zs | `z_search + HP`，当前 GP | baseline |
| Ze | `μ(a_hist) + HP`，相同 GP/refit schedule | 分离 search coordinate 与 encoder representation |
| D | canonical ops/edges/path + HP，相同 observation prefixes | architecture-specific encoding |
| K | architecture distance kernel ×/＋ HP kernel | NASBOT-like structure GP |
| H | `μ(a)` 与 architecture kernel/feature 的 additive hybrid | 检验互补信息 |

为避免“只换 representation”被模型容量混淆，报告两层比较：

- **C1 representation isolation**：Zs/Ze/D 使用同一 GP family、相同 refit schedule与优化预算；
- **C2 literature-native package**：K 使用预注册 architecture kernel，H 使用 additive/product kernel；它回答“哪套 surrogate space 最有效”，不声称只隔离 representation。

已有 offline boosted-tree/ensemble replay作为补充，不替代 C1/C2。

**Primary pass rule：** challenger 必须同时改善 top-20 Kendall 与 top-region pairwise（mean>0 且 CI>0 或 ≥10/15 run wins），至少改善 recall/NDCG/regret 中两项，并保持 late-50 Kendall。该规则与现有 offline predictor gate 对齐。

### 9.5 Diagnostic D — Encode–Decode cycle

[计划] 分两条 cycle：

```text
a_hist → μ(a_hist) → Decode_greedy → a'
z_search → Decode_policy → a → μ(a)=z'
```

指标：

- exact architecture fingerprint reconstruction；
- op/edge/path discrepancy；
- normalized `||z_search−z'||₂`；
- invalid 与 many-z→one collision；
- discrepancy 对 search-radius、validation rank、decode entropy 的条件关系。

[实验事实] 已有第二条 cycle 的 median normalized L2≈0.426（greedy policy 下 latent-distance median≈0.411）与 architecture discrepancy≈0.20，足以证明 mapping 非近似恒等。第一条 architecture reconstruction 仍需按 stochastic/greedy 分开汇总。

**判定：**

- `a→μ→a'` 好而 `z_search→a→μ` 差：search 进入 posterior manifold 外；
- 两条都差：encoder–decoder pair 本身未形成稳定 chart；
- cycle 好但 performance geometry 差：objective mismatch，而不是 mapping reconstruction。

### 9.6 Diagnostic E — Performance-aware representation test

[计划] 仅在 P0/P1 指向 representation 后进入 P2。首选 Graph-VAE/NAO 已验证的最小 objective：

```text
L_total = L_DVAE_reconstruction+KL + λ · MSE(f_perf(μ_arch, h_direct), y_val)
```

不在第一版自行发明 ranking/contrastive loss。实验组：

- E0：冻结当前 checkpoint；
- E1：当前 architecture encoder + performance head，仅训练 head，测可预测上限；
- E2：联合 fine-tune encoder、decoder 与 performance head，使用 reconstruction+KL+performance loss；
- E3（可选）：SVGe-style encoder fine-tune，decoder冻结，用于分离 representation plasticity 与 reconstruction drift。

协议：按 architecture fingerprint 切 train/validation/test；同一 architecture 的不同 HP 不跨 split 泄漏；λ 只在 inner validation 选择；test 不参与选择；decoder evaluation 固定 greedy；报告 reconstruction、validity、collision、cycle、top-region geometry 与 matched surrogate 全套指标。

**支持 objective-mismatch 的判据：** E2/E3 在 held-out architecture 上通过 Diagnostic C primary gate，同时 reconstruction/validity 没有灾难性下降。若只提升训练 predictor 而 held-out geometry/BO recommendation 不提升，则不能称为表示改善。

## 10. Causal Decision Matrix

| Experimental observation | Supports | Falsifies / weakens | Next action |
| --- | --- | --- | --- |
| deterministic decode 修复 consistency/invalid，真实 relabel 后 geometry 与 regret 大幅改善 | decoder-dominant | representation-primary | 固化 deterministic/constrained policy，再 tiny BO |
| consistency=1，但 matched relabel geometry基本不变 | representation/search-coordinate mismatch | decoder-dominant | 做 Ze/D/K/H 与 P2 |
| `z_search` topology/performance差，`μ(a)`显著好 | off-manifold continuous search / cycle mismatch | encoder representation 已坏 | projection、encoded-valid search 或 nearest-valid mapping |
| `z_search` 与 `μ(a)` topology 都差 | encoder/decoder representation distortion | 仅 search-bound 问题 | structure-aware surrogate 或重训 encoder |
| topology alignment 好，performance alignment 差 | reconstruction/structure objective mismatch | 单纯结构丢失 | performance-aware fine-tune |
| D/K surrogate 在 top-region 稳定优于 Zs/Ze | 当前 latent 不适合作为 performance-search coordinates | “只调 GP 即可” | structure-aware/hybrid P3 |
| performance-aware latent 在 held-out top-region 稳定改善 | reconstruction-only 不足 | “latent 不可挽救” | P2→small BO pilot |
| greedy many-z→one collision仍高且 projection后仍重复 | latent folding / decoder collapse | 仅 one-z→many 随机性 | canonicalization、dedup、discrete mutation |
| structure distance有 `|Δy|` 信号但 K-GP不提升 | kernel/normalization/finite-data问题，或 geometry metric不匹配 acquisition | “结构表示必然更好” | kernel audit，不立即正式替换 |
| 所有 representation 在 top region 都接近随机，但 global 尚可 | performance noise ceiling / HP与dataset条件效应 | 单一表示修复 | 重复训练标签、heteroscedastic/noise diagnostics |
| 所有表示与重复标签均可靠，GP仍差 | surrogate/kernel/likelihood bottleneck | decoder/representation 为主 | 最后进入 surrogate redesign |
| greedy architecture performance系统性低于 stochastic mode | greedy policy选择偏差 | “argmax就是正确 decoder” | constrained beam/canonical policy，再重复 P1 |

## 11. Recommended Experiments in Priority Order

### P0 — 不做新 GNN 训练

1. **接受并固化既有 Track A 结论**：停止增加 modal sample count；把 `CASE_B` 与 label caveat 写入实验账本。
2. **坐标分离**：对 3,582 候选生成 `μ(a_hist)`，完成 `z_search` vs `μ(a)` vs ops/edges/path 的 Diagnostic B。
3. **完整 cycle audit**：补齐 `a→μ→a'_greedy` 与 drift 分层；报告 isomorphism-aware exact reconstruction、collision。
4. **matched prefix-only replay**：Zs、Ze、D、K、H 使用同 histories、prefixes、seeds、refit budget；top-region gate 为主。
5. **cache coverage audit**：统计 P1 的 greedy architectures 有多少可由现有 architecture+HP+seed 标签无偏复用。

P0 停止条件：若 Ze 或 D/K 已稳定通过 primary gate，直接据此选择 P1/P2 路线；若全部失败，先量化重复训练噪声，不修改正式 GP。

### P1 — 只改变 decoder/inference policy 的低成本因果实验

1. 实现/version 一个显式 `decoder_policy={historical_modal3, deterministic_greedy}` 开关，不更改 checkpoint。
2. 做 48 个 informative `z` 的 cache-first matched relabel；只训练 cache 未覆盖的 greedy architecture，固定 3 个 matched train seeds。
3. 重算真实 performance geometry 与 recommendation regret。
4. 只有 greedy performance 本身不差但存在 sequence-level选择问题时，才比较 constrained beam；不先调 temperature。

P1 放行到 tiny BO 的条件：decoder-dominant criterion 全部满足。否则不因复现性修复而宣称 BO 已修复。

### P2 — 成本可控的 representation retraining

1. 先做 frozen encoder + performance head，测 current `μ(a)` 的可预测上限。
2. 采用 Graph-VAE/NAO式 reconstruction+KL+performance regression 联合 fine-tune。
3. matched architecture split、greedy decoder、相同标签预算；比较 E0/E1/E2/E3。
4. 只有 held-out top-region gate、cycle/validity 与 reconstruction trade-off 同时可接受，才进入 search pilot。

### P3 — 替换 surrogate/architecture representation 或重跑正式 BO

按 P0/P2 证据选择一个，不并行全面重构：

- **P3-A**：architecture-kernel GP + HP additive/product kernel；
- **P3-B**：BANANAS-like path/adjacency predictor + discrete mutation；
- **P3-C**：performance-aware latent + deterministic/canonical decoder；
- **P3-D**：hybrid `μ(a)+structure+HP` surrogate。

先做单数据集、matched 5 seeds 的小预算 pilot；通过预注册 top-region/recommendation gate 后再进入跨数据集正式 BO。不要用 test accuracy 选择路线。

## 12. Recommended Implementation Directions

### Direction 1 — Separate identity from generation randomness

[计划] 为 decoder policy 建立版本化 API；同一正式实验只能使用一种 policy。候选 identity 以 decoded canonical architecture + HP 为准，同时保留原 `z_search`。formal history 必须保存 policy 与 fingerprint，避免一个连续坐标被误当作固定 architecture。

### Direction 2 — Add `z_enc` as a first-class diagnostic, not silently replace `z_search`

[计划] 每次 decode 后可选地 re-encode architecture 并记录 `μ(a)` 与 roundtrip drift。先作为只读 feature；未经 matched replay 不把现有 BO 输入直接替换为 `μ(a)`，因为 projection 会改变 candidate distribution 与 acquisition 语义。

### Direction 3 — Build a structure kernel baseline before a complex new encoder

[计划] 复用现有 canonical ops/edges/path signatures构造距离与 Gram matrix，显式检查 PSD/数值稳定性。先完成 offline replay；只有 top-region gate 通过才添加到 `surrogate/` 的实验实现。

### Direction 4 — Performance-aware fine-tuning with leakage controls

[计划] 新建隔离 experiment module，而不是改写正式 checkpoint。objective 与 split 写入 config；architecture fingerprint 分组；validation-only 选择 λ；每个 checkpoint 记录 source labels、decoder policy 与 split manifest。

### Direction 5 — Preserve the option to abandon latent BO

[推断] 如果 D/K 在 matched replay 与小预算 search 中稳定胜出，应转向 discrete architecture search，而不是为保留 VAE 而继续修补。BANANAS 与 NASBOT 证明 decoder-free、architecture-native NAS 是成熟方案；VAE 不应成为论文贡献的先验约束。

## 13. What Would Constitute a Publishable Finding?

以下任一条都可能形成论文级贡献，但需要相应证据闭环。

### Finding A — Decoder repair is necessary but insufficient

需要：跨多个 policy 的 reproducibility/validity证据、greedy matched relabel、`z_search`/`μ(a)`/architecture geometry 分解，以及至少一个第二数据集复现。可形成“latent NAS 中 generation stability 与 performance geometry 是独立轴”的机制发现。

### Finding B — Reconstruction-only coordinates are misaligned with GNN performance

需要：matched observation/surrogate test 证明结构或 performance-aware representation 在 top-region/recommendation metrics 稳定胜出；同时控制 decoder、HP、seed、budget。不能只展示 PCA/t-SNE 或 global correlation。

### Finding C — Performance-aware retraining recovers useful BO geometry

需要：E2/E3 在 held-out architecture 上改善 top ranking 与 regret，保持 validity/cycle；随后小预算 BO 改善 best-so-far AUC/最终 validation，并以 validation-only protocol选择候选。

### Finding D — Structure-aware discrete NAS is better for this GNN search space

需要：architecture kernel/path predictor在 matched replay 与实际 acquisition 中胜出；与 latent baseline 使用同 full-evaluation预算、同 seeds、同 candidate feasibility规则；跨至少两个 dataset 方向一致。

无论哪条路线，最低发表标准包括：

- 清晰分离文献、代码、实验与推断；
- 预注册 primary gate 与 stopping rule；
- search seed 与 train seed 分层；
- test 不参与选择；
- 报告 invalid、duplicate、collision、wall time 与 equivalent evaluation cost；
- 提供能回到原始 history/result JSON 的 provenance；
- 明确负结果：global predictor gain 不等于 top-region NAS gain。

## 14. Open Questions

1. VAE 的训练 architecture 分布与 Phase4 的有界 `z_search` candidate distribution 有多大 support mismatch？
2. `z_bound` 内多少体积远离 encoder posterior means？roundtrip drift 是否随半径单调增加？
3. 当前 architecture canonicalization 是否真正处理 graph isomorphism，还是只规范了 ops/edge tuple 顺序？
4. many-z→one greedy collision 是 decoder mode collapse、搜索空间本来较小，还是多个坐标落在同一 decision cell？
5. Cora top region 的 validation spread 很小；重复训练噪声占 `|Δaccuracy|` 的比例是多少？
6. performance geometry 是否必须条件于 HP？同一 architecture 在不同 lr/dropout/hidden/l2 下可能改变排序。
7. architecture representation 应是 dataset-specific、task-conditioned，还是存在可跨 Cora/CiteSeer/PubMed 迁移的共享 geometry？
8. `μ(a)` 如果比 `z_search` 好，应该在 encoded valid set 上做 discrete/nearest-neighbor search，还是学习 posterior-manifold projection？
9. greedy policy 虽稳定，但是否偏向少数 architecture modes，从而解释 collision 从 0.26 升到 0.43？
10. structure predictor 在 late-stage/NDCG 有益而 top region 无益，究竟是 feature 不足、label noise、样本量，还是 gate 已接近不可分辨上限？
11. current GP 的 default `SingleTaskGP` kernel 与高维 structure features 是否公平匹配？C1/C2 必须分别回答 representation isolation 与完整方法比较。
12. 如果 P2 只改善 predictor 而损害 decoder reconstruction，应选择 encoder-only representation + discrete search，而不是强求一个生成式 latent 同时承担所有目标吗？

---

## Final Priority Conclusion

```text
P0  复用现有数据：分离 z_search 与 μ(a)，补齐 cycle/alignment，做 Zs/Ze/D/K/H matched replay。
P1  cache-first 的 current-vs-greedy matched relabel，关闭 Track A 的历史标签因果缺口。
P2  采用 NAO/Graph-VAE 文献中的 reconstruction+KL+performance regression，做小规模 performance-aware fine-tune。
P3  依据 P0/P2 选择 performance-aware latent、structure-kernel、discrete mutation 或 hybrid，再做小预算正式 BO。
```

**当前推荐结论：** 不再把增加 stochastic decode 次数或直接修改 GP 当作第一修复。先完成 P0 的 `z_search`/`μ(a)` 分解与 P1 的真实 relabel；若 consistency 修复后 performance geometry 仍不改善，就应把主线转向 performance-aware representation 或 architecture-native surrogate，而不是继续维护 reconstruction-only latent 作为默认 BO 坐标。
