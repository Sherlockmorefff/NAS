---
last_updated: 2026-08-14
status: living-draft
claims: [C001, C002, C004, C005, C008, C009, C011]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

最后更新日期：2026-08-13
当前状态：[已验证] 方法描述与当前实现、正式配置和 checkpoint 形状一致；KG 不属于本文已实现方法。

# Method（论文底稿）

## Problem formulation

[已验证] 给定数据集、固定 split 与评估预算，我们在 GNN DAG 架构和训练超参数的联合域上最大化 validation accuracy。一次 full-fidelity observation 是候选架构/HP 在固定 seed 派生规则、epoch 上限和 early-stopping 规则下完成的 GNN 训练；test split 不参与搜索。

记联合搜索坐标为

\[
x = [z_{\mathrm{arch}}, h_{\mathrm{direct}}] \in \mathbb{R}^{12+4}.
\]

[已验证] `z_arch` 是 D-VAE 的 12 维架构 latent；`h_direct` 是四个归一化直接搜索坐标。虽然 checkpoint 中还包含一个 4D HP VAE，Phase4 evaluation 直接通过 `decode_hp_by_mode()` 解释 `h_direct`，没有调用 `hp_vae.decode()`。因此全文不得把 `x` 整体称作 VAE latent。术语权威定义见 [项目总览](../project_overview.md#联合搜索空间与术语边界)。

## Architecture and hyperparameter decoding

[已验证] D-VAE decoder 将 `z_arch` 解码为最多 5 个内部操作节点的 DAG；候选操作为 GCN、GAT、GraphSAGE、GIN、Identity 和 GCNII。start/end 节点不计入这 5 个操作节点。

[已验证] `global4` 中四个 HP 坐标依次控制 learning rate、dropout、hidden dimension 和 L2：learning rate 在 log10 `[-4,-1.5]` 上映射，dropout 在 `[0.1,0.6]` 上映射，hidden/L2 映射到固定离散集合。操作条件参数保留默认/条件解码逻辑，不增加搜索维度。

证据：[nas_space.py](../../nas_space.py)、[hp_modes.py](../../hp_modes.py)、[eval_utils.py](../../eval_utils.py)、[正式 checkpoint](../../checkpoints/joint_model_pipeline_global4_best.pth) 与 [checkpoint manifest](../../artifacts/reference/reference_manifest.json)。

## Candidate domain and reproducibility

[已验证] 正式方法共享由 LHS 生成的 768-candidate pool、`z_bound=2.5` 与 architecture scale `lhs_sigma_arch=0.8`。candidate fingerprint 标识解码后的架构与 HP；evaluation seed 由 search seed、candidate fingerprint 和 fidelity context 派生。dataset、split、checkpoint 与 source identity 参与 run attribution。

[已验证] 这些 identity 机制用于避免跨数据集 cache 混用、保证 low/full fidelity seed 归属和复核候选选择，不等同于额外模型组件。

## Initialization strategies

### Global Schur baseline (S0)

[已验证] S0 在完整 16D fixed-domain-normalized coordinates 上对候选池进行全局 Schur 选择，获得 50 个 full-fidelity seed；随后运行 250 个 online evaluations。

### Clustered TED with low-fidelity promotion (G100/G150)

[已验证] G100/G150 在候选池上拟合 K=4 普通对角 GMM。历史接口名虽含 `WGMM`，正式 `gmm_fit_pool` source 使用 uniform candidate weights。hard assignment 将候选划分到簇，hybrid quota 以 0.5 equal 与 0.5 responsibility mass 组合。

[已验证] 簇内 Transductive Experimental Design 使用 fixed-domain normalization 后的完整 16D coordinates 和 RBF kernel；lengthscale 为 4.0、regularization 为 0.1、jitter 为 `1e-8`，每簇 shortlist 上限为 100。shortlist 候选运行 20 epochs 且不 early-stop。promotion score 为

\[
s(x)=0.50\,s_{\mathrm{LF}}(x)+0.25\,s_{\mu}(x)+0.25\,s_{\sigma}(x),
\]

其中各项按实现的归一化规则组合。G100/G150 分别提升 100/150 个 full candidates，然后执行 150/100 次 online qLogEI。

### Fixed-K6 clustered Schur low fidelity (K6-LF)

[已验证] K6-LF 是独立实验协议：在 12D architecture coordinates 上拟合 K=6 普通对角 GMM，5 restarts；簇内使用共享 RBF kernel 的 Schur selector，在 200 个 low-fidelity candidates 中提升 100 个，然后运行 150 次 qLogEI。它不是仅把 G100 的 K 从 4 改成 6，因为 clustering feature、selector 和 shortlist 也不同。

完整预算与混杂因素见 [初始化实验](../experiments/initialization.md)。实现证据：[initialization_wgmm_ted.py](../../initialization_wgmm_ted.py)、[initialization_gmm_schur.py](../../initialization_gmm_schur.py)、[正式配置](../../configs/cross_dataset_methods.json)。

## Accuracy surrogate

[已验证] 正式 S0/G100/G150 使用 scratch Exact GP，而非 DKL。输入先按预先固定的候选域边界归一化；模型使用 `SingleTaskGP`，以观测 validation accuracy 拟合。online update 采用 warm refit，每 5 次重新优化 20 steps。

[已验证] prequential diagnostics 包括 window MAE/RMSE、95% interval coverage、Spearman 和 top-region MAE。DKL surrogate 有独立实现与测试，但不进入正式方法结论。

证据：[accuracy_gp.py](../../surrogate/accuracy_gp.py)、[dkl_accuracy_gp.py](../../surrogate/dkl_accuracy_gp.py)。

## Online Bayesian optimization

[已验证] 正式 acquisition 是 qLog Expected Improvement（qLogEI）。每次 acquisition optimization 使用 8 restarts、256 raw samples，并考虑 128 个额外候选。随机在线策略作为已实现基线存在，但不属于当前 S0/G100/G150 结果表。

[未知] 仓库内没有 Knowledge Gradient 实现；KG 仅作为未来对照，不属于本方法。

## Budgets and cost accounting

| 方法 | full 初始化 | online full | full 总数 | 额外 low-fidelity |
| --- | ---: | ---: | ---: | ---: |
| S0 | 50 | 250 | 300 | 0 |
| G100 | 150 | 150 | 300 | 每簇至多 100×20 epochs；按 run 实际值核算 |
| G150 | 200 | 100 | 300 | 每簇至多 100×20 epochs；按 run 实际值核算 |
| K6-LF | 150 | 150 | 300 | 200×20 epochs = 26.67 eq-full |

[已验证] G100/G150 的 LF shortlist 大小不是固定 207：Cora seeds 0–4 实测为 207–356 个，历史跨数据集 seeds 5–9 为 259–400 个。所有样本效率结果必须至少报告 full-evaluation 与 equivalent-full 两个横轴，并辅以 wall time。`eq-full = low_fidelity_actual_epochs / 150` 只是一致成本换算，不能替代实测时间。

## Final candidate selection

[已验证] 搜索阶段只使用 validation。final evaluation 从每个 search seed 的 Top-K search-validation candidates 开始，以独立 replicate seeds 重新训练；候选按 final-validation mean 选中，最后才读取其 test distribution。test 既不用于 selection，也不用于 tie-breaking、retry 或 seed derivation。

[已验证] 主 Cora 协议使用 Top-10、30 replicates、300 epochs、patience 80；K6 协议使用 Top-7、30 replicates、150 epochs、patience 40。两者必须分表。

## Algorithm steps and Claim boundary

```text
Input: fixed candidate pool, checkpoint, dataset/split identity, search seed
1. Decode candidate identity and derive candidate-evaluation seeds.
2. Select 50 full-fidelity seeds with S0, G100/G150, or K6-LF rules.
3. For LF methods, evaluate the registered shortlist at 20 epochs and compute promotion scores.
4. Promote the configured number of candidates to full fidelity.
5. Fit/refit the scratch Exact GP on valid full observations.
6. Propose unevaluated candidates with qLogEI and perform full evaluation until 300 full observations.
7. Save history, diagnostics, budget summary, RNG/provenance and completion state.
8. In a separate final-evaluation stage, retrain Top-K candidates; select by final validation and only then report test.
```

| 组件 | 代码模块 | Claims | 实现状态 | 实验验证边界 |
| --- | --- | --- | --- | --- |
| joint representation/decoding | [`../../nas_space.py`](../../nas_space.py)、[`../../hp_modes.py`](../../hp_modes.py)、[`../../eval_utils.py`](../../eval_utils.py) | C012 | [已验证] 实现映射 | 不证明 representation 优于替代方案 |
| global Schur | [`../../bo_phase4.py`](../../bo_phase4.py) | C001/C004 | [已验证] 已运行 | coverage/sample efficiency 仍需统一分析 |
| clustered TED + LF | [`../../initialization_wgmm_ted.py`](../../initialization_wgmm_ted.py) | C001/C004/C005/C009 | [已验证] 已运行 | 无 matched mechanism ablation |
| K6 cluster-wise Schur + LF | [`../../initialization_gmm_schur.py`](../../initialization_gmm_schur.py) | C004/C005/C009 | [已验证] 已运行 | 独立 final protocol，不可直接排名 |
| Exact GP | [`../../surrogate/accuracy_gp.py`](../../surrogate/accuracy_gp.py) | C002 | [已验证] 正式运行 | diagnostics 无一致支配，非性能证明 |
| qLogEI acquisition | [`../../bo_phase4.py`](../../bo_phase4.py) | C008 | [已验证] 正式运行 | 没有 KG matched comparison |
| KG | [`../experiments/KG.md`](../experiments/KG.md) | C008 | [未知] | 无实现、配置、artifact |
| final selection | [`../../final_eval.py`](../../final_eval.py) | C011 | [已验证] Cora | 跨数据集 final evaluation 缺失 |

## Computational complexity and measured cost boundary

[已实现未充分验证] Selector 的精确渐进复杂度取决于 shortlist、cluster capacity 和 kernel linear-algebra implementation；当前知识库未核实可投稿的闭式复杂度，故保留 TODO：从实际实现逐函数核对 time/memory complexity。现阶段只报告可直接核实的工作量：S0 无 LF；G100/G150 的 LF candidates 随 seed 为 207–400；K6 固定 200。任何复杂度陈述必须同时报告候选池 768、kernel/cluster 设置和实际 wall time。

## Acquisition interface

[已验证] `online_candidate_strategy=qlogei` 是正式配置。接口必须在未知 acquisition 名称时清晰失败；论文不得暗示当前代码会自动回退。KG 若实现，必须只替换 acquisition，并冻结 surrogate、initialization、candidate domain、full budget、seed 和 final protocol；详见 [`../experiments/KG.md`](../experiments/KG.md)。

当前可写：算法组件、接口、预算、selection protocol。当前禁止：从实现推出 superiority、为未核实复杂度填写公式、把 KG 写成已实现。
