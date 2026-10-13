---
last_updated: 2026-08-14
status: controlled-vocabulary
claims: [C001, C002, C004, C005, C008, C009, C011]
evidence: [E001, E002, E003]
literature: [D-VAE, TED]
---

# Glossary

全文首次出现按“首次展开”列书写，之后使用“标准写法”。代码/配置是实现术语的来源，不能替代理论引用。

| 标准写法 | 简短定义 | 不推荐写法 | 首次展开 | 来源代码或配置 |
| --- | --- | --- | --- | --- |
| architecture D-VAE | 对 DAG architecture 编码/解码的 variational autoencoder | VQ-VAE；把完整 16D 都称为 D-VAE latent | directed-acyclic-graph variational autoencoder (D-VAE) | [`../../nas_space.py`](../../nas_space.py)、[`../../models.py`](../../models.py)、checkpoint manifest |
| `z_arch` | 12D architecture D-VAE continuous latent | architecture embedding（未说明来源）；16D VAE latent | 12-dimensional architecture latent, `z_arch` | [`../../nas_space.py`](../../nas_space.py)、正式 checkpoint |
| HP VAE / `z_hp` | checkpoint 模型中独立 HP VAE 的 4D latent；Phase4 `global4` evaluation 不使用其 decoder | 将 Phase4 后 4D 坐标称为 `z_hp` | 4-dimensional HP-VAE latent (model component only) | [`../../nas_space.py`](../../nas_space.py)、[`../../eval_utils.py`](../../eval_utils.py) |
| `h_direct` | Phase4 中 4D normalized HP search coordinates，直接解码为 lr/dropout/hidden/L2 | `z_hp`；HP VAE latent | four direct normalized hyperparameter coordinates, `h_direct` | [`../../hp_modes.py`](../../hp_modes.py)、[`../../eval_utils.py`](../../eval_utils.py) |
| `z_search` | 联合搜索向量 `[z_arch, h_direct]`，共 16D | full VAE latent；joint VAE code | 16-dimensional joint search coordinate, `z_search` | [`../../bo_phase4.py`](../../bo_phase4.py) |
| S0 | 50 个 global Schur full seeds + 250 qLogEI full，300 full total | random initialization；zero initialization | global Schur baseline (S0) | [`../../configs/cross_dataset_methods.json`](../../configs/cross_dataset_methods.json) |
| G100 | K=4 GMM + clustered TED/LF，提升 100 full，随后 150 qLogEI full | WGMM-100（正式 source 并非 weighted GMM） | clustered TED low-fidelity initialization with 100 promotions (G100) | 同上；[`../../initialization_wgmm_ted.py`](../../initialization_wgmm_ted.py) |
| G150 | 同 G100，但提升 150 full，随后 100 qLogEI full | WGMM-150 | clustered TED low-fidelity initialization with 150 promotions (G150) | 同上 |
| K6-LF | K=6 architecture-only GMM + cluster-wise Schur + 200 LF/100 promotions | G100 with K=6（其他组件也不同） | fixed-K6 clustered Schur low-fidelity initialization (K6-LF) | [`../../initialization_gmm_schur.py`](../../initialization_gmm_schur.py)、E002 |
| global Schur | 在完整 normalized joint coordinates 上的全局 Schur selection | random Schur；DPP（除非有理论证明） | global Schur-complement selection | [`../../bo_phase4.py`](../../bo_phase4.py) |
| diagonal GMM | uniform candidate weights 的普通对角 Gaussian mixture model | WGMM（正式 G100/G150 与 K6 source 均非 weighted） | diagonal Gaussian mixture model (GMM) | [`../../configs/cross_dataset_methods.json`](../../configs/cross_dataset_methods.json)、初始化配置 |
| WGMM | weighted GMM 的概念/历史接口名；当前正式 `gmm_fit_pool` 语义为普通 GMM | 用 WGMM 指称当前正式 estimator | 仅在讨论历史接口时展开，并立即说明 estimator semantics | [`../../initialization_wgmm_ted.py`](../../initialization_wgmm_ted.py) |
| TED | G100/G150 的 cluster-wise transductive experimental design-inspired selector | exact reproduction of TED paper | transductive experimental design (TED)-inspired selection | [`../../docs/wgmm_clustered_ted.md`](../../docs/wgmm_clustered_ted.md)、BO 文献卡 |
| cluster-wise Schur | K6-LF 中各 cluster 配额内用共享 RBF kernel 的 Schur selector | TED；global Schur | cluster-wise Schur-complement selection | [`../../initialization_gmm_schur.py`](../../initialization_gmm_schur.py) |
| low-fidelity (LF) evaluation | 20 epochs、patience 0 的 shortlist candidate evaluation | cheap evaluation（不披露预算）；proxy test | low-fidelity (LF) evaluation | 配置、E001/E002/E003 |
| promotion | 根据 LF/GP score 把 shortlist candidate 升级为 full evaluation | final selection；test promotion | low-to-full-fidelity promotion | [`../../initialization_wgmm_ted.py`](../../initialization_wgmm_ted.py) |
| online BO | 初始化完成后的 sequential candidate proposal/evaluation | final evaluation | online Bayesian optimization (BO) | [`../../bo_phase4.py`](../../bo_phase4.py) |
| qLogEI | 正式 online acquisition，log-space numerically robust expected-improvement variant | EI；KG | logarithmic expected improvement acquisition (qLogEI) | [`../../bo_phase4.py`](../../bo_phase4.py)、正式配置；文献 TODO |
| KG | 尚未实现/运行的 Knowledge Gradient 对照 | 当前方法；已完成 baseline | Knowledge Gradient (KG), a planned acquisition baseline | [`../experiments/KG.md`](../experiments/KG.md) |
| scratch Exact GP | 从当前 run observations 拟合的 `SingleTaskGP` 型 accuracy surrogate | DKL；offline GP | exact Gaussian-process (GP) surrogate fitted from scratch | [`../../surrogate/accuracy_gp.py`](../../surrogate/accuracy_gp.py) |
| search seed | 决定候选域/搜索随机过程及 evaluation seed 派生上下文的独立搜索编号 | training seed；replicate | independent search seed | run summary/provenance |
| evaluation seed | 由 search seed、candidate fingerprint 和 fidelity context 派生的候选训练 seed | search seed | candidate-evaluation seed | run summary/provenance |
| final-evaluation seed | 由 final base seed、candidate fingerprint 和 replicate index 派生 | test seed（含义不精确） | final-evaluation replicate seed | `final_eval_summary_global4.json` |
| training replicate | 同一候选的独立 final retraining 单位；不是独立 search | evaluation seed count = search count | final-evaluation training replicate | [`../../final_eval.py`](../../final_eval.py) |
| test replicate | 不推荐单独使用；如出现应指 final-evaluation training replicate 中读取的 test metric | 独立 test search | 不使用；改写为 training replicate | final summaries |
| candidate | 一个待评估的 joint search coordinate 及其解码配置 | architecture（忽略 HP） | candidate configuration | [`../../bo_phase4.py`](../../bo_phase4.py) |
| architecture | 解码后的 GNN DAG，仅是 candidate 的架构部分 | configuration | GNN architecture | [`../../models.py`](../../models.py) |
| configuration | 架构、HP、fidelity 和训练协议合成的可执行设置 | candidate fingerprint | architecture–hyperparameter configuration | evaluation code |
| candidate fingerprint | 解码后架构/HP 的稳定 identity，用于 seed/cache/provenance | random ID；architecture hash | candidate fingerprint | [`../../eval_utils.py`](../../eval_utils.py)、run provenance |
| search validation | search history 的 `val_acc`，用于 BO/Top-K | accuracy；test | search-validation accuracy | histories |
| final validation | final retraining replicates 的 validation 汇总，用于选最终候选 | final test | final-evaluation validation accuracy | final summaries |
| final test | validation-only selection 完成后读取的 test 分布 | validation；search accuracy | final test accuracy | final summaries |

## 数值与命名规则

- latent dimension 必须写成 `12 + 4 = 16`，并说明后 4D 是 `h_direct`，不是 Phase4 解码的 HP-VAE latent。
- 方法首次出现同时给简称和 artifact/config label；后续统一使用 S0/G100/G150/K6-LF。
- 指标必须带 evidence tier：search validation、final validation 或 final test；不单写“accuracy”。
- seed 必须带角色：search seed、candidate evaluation seed 或 final-evaluation replicate seed。

当前可写：上述实现映射。当前禁止：把 D-VAE 写成 VQ-VAE、把 GMM 写成 WGMM、把 qLogEI 写成 KG、把 candidate 等同于 architecture。
