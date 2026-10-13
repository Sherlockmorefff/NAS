---
last_updated: 2026-08-14
status: current-audit
claims: [C001, C002, C003, C004, C005, C006, C008, C009, C010, C011]
evidence: [E001, E002, E003]
literature: []
---

# Consistency audit

## 已对齐项目

| 项目 | 统一口径 | 检查结果 |
| --- | --- | --- |
| representation | 12D `z_arch` + 4D `h_direct` = 16D `z_search` | 与 project overview、Method、glossary 对齐；HP VAE 不参与 Phase4 direct decode |
| architecture model | D-VAE，不是 VQ-VAE | 对齐 |
| methods | S0/global Schur；G100/G150 K=4 ordinary diagonal GMM + TED/LF；K6-LF 独立 | 对齐；禁止把正式 GMM 写成 WGMM estimator |
| full budgets | S0 50+250；G100 50+100+150；G150 50+150+100；均 300 | 对齐 |
| LF | G100/G150 20 epochs、实际 shortlist；K6 200×20 | 对齐；额外成本已披露 |
| surrogate/acquisition | scratch Exact GP + fixed-domain normalization + qLogEI | 对齐；DKL 非正式，KG 未实现 |
| search training | 150 epochs、patience 40 | 对齐 |
| Cora main final | Top-10×30、300 epochs、patience 80 | 对齐 |
| K6 final | Top-7×30、150 epochs、patience 40 | 对齐且分表 |
| statistics | method unit = search seed；SD `ddof=1`；df=4 t-CI | 对齐 |

## 配置与 provenance 冲突

1. **Seed authorization。** [`../../../configs/cross_dataset_methods.json`](../../../configs/cross_dataset_methods.json) 定义 development seeds 0–4 和 formal seeds 5–9；`training_mode_decisions.json` 仅授权 CiteSeer/PubMed/DBLP 0–4、Flickr seed0。Historical E003 已完成 5–9，但 current runner authorization 不能由历史结果替代。
2. **Resource decision 时间层。** `formal_search_seed_5_9_decision.json` 的 protocol `accuracy-diag_v1_20260813_bc2112d5` 先记录无 CUDA；后续真实 GPU preflights 对 CiteSeer/PubMed/DBLP pass、Flickr initial/retry OOM。审计必须按时间分别记录。
3. **Protocol ID vs run tag。** E003 的 `formal_search_20260731_8831cd0a5337` 和 E002 的 K6 tag 是 historical run tags，不等于 current preflight protocol。E001–E003 均没有可用的正式 protocol-id field。
4. **Git/source identity。** E002/E003 运行 commit 是 `d73c849…` 且 dirty；E001 Cora main search 无独立 run provenance，commit 未知。当前 HEAD `9df71a3…` 不能代填。
5. **Checkpoint path。** Historical provenance 使用 `results-wgmm1/...pth`；current reference path 是 `checkpoints/...pth`。[`../../../artifacts/reference/reference_manifest.json`](../../../artifacts/reference/reference_manifest.json) 核实两者 SHA-256 相同并记录迁移，正文应按 identity 而非路径误判为不同 checkpoint。
6. **Current knowledge-base date。** 原底稿保留 2026-08-13 的人工日期，新 frontmatter 记录 Paper Agent 更新为 2026-08-14；两者不冲突，但投稿冻结时应统一显示策略。

## 文件间数字检查

- `paper/tables/main_results.md` 只转写 `results/tables.md` 的 Cora 和 cross-dataset mean；E-cards 同值。
- `paper/tables/computational_cost.md` 的 LF ranges 与 results/initialization pages 一致。
- Abstract 未加入新数字；Introduction 原有 Cora 数值与 facts source 一致。
- Dataset statistics 保持 TBD，没有凭记忆填值。

## 未完成一致性检查

- TODO：在生成图表后核对 analysis script 的 invalid handling、budget indexing、CI 和 LF epoch aggregation。
- TODO：在新增跨数据集 final eval 前冻结 Top-K/replicates/epochs/patience；不能继承 K6 或 Cora 协议而不声明。
- TODO：引用补齐后统一 GNN operation、dataset 和 qLogEI/KG 首次展开。

当前高风险：historical completion 被写成 current reproduction；run tag 被写成 protocol ID；full budget fairness 被写成 total-cost fairness。
