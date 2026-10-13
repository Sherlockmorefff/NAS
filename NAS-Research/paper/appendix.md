---
last_updated: 2026-08-14
status: skeleton-with-verified-index
claims: [C001, C002, C003, C004, C005, C006, C007, C009, C010, C011]
evidence: [E001, E002, E003]
literature: []
---

# Appendix（living structure）

## A. 搜索空间

[已验证] `z_search = [z_arch, h_direct]`，维度 `12+4=16`。DAG 最多 5 个内部 operation nodes，另有 start/end；operations 为 GCN、GAT、GraphSAGE、GIN、Identity、GCNII。来源：[`../../nas_space.py`](../../nas_space.py)、[`../../models.py`](../../models.py)。

## B. 超参数范围

[已验证] `h_direct` 对应 learning rate log10 `[-4,-1.5]`、dropout `[0.1,0.6]`、hidden dimension `{16,32,64,128,256,512}`、L2 `{1e-5,5e-5,1e-4,5e-4,1e-3}`。来源：[`../../hp_modes.py`](../../hp_modes.py)、[`../../eval_utils.py`](../../eval_utils.py)。

## C. 数据集处理与 split

[已验证] Cora/CiteSeer/PubMed 使用 Planetoid public split + `NormalizeFeatures`；DBLP 使用 seed 0 构造的 20/class train、500 validation、1000 test split + normalization；Flickr 使用 official masks 且不额外 normalize。精确 identity 必须从每个 `dataset_manifest.json` 读取。来源：[`../experiments/cross_dataset.md`](../experiments/cross_dataset.md)。

## D. 完整实验配置

- S0/G100/G150：[`../../configs/cross_dataset_methods.json`](../../configs/cross_dataset_methods.json)。
- Cora/K6 和历史实际运行参数：E002/E003 链接的 `run_provenance.json` 与 initialization config。
- [已验证] `accuracy-diag_v1_20260813_bc2112d5` 出现在 current formal-seeds resource decision [`../../artifacts/reference/cross_dataset_v1_preflight/formal_search_seed_5_9_decision.json`](../../artifacts/reference/cross_dataset_v1_preflight/formal_search_seed_5_9_decision.json)，也被 launcher tests 使用。它标识 2026-08-13 的 current preflight/decision scope，不是 E001–E003 historical result artifacts 的 protocol ID；不得反向套用。

## E. Seed 体系

[已验证] search seed 是方法级独立单位。candidate evaluation seed 由 search seed、candidate fingerprint、fidelity 派生；final-evaluation seed 由 final base seed、candidate fingerprint、replicate index 派生。Cora development searches 用 0–4，历史 formal cross-dataset 用 5–9；不得混称。详见 [`glossary.md`](glossary.md)。

## F. 复现命令

[计划] TODO：投稿冻结时从 `run_provenance.json.argv` 和正式 runner 生成只读复现命令，替换绝对路径为参数占位符，并注明 GPU/资源要求。本轮不运行训练，也不把旧命令宣称为当前 HEAD 已验证命令。

## G. 额外结果

[已验证] 项目级完整结果索引在 [`../results/tables.md`](../results/tables.md)；论文候选视图在 [`tables/README.md`](tables/README.md)。本附录不复制未经核实的新数字。

## H. 统计方法

[已验证] 现有表以 5 个 search seeds 为单位，sample SD 使用 `ddof=1`，95% CI 使用 df=4 的双侧 t 区间。候选内 30 replicates 只描述训练噪声。TODO：预注册 paired difference、effect size、practical threshold 和 multiplicity handling。

## I. Surrogate diagnostics

[已验证] artifact 包含 prequential MAE/RMSE、95% coverage、Spearman、pairwise/ranking 和 top-region MAE。当前表是最后窗口汇总；TODO：补全随 budget 的曲线、top-region 定义和 invalid handling。

## J. 计算资源

[已验证] E002/E003 的 provenance 记录 NVIDIA GPU、driver、CUDA、dependency versions 与 runtime source manifest。历史 elapsed 不能作为跨硬件 benchmark。TODO：投稿时汇总每方法 GPU-hours、LF/full/acquisition 时间和失败率。

## K. Failure / blocked runs

[已验证] 历史 60-run search 的 18,000 full records 中有 3 invalid；具体位置见 E003。[已验证] Flickr 当前 formal seeds 的最新 resource preflight/retry OOM；历史完成不等于当前放行。[未知] 跨数据集 final evaluation 缺失而非已失败。

## L. Artifact 路径索引

- E001：[`evidence/E001.md`](evidence/E001.md) — Cora main search/final roots。
- E002：[`evidence/E002.md`](evidence/E002.md) — K6 search/final roots及运行 commit。
- E003：[`evidence/E003.md`](evidence/E003.md) — historical 60-run root、migration、current gate。
- 图源：[`../results/figures/README.md`](../results/figures/README.md)。

## M. Reproducibility checklist

- [x] 区分 architecture latent 与 direct HP coordinates。
- [x] 记录 candidate/dataset/split/checkpoint/source identity 机制。
- [x] 区分 search、evaluation、final-replicate seeds。
- [x] 区分 search validation、final validation、final test。
- [x] 记录 full 与 LF budget，声明 K6 protocol mismatch。
- [ ] 冻结所有正文使用的 protocol IDs / run tags / commits。
- [ ] 提供可执行 reproduction commands 与环境锁定。
- [ ] 补齐 paired statistics、effect size、multiplicity 和分析脚本 identity。
- [ ] 补齐跨数据集 final evaluation 或明确投稿时仍缺失。
- [ ] 完成 citation audit 与数据集许可/引用。

当前可写：已核实配置和 artifact 索引。当前禁止：复制未核实数字、提供未执行却称已验证的复现命令、把计划 protocol 当现有结果来源。
