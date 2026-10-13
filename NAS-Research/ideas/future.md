最后更新日期：2026-08-13
当前状态：[计划] 按论文结论风险排序；仅列当前证据明确暴露的缺口与 ablation。

# 后续研究与执行优先级

## P0：补齐会改变论文结论的证据

### 跨数据集 final evaluation

[计划] 为 CiteSeer、PubMed、DBLP 的 S0/G100/G150 seeds 5–9 冻结同一 Top-K、30-replicate、epochs、patience 与 validation-only selection 协议。先修正 formal-seed 授权冲突，再在远端 GPU 执行。输出必须含 dataset/split/checkpoint/source identity 和 per-candidate replicate summary。

[计划] Flickr 单独处理：最新预检在 seeds 5–9 及 retry1 均 OOM。需要预先决定维持 blocked，或采用明确且所有方法一致的新资源协议；不能静默降级后并入原表。

### Cost-aware initialization curves

[计划] 从现有 histories/run summaries 生成 full-only、equivalent-full 和 wall-clock 三种 best-so-far 曲线，并预注册 AUC/evaluations-to-target。该工作不需重新训练，却会直接决定能否声称 sample efficiency。

## P1：隔离初始化机制

[计划] 在同一候选池、K、clustering feature、selector、shortlist、promotion score、full/LF 预算和 final-evaluation 协议下逐项消融：

1. global vs clustered allocation；
2. 12D architecture clustering vs 16D joint clustering；
3. TED vs Schur selector；
4. K=4 vs K=6；
5. LF-only、GP-only 与混合 promotion score。

[已验证] 当前 G100/G150/K6 同时改变多个因素，现有差异不能归因于单一机制。

### Metric-learning ablation

[计划] 冻结 VAE 与候选池，只改变用于选择/GP 的 metric，并用 prequential diagnostics 和最终 search 指标评价。避免重新训练 representation 导致 source identity 与候选域同时变化。

## P2：Acquisition 与 surrogate 扩展

### Knowledge Gradient

[计划] 实现 KG adapter、CPU synthetic smoke test 和与 qLogEI/random 的固定初始化对照；计入 acquisition optimization overhead。完整设计见 [KG.md](../experiments/KG.md)。

### DKL 正式化条件

[已实现未充分验证] DKL 已有代码和测试，但不是正式 S0/G100/G150 方法。[计划] 若升级为论文对照，必须冻结 architecture、training schedule、normalization、refit budget 和 seed，先验证 uncertainty calibration，再执行同预算 search。

## P3：扩大数据规模

[已验证] ogbn-arxiv 当前因 full-batch 资源需求被 blocked。[计划] 若采用 neighbor sampling 或其他训练模式，应创建新 protocol identity，并先验证 ranking/accuracy 与原 full-batch 定义的可比性；不能覆盖当前 blocked 记录。

## 完成定义

[计划] 每项完成需同时具备：冻结配置、明确 seeds、原始 histories/summaries、成本、invalid 记录、provenance、统计脚本输出和论文表/图链接。只有相应判定标准满足后，才从 Introduction 的条件式贡献升级为确定主张。
