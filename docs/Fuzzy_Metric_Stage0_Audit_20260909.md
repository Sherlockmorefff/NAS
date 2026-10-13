# 模糊邻域交替搜索：阶段 0 实际核对报告

核对日期：2026-09-09。状态：阶段 0 的阶段 1 工程运行必要条件已满足；未开展阶段 2–6。以下服务器路径均为 `/home/fangxy/D-VAE` 下的实际文件，行号以本轮工作区为准。

## 1. 主机、工作区与原文

- 工具最初运行在本地 `SherlockmoredeMacBook-Air.local`，随后通过 SSH 别名 `ismc36-ts` 连接实际服务器 `ismc36`。
- 服务器登录目录 `/home/fangxy`；项目及 Git 根目录均为 `/home/fangxy/D-VAE`。
- 分支 `feature/accuracy-bottleneck-diagnostics`；完整 HEAD `64a44c14316f11debaf01cf8583426acfe1410e6`。
- 起点包括已修改的 `bo_phase4.py`、`eval_utils.py` 和大量未跟踪分析、配置、测试文件；没有拉取、切分支、清理、提交或推送。起点状态、已有差异和 1230 个已跟踪/未忽略文件的哈希保存在本轮结果目录 `workspace_before.json`、`git_status_before.txt`、`git_diff_before.patch`。
- 已读取服务器 `AGENTS.md`、`README.md`、`docs/repository_layout.md`、`docs/experiment_artifact_guide.md`、`docs/cross_dataset_protocol.md`、`checkpoints/README.md`。服务器未找到更深层 AGENTS.md。原规则中不修改已有 results/logs 的要求通过只创建独立新实验目录遵守；本轮用户明确授权服务器修改和新结果。
- 已先查当前本地项目及服务器整个项目的 Markdown（含 docs）；两份目标原文均缺失。当前本地镜像仅有只读 `sources/GitHub代码地址.txt`；未修改 sources。
- 之后完整读取用户指定原文并复制到服务器 docs。方案原文件名在 `.md` 前有一个空格；服务器副本仅规范文件名，字节未改。

| 原文绝对路径 | 输入版本 | SHA-256 | 服务器副本 |
|---|---|---|---|
| `/Users/sherlockmore/Downloads/Fuzzy_Metric_Alternating_Search_Plan_Revised .md` | 文内日期 2026-09-08，23579 字节 | `4d3514c1b0219aaaf5b0effe3ba218f40c7569a1d15f6e150cf32864bacf8cd6` | `/home/fangxy/D-VAE/docs/Fuzzy_Metric_Alternating_Search_Plan_Revised.md` |
| `/Users/sherlockmore/Downloads/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md` | 文内日期 2026-09-09，14478 字节 | `945db15c79197ef49124198c491306e236c435ee922a496802f64fc194c35fb0` | `/home/fangxy/D-VAE/docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md`，本轮修订工作版 |

输入版本详情见 `results/posthoc/fuzzy-metric-stage01-20260909-v1/input_documents.json`。原 Stage_Tasks 与方案另存本轮结果目录 `input_originals/`，任务修订差异单独保存。

## 2. 实际环境

正式验证使用 `/home/fangxy/.conda/envs/nas/bin/python`，Python 3.10.20，Linux x86_64。默认 `/usr/bin/python` 是 3.9.23，未拿它运行新实验。

| 依赖 | 实际版本 |
|---|---|
| PyTorch / CUDA runtime | 2.5.1+cu121 / 12.1 |
| torch_geometric | 2.7.0 |
| gpytorch / botorch | 1.15.2 / 0.16.1 |
| numpy / scipy / sklearn | 2.2.6 / 1.15.3 / 1.7.2 |
| igraph / pytest / PyYAML | 1.0.0 / 9.1.1 / 6.0.3 |

两张 NVIDIA RTX 4500 Ada Generation，各 24570 MiB，驱动 595.45.04；`torch.cuda.is_available()` 为真，已实跑小张量 CUDA 运算成功。阶段 1 编码、映射、GP 和测试使用 CPU 单线程；没有运行候选 GNN，也没有下载数据。

## 3. 候选与同配方成绩

本轮唯一历史来源：`/home/fangxy/D-VAE/results/search/high-regime-prior-model-blind-validation-global4-v1/` 的 `phase_b/initial/gpu0/seed_results.jsonl` 和 `phase_b/initial/gpu1/seed_results.jsonl`。

| 项目 | 已验证数量/规则 |
|---|---|
| 原始记录 | 1008；两文件各 504 |
| 可查询完整候选 | 336 |
| 架构 | 192；位置敏感，操作和排序后的边定义指纹 |
| 原始子池 | 24；本轮合并为共同固定开发候选池 |
| 每候选训练种子 | 恰好 0、1、2；各 336 条 |
| 重复候选/种子、缺失、失败、非有限成绩、重试 | 均为 0 |
| 查询目标 | 同一配置三种子 validation accuracy 的均值 |
| 配方 | Cora / public / NormalizeFeatures / full_batch / global4 / normalization none / 最大 300 轮 / patience 80 |
| 数据内容指纹 | `373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7` |
| 划分指纹 | `bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09` |
| 测试边界 | preregistration 明确 `track_test=false`，两成绩文件没有递归 test 命名字段 |

`candidate_id` 使用现有 `joint_config_fingerprint`（聚合后名 `joint_fingerprint`），由架构指纹和 HP 指纹共同确定。保存 operations、edges、原始归一化 HP 向量及解码后的 lr、dropout、hidden_dim、l2；固定条件还包括 gcnii_alpha=0.1、gcnii_theta=0.5（源评估器 run_high_regime_prior_model_blind_validation_labels.py:243 的固定值），以及 gat_heads=1、sage_aggr=mean、gin_eps=0、各按层字段=null、hp_mode=global4、normalization=none。配置和成绩完全分开，通过 ID 字典取配置，不从新表示反解。

可复用 `analyse/joint_configuration_metric_gate.py:299 load_and_aggregate_source`：校验原始 manifest 哈希、配方/数据/划分、三种子、重复、有限值、架构和 HP 身份，拒绝测试字段。全部聚合成绩仅由准备/揭示边界读取；模型收到白名单候选、初始 50 个标签以及每次已选择的 5 个标签。

其他已发现来源不混入：76 配置/60 架构/2130 条旧记录的 no_test 池使用 15 或 30 种子；旧 150/40 单种子搜索配方与本轮 300/80 不同；phase_c 的 30 种子稳定性结果属于其他用途。详见本轮 `data_audit.json` 和 `data_audit.md`。数字 1008、2130 均为已有记录，不是本轮训练次数。

## 4. 真实调用关系与标签审计

| 环节 | 已验证实现 | 可复用范围与缺口 |
|---|---|---|
| 确定性 VAE 编码 | `architecture_diagnostics.py:177 load_joint_vae` → `:132 encode_architecture_mu` → `:106 config_to_igraph` → `arch_vae.encode` 的均值 | 12 维架构编码可复用；显式冻结所有参数并按架构缓存 |
| 联合候选身份 | `analyse/b4_guided_search_accuracy_gate.py:726 _candidate_row`；`joint_configuration_metric_gate.py:299` 聚合 | 现有指纹和完整配置可复用；新增稳定 candidate_id 接口 |
| B4 | `joint_configuration_metric_gate.py:860 _joint_model_classes` / `:1193 train_direct_joint_regressor` | 双向图编码器＋HP 回归器，使用 historical-336 标签及内部验证早停；禁止加载其归档权重/预测来做本轮隐藏回放 |
| GP | `joint_configuration_metric_gate.py:1341 exact_matern_gp_predict` | Matérn 5/2、SE²＋学习噪声下限；原函数只有均值，无方差和可恢复模型，需要小型适配 |
| 采集评分 | `bo_phase4.py:1609 optimize_acq`、`:2825 _score_logei`；`surrogate/accuracy_gp.py:778 make_logei` | 复用现有 `StableQLogExpectedImprovement`，固定采样种子，离散候选逐个 q=1 评分 |
| P2 | `analyse/b4_arch_diversity_prospective_search_accuracy_gate.py:979 select_search_plans`，`:1005–1023` | 分数降序、同分按联合指纹；同批架构唯一，新增搜索内同架构最多两次；原网格和 batch=4 硬编码，需批量 5 的小适配 |
| 真实候选 GNN | `eval_utils.py:861 train_and_eval_arch` | 输入显式架构/HP，训练梯度和 validation 早停，默认 track_test=false；本轮不调用 |
| 数据边界 | `dataset_utils.py`；`analyse/m1_m2_metric_guided_search_accuracy_gate.py:1304 validation_only_data_view` / `:1322 load_validation_only_dataset` | 未来真训练可复用；本轮只读历史验证字段 |
| 原恢复 | `bo_phase4.py:3537 _load_resume_full_history`、`:5544` 成绩/RNG 保存、`:6220 _save` | 原潜向量身份不够支持新表示版本；新增原子保存的模型、GP、候选、揭示顺序和预算 |
| 原历史回放 | `analyse/direct_head_chronological_offline_replay_v2.py` | 只重放冻结预测排序，不重构 GP，不等于交替搜索已实现 |

冻结检查点为 `/home/fangxy/D-VAE/checkpoints/joint_model_pipeline_global4_best.pth`，SHA-256 `e5c8579fbbc78ce166b7f6bcd7718eefbbd5dbaa3c1f556b834b3f84759844d4`，与 `artifacts/reference/reference_manifest.json` 一致。旧日志 `legacy_artifacts/pre_20260813/logs-wgmm1/train_joint/train_joint/train_pipeline_0612_0609.json` 记载 mini_gnn_dataset_global4.pkl、100 轮、seed 42；对应 .log 明确输出该 best 文件。

当前生成与训练代码支持标签无关：`legacy/phased_cora_pipeline/generate_mini_data.py:73/:101/:214` 随机生成 HP/架构并设 seed42；`train_joint.py:273` 读取 types/adj/hp 三元组，`:289–295` 随机90/10划分，`:62 compute_loss` 只有重建与 KL，`:364` 按无监督 val_loss 保存最佳检查点。旧日志没有记录当时 pkl 的哈希，历史输入字节绑定仍有溯源限制；不声称完整证明原文件每个字节。VAE 可能见过候选池的无标签架构，本方案允许；不加载性能监督 VAE 变体。

新映射随机初始化，初始输入标准化仅拟合初始 50 个配置并固定；每次回归标签标准化、邻域差、配对种子标准误及 GP 只用当前已揭示记录。邻域始终用原始 0–1 accuracy；固定 100 步训练，无早停，也无隐藏成绩参数选择。整个336池已用于旧方法开发，本次用途统一标为开发；独立确认集合为空，不能把换随机种子视为独立确认。

### 旧缓存与随机种子规则的补充核对

`bo_phase4.py:702 candidate_evaluation_seed` 用 search_seed、candidate_fingerprint、full/low 保真度，经 `eval_utils.py:167 stable_seed` 派生训练种子，不依赖方法、选择顺序或阶段；LayerNorm分支使用匹配训练种子，不能混作本轮No-LN记录。当前336池由显式种子0/1/2评估，不重新用回放搜索种子派生训练种子。

`dataset_utils.py:734 evaluation_cache_key` 包含候选指纹、数据上下文（数据/划分指纹、变换、训练模式、依赖版本等）、保真度和评估种子；`bo_phase4.py:1091` 保存该键。**它没有包含最大训练轮数、patience或完整训练配方版本**，因此不能仅凭这个旧键跨配方复用成绩。本轮不使用旧键寻找缓存命中，而由原文件哈希及唯一300/80配方、种子、候选身份共同核验来源，再通过独立历史成绩库按ID查询。真实训练阶段前应补齐配方身份校验；本轮保留旧代码与既有修改，不顺手改变旧主线行为。

## 5. 任务要求—现有实现—缺口—拟修改文件

| 任务要求 | 现有实现 | 缺口 | 本轮文件 |
|---|---|---|---|
| 冻结 VAE 并缓存 | load_joint_vae / encode_architecture_mu | 缓存身份、冻结审计 | 新 `analyse/fuzzy_metric_replay.py` |
| 架构＋HP＋预测头 | 旧 B4 容量和输入不同 | 12维 z＋HP4 → 16维HP → 32/32 → 8维联合表示＋线性头 | 新 `surrogate/fuzzy_metric.py` |
| N0/N1/N2 | 旧 soft metric 使用别的相似度和损失，且单位长度归一化 | 绝对差筛选、全对交叉熵、回归、大小项 | 同上及 `tests/test_fuzzy_metric.py` |
| 每10新标签更新/每批5 | 旧主线不具备该循环 | 全池重算、版本同步 | 新 `analyse/fuzzy_metric_replay.py` |
| GP每批随标签拟合 | 原GP工具可参考 | 新空间、方差、持久化 | 同上；复用原采集类 |
| 固定P2按ID选择 | 旧冻结规则存在 | 拆出单批循环，不更换多样性规则 | 同上及 `tests/test_fuzzy_metric_replay.py` |
| 选点前预测/揭示/恢复 | 原日志和恢复机制可参考 | 保存待揭示批次、查询计数、GP与映射状态 | 同上 |
| 50＋20真实短回放 | 有336个完整真实候选 | 待实际运行和恢复验证 | `configs/fuzzy_metric_stage1_v1.json`、独立实验目录 |
| 修订任务和阶段状态 | 原文件仅GitHub预检 | 以服务器证据替换状态，保留研究要求 | `docs/Fuzzy_Metric_Alternating_Search_Stage_Tasks.md`、本报告与阶段1报告 |

## 6. 显式适配与尚未解决的研究问题

1. 方案说“P2 使用固定架构距离”，实际冻结 P2 是架构指纹约束，没有连续结构距离。本轮保留真实 P2 行为，批量从 4 适配 5；共同初始 50 不计入其新增搜索 cap。没有另加距离阈值或随机探索。阶段 2/3 文档须明确该语义，若改为距离型 P2，应另立变体，不静默换规则。
2. 方案未固定 GP 数值实现和步数；本轮沿用已核实 Matérn5/2＋SE²＋噪声下限、20步 Adam 0.05，补充保存方差/状态；采集继续现有 qLogEI，固定128个 Sobol 样本、seed0，q=1 后按P2取5。工程起点不能当作开发选优结果。
3. $\varepsilon=0.01$、$\lambda_f=0.1$、$\lambda_e=0.001$、100步映射训练仅为方案范围内固定工程起点。本轮不选参数、不比较方法有效性，不删除后续 N0/N1/N2、F0/R1/T1/B4/F1、一维GP、D1 等要求。
4. 配方和预算未改变：历史三种子均值为一个逻辑查询；本轮新增候选 GNN 训练预算 0。336池足够50＋20，不需要缩减或补标。
5. 阶段2前需按架构分组冻结开发划分；确认前另行指定未用于调参的数据/运行、数值判据和对照预算。历史 VAE 输入文件哈希溯源限制需保留说明。以上不阻塞仅工程用途的阶段1。

## 7. 交付目录与阶段结论

独立结果根：`/home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage01-20260909-v1/`。既有原始结果全部保留。本报告完成阶段0的真实入口与数据边界核对；阶段1是否完成以实际测试、回放、恢复结果及 `Fuzzy_Metric_Stage1_Validation_20260909.md` 为准。
