# DKL/GP 代码映射

状态说明：`存在`仅表示代码存在；`入口调用`表示静态调用链可达；`执行记录`仅表示已有冻结清单/状态记录，未在本轮运行验证。

## 正式搜索主线

| 实际路径 | 函数/类/配置项 | 在流程中的作用 | 调用来源 | 使用证据 |
| --- | --- | --- | --- | --- |
| `hp_modes.py` | `validate_hp_mode`、`hp_dim_from_mode`（81–96） | 固定 HP 模式及输入维度 | `bo_phase4.py`、GP 预测器 | 入口调用；本轮未运行 |
| `nas_space.py` | `JointSpaceVAE`（155–383） | 架构/HP VAE；正式搜索时加载检查点并负责候选解码 | `bo_phase4.load_vae`、`eval_utils._decode_arch` | 入口调用；权重未同步 |
| `surrogate/accuracy_gp.py` | `normalize_search_vector`（124–155） | 将架构潜变量裁剪并映射到 `[0,1]`，HP 直接裁剪到 `[0,1]` | Exact GP 与 DKL 共用 | 入口调用 |
| `surrogate/dkl_accuracy_gp.py` | `SmallFeatureExtractor`（39–89）、`DeepFeatureKernel`（91–135） | 一层 `Linear→SiLU/ReLU→Linear` 表示映射；其后为 ScaleKernel(ARD RBF) | `DKLAccuracyGPPredictor._build_model` | 代码存在且正式入口可调用 |
| `surrogate/dkl_accuracy_gp.py` | `DKLAccuracyGPPredictor`（185–485） | 构建 Exact GP、联合优化特征网络与 GP 参数、早停并保留最佳训练状态 | `bo_phase4._fit_scratch_predictor` 或 checkpoint load | 入口调用；无安全启动记录证明正式 DKL-BO 已执行 |
| `surrogate/dkl_accuracy_gp.py` | `_build_model`（265–283） | 使用 `SingleTaskGP`；未显式传入 mean，沿用 BoTorch 默认均值模块；使用默认同方差 likelihood 并施加噪声下限 | `AccuracyGPPredictor.refit` | 入口调用 |
| `surrogate/dkl_accuracy_gp.py` | `_optimize_model`（323–407） | 负 Exact MLL（`.sum()`）、Adam；特征网络有 weight decay，其他 GP 参数无 weight decay；梯度裁剪和早停 | `refit` | 入口调用；不是诊断包的 FixedNoise GP |
| `surrogate/dkl_accuracy_gp.py` | `fit_offline`（422–485） | 以初始有效候选的均值/总体标准差归一化目标，建立初始 DKL | `bo_phase4._fit_scratch_predictor` | 入口调用 |
| `surrogate/accuracy_gp.py` | `predict_batch`（749–770） | 后验预测并反标准化均值/标准差；包含数值恢复路径 | `bo_phase4._append_eval`、诊断 | 入口调用 |
| `surrogate/accuracy_gp.py` | `append_observations`（804–856）、`refit`（915–941） | 回填 validation 观测；重复点聚合；每次重建 GP 并载入兼容旧状态，避免沿用失效预测缓存 | `bo_phase4._append_eval` | 入口调用 |
| `bo_phase4.py` | CLI `--surrogate_type` 与 DKL 参数（469–479） | `exact_gp` 为默认；`dkl_gp` 必须显式选择 | `main` | 代码存在 |
| `bo_phase4.py` | `_fit_scratch_predictor`（3038–3120） | 从有效初始化候选建立 Exact GP 或 DKL；DKL 初始训练种子由搜索种子稳定派生 | `run_bo` | 入口调用 |
| `bo_phase4.py` | `optimize_acq`（1609–1725） | 正式在线选点使用 qLogEI；不是“按均值选前五” | `run_bo` 的在线循环 | 入口调用 |
| `bo_phase4.py` | `_append_eval`（1997–2335） | 严格执行预测→候选真实评估→记录→安全回填→可选重拟合；形成正式选点—评估—回填闭环 | 初始化与在线 BO | 入口调用；本轮未运行 |
| `eval_utils.py` | `eval_z_search`（1064–1331）、`_decode_arch`（1332–1384） | 解码架构与 HP，调用 GNN 训练/validation 评估 | `bo_phase4.eval_candidate` | 入口调用；本轮禁止执行 |
| `surrogate/metrics.py` | `prediction_metrics`（93–139）、`GPConvergenceMonitor`（297 起） | 记录预序列预测质量与自适应停止信息 | `bo_phase4.run_bo` | 入口调用 |
| `final_eval.py` | `select_history_records`（777–840）、`evaluate_candidate`（1050–1206）、`select_by_final_validation`（1259–1289） | 从历史独立选取候选并重新训练；最终按 validation 选择，test 仅作为禁止用于选择的诊断 | 独立 final-eval 入口 | 代码存在；本轮未访问/运行 test |

正式 DKL 中没有独立 VAE 冻结/解冻步骤：VAE 用于搜索向量到架构的解码，DKL 特征网络则直接接收标准化搜索向量。正式 DKL 也没有固定测量方差输入；这一点与下述阶段实验的 FixedNoise GP 不同。

## 当前诊断与阶段实验

| 实际路径 | 函数/类/配置项 | 在流程中的作用 | 调用来源 | 使用证据 |
| --- | --- | --- | --- | --- |
| `analyse/compare_exact_vs_dkl_prequential.py` | `load_ordered_history`（157 起）、`_fit_predictors`（254 起）、`run_history`（307 起） | 对已有搜索历史做泄漏受控的 Exact GP / 小型 DKL 预序列比较；不是正式搜索 | CLI `main` | 代码存在；脚本文档明确为 off-policy screening |
| `analyse/diagnose_dkl_uncertainty.py` | `run_history_diagnostics`（436 起） | 只读计算距离、核和不确定性诊断；不替换生产预测/qLogEI | CLI `main` | 代码存在 |
| `surrogate/fuzzy_metric.py` | `JointMetricModel`（17–58）、`metric_loss`（105 起） | 阶段实验中的 12D 架构 + 4D HP 小映射与回归头 | artifact `stage01_core` | 当前工作区未跟踪文件，已原样同步 |
| `artifacts/dkl-stage01-v1-20260922T071053Z/code/controlled_dkl.py` | `ExactGP`（39–64）、`Branch`（66–178） | Matern-5/2、常数均值、FixedNoise + 可学习额外噪声；映射/回归/GP 的隔离分支 | `run_stage1.py` | 冻结协议与代码清单存在；状态来源未上传 |
| 同上 | `objective`（112–127）、`train_block`（207–279） | 回归阶段优化映射+头；GP 阶段冻结映射；joint 阶段优化映射体+GP，回归头不参加 joint | `run_stage1.py` | 执行记录存在；本轮未运行 |
| `artifacts/dkl-stage2-v1-20260922T080157Z/code/stage2_data.py` | `build_folds`（57–113）、`build_subsets`（115–133）、`release_training`（251–289） | Cora public/A-only 五折；按选中 20/50 训练样本计算 x/y scaler 与已知方差 | `run_stage2.py` | 协议/冻结清单存在；标签与快照未上传 |
| `artifacts/dkl-stage2-v1-20260922T080157Z/code/stage2_core.py` | `RawBranch`（105–178）、`_optimizer`（230–255）、`train_block`（284–405） | 区分 RAW16、FROZEN、REG、DKL；按模式设置 requires_grad；每 100 步诊断/保存 | `Runner.context` | 执行记录存在 |
| `artifacts/dkl-stage2-v1-20260922T080157Z/code/run_stage2.py` | `Runner.context`（164–227）、`prediction`（118–158） | common 初始化后分叉；重算训练/查询表示、反标准化预测并封存 | `Runner.run` | 执行记录存在 |
| `artifacts/dkl-stage2-length-v1-20260922T092924Z/code/` | `run_length.py`、`length_diagnostics.py` | 对统一延长训练及稳定窗口做诊断；不是正式搜索 | 独立 artifact 入口 | 冻结协议存在 |
| `artifacts/dkl-residual-constraint-v1-20260924T024011Z/code/` | `residual_core.py`、`residual_diagnostics.py` | 残差映射约束与几何/稳定性诊断 | 独立 artifact 入口 | 冻结协议存在 |
| `artifacts/dkl-readout-frozen-v2-20260924T081732Z/code/readout_core.py` | `_ReadoutGP`（158–185）、`ReadoutModel`（188 起） | 固定表示 readout；GP_CONST 或固定趋势均值，不训练 mapper/head | `run_readout.py` | 与 joint DKL 明确不同；冻结清单存在 |
| `artifacts/dkl-oof-calibration-v1-20260924T095621Z/code/oof_core.py` | `prepare_common`（206–227）、`freeze_features`（238–249）、`readout_session`（250 起） | OOF 校准与特征冻结诊断 | `run_oof.py` | launcher、协议、冻结清单存在 |
| `artifacts/gp-block-distance-v1-20260924T125944Z/code/block_core.py` | `BlockMaternKernel`（88–120）、`diagnose`（316–379） | 原始输入 GP 的分块距离核；诊断中按后验均值产生 top5 | `run_block.py` | 该 top5 仅属于诊断，不是正式 qLogEI 选点 |
| `artifacts/gp-block-optcheck-v1-20260927T062149Z/code/nested_core.py` | `run_trajectory`（379 起）、`select_candidates`（370 起） | 有界嵌套求解与只读误判溯源 | `run_optcheck.py` | 协议存在；最新协议定义 5 个 A 池病例 |

## 选择、缓存和闭环结论

- 正式 `bo_phase4.py`：在线候选由 qLogEI 选取；存在真实 evaluation 与 validation 回填闭环。
- `gp-block-distance` 等诊断：存在“按预测均值取 top5”代码，但它只用于离线诊断/评分，不应描述为正式 BO 策略。
- GP 更新后通过重建 `SingleTaskGP`、重新设置训练输入并执行后验检查来刷新状态；阶段实验另外显式清空 `prediction_strategy` 或重算表示。
- 本轮仅做静态定位。上述“入口调用/执行记录”不等于本轮运行通过，也不证明算法结论正确。
