# DKL/GP 科研代码安全同步审查入口

## 快照身份

- 采集时间：2026-09-29 17:01:21 +0800（Asia/Shanghai）；服务器同时为 2026-09-29 05:01:14 -0400（EDT）。
- 源工作区：`~/D-VAE`，分支 `feature/accuracy-bottleneck-diagnostics`，原 HEAD `64a44c14316f11debaf01cf8583426acfe1410e6`。
- 审查分支基准：远程已核实的 `origin/feature/accuracy-bottleneck-diagnostics`，SHA `397a12526169feddf7e8a08e707accd35881e669`。
- 审查分支：`review/dkl-sync-20260929T170121+0800`。
- 源工作区采集时状态：暂存 0 个、未暂存 2 个（`bo_phase4.py`、`eval_utils.py`）、未跟踪 17,313 个；本轮没有在源工作区切换分支、暂存或提交。
- 源分支相对远程领先 11 个本地提交；其中历史包含 `raw_progress_test_sealed.json` 等受保护结果路径，因此这 11 个提交没有成为审查分支祖先。审查分支从远程安全基准创建，再按白名单复制当前文件内容。

## 本轮纳入范围

1. 正式搜索主线：`bo_phase4.py`、`eval_utils.py`、`final_eval.py`、`layernorm_search_space.py`，以及 DKL/Exact GP、检查点、指标和 HP 处理模块。
2. 当前只读诊断入口：`analyse/compare_exact_vs_dkl_prequential.py`、`analyse/diagnose_dkl_uncertainty.py` 及对应测试。
3. 服务器现存的 8 组阶段化 DKL/GP 实验代码快照：仅同步 `code/**/*.py`、冻结协议、代码清单及冻结清单；不含训练状态、预测、评分表、日志、压缩证据包或原始数据。
4. 本目录中的审查索引与 SHA-256 manifest。

科研代码与配置均为服务器现有内容的原样副本；本轮只新增审查文档，没有修改算法、重构或格式化科研代码。

## 正式流程与诊断流程

正式搜索入口是 `bo_phase4.py`。当且仅当命令显式选择 `--surrogate_type dkl_gp` 时，它构建或加载 `DKLAccuracyGPPredictor`；该预测器以标准化的 `[12 维架构潜变量, HP]` 为输入，经过一层小型特征网络后进入 ARD RBF Exact GP。在线阶段使用 qLogEI 提议候选，先预测，再解码并执行真实候选评估，最后将允许回填的 validation 观测追加到 GP 并按配置重拟合。默认 `--surrogate_type` 是 `exact_gp`，所以“代码支持 DKL”不等于“每次正式搜索实际运行 DKL”。本次安全材料中没有找到可证明正式 DKL-BO 已实际启动的无敏感启动记录。

当前诊断流程分两类：

- `analyse/compare_exact_vs_dkl_prequential.py` 与 `analyse/diagnose_dkl_uncertainty.py` 是对已有轨迹的按时间顺序离线回放；脚本自身明确说明它们不等同于 DKL-BO。
- `artifacts/*/code/` 是独立的阶段化实验快照，覆盖联合 DKL、冻结表示 GP、只读 readout、OOF 校准、分块核和嵌套优化检查。这些流程不是 `bo_phase4.py` 的正式搜索主线，不能统称为“当前 DKL”。

## 推荐阅读顺序

1. 本文件；
2. `CODE_MAP.md`；
3. `bo_phase4.py` 中 `_fit_scratch_predictor`、`optimize_acq`、`_append_eval`；
4. `surrogate/dkl_accuracy_gp.py` 与 `surrogate/accuracy_gp.py`；
5. 两个 `analyse/` 诊断入口；
6. `EXPERIMENT_INDEX.csv`；
7. 按问题选择对应 `artifacts/<run-id>/code/`；
8. `ENVIRONMENT_AND_GAPS.md` 与 `SNAPSHOT_MANIFEST.json`。

## 没有执行的步骤

本轮没有运行 GNN、VAE、DKL 或 GP 训练，没有运行搜索、推理、预测、重新评分、重新编码/解码或候选生成；没有导入项目训练模块，没有运行 pytest；没有安装或升级环境，也没有停止服务器任务。只执行了 Git/文件只读检查、SHA-256 核对、Python AST/JSON 静态检查和发布内容安全扫描。

## 缺件与复现边界

- 权重、检查点、状态、缓存、预测、评分输出、B/test 材料、Cora 原始/处理数据和证据压缩包均未上传。
- 仓库没有 requirements、environment.yml 或 pyproject 等依赖声明；已安装版本仅记录在 `ENVIRONMENT_AND_GAPS.md`。
- 8 个阶段实验记录的 Git 上下文是 `64a44c...`，但其实际实验代码位于未跟踪的 artifact/worktree 快照；因此只能由本分支中的逐文件哈希固定，不能声称由该 Git commit 单独复现。
- 未找到能无歧义对应“最近误判诊断四个查询候选”的完整 candidate_id 清单。最新可确认的 `gp-block-optcheck` 协议定义的是 5 个 A 池病例，不能替代四候选集合；因此没有导出候选结构，也没有读取任何 B/test 标签来补齐。
- 此快照足以进行静态代码审查、调用链核对和配置比较；不具备完整数值复现条件。

## 受保护历史处理

审查分支不包含源 HEAD `64a44c...` 及其相对远程多出的 11 个提交作为祖先，避免把其中受保护结果变成新的可达历史。需要审查这些历史提交时，应先单独完成数据隔离与历史安全审计；本轮没有改写、清理或强推任何历史。
