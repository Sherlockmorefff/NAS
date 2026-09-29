# 环境、外部依赖与缺件

## 服务器环境（安全版本查询）

- 环境名线索经核实：`nas`；仅通过现有解释器查询版本，没有安装、导入训练入口或初始化模型。
- Python 3.10.20（GCC 14.3.0）
- PyTorch 2.5.1+cu121
- GPyTorch 1.15.2
- BoTorch 0.16.1
- PyTorch Geometric 2.7.0
- NumPy 2.2.6
- SciPy 1.15.3
- scikit-learn 1.7.2
- pandas 2.3.3

仓库根目录两层内没有发现 requirements、environment.yml、pyproject.toml、setup.py、setup.cfg 或 Pipfile。环境的完整导出没有同步，以免公开私有可执行路径、认证源或无关包。

## 外部依赖与绝对路径

- `bo_phase4.py` 仍包含 `/mnt/project` 的历史 `sys.path` 注入；实际服务器仓库位于用户主目录下。本轮只记录，没有修改。
- 正式流程需要 VAE checkpoint、数据集、数据划分及可能的 GP checkpoint；这些均未同步。
- 阶段实验依赖各 artifact 中的训练子集、状态、预测和父包；本分支仅含源代码与冻结协议，不能直接重放。
- Cora 的 raw/processed 文件在源工作区中是未跟踪数据，本轮全部排除。

## Git 版本缺口

- 源工作区 HEAD `64a44c...` 比远程跟踪分支 `397a125...` 多 11 个本地提交。
- 这 11 个提交包含名称明确的 sealed test 结果，因此审查分支没有以它们为祖先，也没有推送它们。
- 当前 `bo_phase4.py` 与 `eval_utils.py` 还有未提交修改，已按文件内容原样纳入；其他选定正式流程文件来自当前 HEAD。
- 8 组 artifact/worktree 实验代码均是未跟踪文件。它们的 Git 上下文是 `64a44c...`，但该 commit 不包含这些实际代码；因此版本身份以 `SNAPSHOT_MANIFEST.json` 的逐文件 SHA-256 为准。

## 未同步材料

- B 池/test 标签、成绩、预测和任何受保护实验内容：数据隔离要求。
- 模型权重、`.pt` 状态、checkpoint、缓存、`__pycache__`：体积与安全边界。
- Cora 数据和未授权原始数据：数据授权边界。
- `results/`、评分 CSV/JSON、预测 manifest/arrays、训练 traces、日志和原始证据压缩包：只做小型代码审查入口，不搬运结果目录。
- 完整环境导出与私人绝对路径：隐私与凭据风险。
- 四个查询候选及训练参考候选的原始结构：没有找到无歧义的完整 candidate_id 清单；最新误判审计定义的是 5 个 A 池病例，未用其替代。

## 静态发现但未修复

1. 正式 CLI 默认是 `exact_gp`；必须由命令或 checkpoint 明确选择 `dkl_gp`。没有安全证据证明正式 DKL-BO 已运行。
2. 正式 DKL 使用 `SingleTaskGP` 默认 likelihood/mean 组合，与阶段实验中的 FixedNoise + 可学习额外噪声不同；审查时不得混用结论。
3. 诊断脚本、冻结 readout、原始输入 GP 与联合 DKL 是不同实验族。
4. `final_eval.py` 包含 test 评估路径；本轮没有运行，也没有读取其输出。
5. 缺少仓库级依赖声明，完整复现环境不可由 GitHub 分支单独建立。

以上仅是静态代码与材料缺口，不是性能归因或算法正确性结论。
