# 阶段 1 最小实现与验证记录

日期：2026-09-09。状态：已完成本轮阶段 1 工程交付；无方法有效性结论，阶段 2–6 未执行。

服务器：`ismc36`；项目：`/home/fangxy/D-VAE`；分支：`feature/accuracy-bottleneck-diagnostics`；HEAD：`64a44c14316f11debaf01cf8583426acfe1410e6`。运行起点保留服务器所有未提交修改，没有提交或推送。依赖实测记录见本轮 `environment.json`。

## 实现

- `surrogate/fuzzy_metric.py`：冻结的12维架构编码＋4项HP，HP映射4→16，联合网络28→32→32→8，SiLU，线性预测头8→1；非零固定随机初始化。N0/N1/N2、回归均方误差、全部无序对的加权邻域交叉熵和表示大小项均已实现。
- `analyse/fuzzy_metric_replay.py`：复用原数据聚合和VAE编码；准备候选白名单、完整HP、独立验证成绩库及按架构缓存的确定性编码；通过ID查配置。新模型不能取得隐藏均值、种子成绩、SE、测试成绩或旧B4预测。
- `configs/fuzzy_metric_stage1_v1.json`：初始50、新增历史查询20、每批5、每10更新；固定100步映射训练，不早停；N2、$\varepsilon=0.01$、$\lambda_f=0.1$、$\lambda_e=0.001$。相同种子的逐候选成绩差估计标准误，只给已揭示候选对降权。没有用噪声扩大邻居资格。
- GP采用现有20步Adam、Matérn5/2、SE²＋拟合噪声下限设置，保存模型和方差；复用现有 `StableQLogExpectedImprovement`，固定128个Sobol样本/seed0，逐候选q=1评分。P2保留架构指纹唯一和累计两次规则，批量适配为5。
- 每次表示更新重算全部336候选；GP总是用同版本已知点。保存每批全部待选候选的均值、区间、评分、选中ID、当时已知集合，以及逐条揭示顺序、表示版本、预算、随机状态、优化器和GP状态。恢复前校验配置、候选、基础编码、成绩源和源码版本。
- 候选配置包含4个可变HP、固定GAT/SAGE/GIN/GCNII参数、normalization、hp_mode、有效层数和来源编号。`candidate_id`始终是现有联合指纹，不反解新向量。运行入口只支持50＋20工程回放且GNN预算必须为0，没有调用真实候选训练入口。

## 实际验证结果

| 检查 | 已执行结果 | 证据 |
|---|---|---|
| 单元与集成工程检查 | **29 passed in 3.31s** | `tests_final.log` |
| 绝对差/严格阈值/无邻居 | N2 的60%/80%为0；无正邻居仍有回归与负对损失；无自配对 | `tests/test_fuzzy_metric.py` |
| 数值与VAE边界 | 三种邻域损失/梯度有限；冻结参数不变；确定性初始化 | 同上 |
| 真VAE缓存 | 全336配置两次编码逐项相等；实际192架构缓存；冻结参数数目检查为0可训练；张量哈希前后相同 | `prepared_v2/manifest.json` |
| 候选身份/隐藏成绩隔离 | 顶层和HP嵌套标签拒绝；未揭示均值/SE/测试字段污染不影响下一次预测及选择；HP双份值冲突被拒 | `tests/test_fuzzy_metric_replay.py` |
| 坐标同步 | 全池重新计算检查、GP已知点检查；故意污染任一坐标均被拒 | 同上；真实运行也执行此检查 |
| 工程恢复边界 | 选点未揭示、已揭示尚未刷新、10查询刷新后三种断点，与连续运行一致 | 同上；`audit_recovery_probe.json`为独立合成补充检查 |
| **真实历史短回放** | **50初始＋20次查询，70个不同候选，210条已有种子成绩** | `replay_final/summary.json`、`events.json` |
| 表示更新 | 初始版本0；新增查询10/20后版本1/2，恰好两次后续更新 | `replay_final/events.json` |
| GP拟合 | 查询数0/5/10/15/20处，共5次；均从头拟合 | 同上 |
| 真实数据恢复 | 第2批已选择未揭示处中断，恢复后查询序、全部4份预测文件、标签、网络、GP、全池坐标完全一致；仍为20次查询 | `real_resume_comparison_final.json` |
| 版本 | 连续与恢复运行8个相关源码文件哈希相同，配置另存 | `replay_final/source_identity.json`、`resume_final/source_identity.json` |
| 新GNN训练 | **0次**；没有补标、VAE微调或最终测试 | 独立回放入口和运行账本 |

工程测试使用合成数据，明确只检查程序；上表真实50＋20来源于服务器现存336配置历史池，二者没有混算。正式结果采用最终源码下的 `replay_final/`、`resume_final/`；较早调试/验证的 `prepared/`、`replay/`、`replay_resumed/` 和失败日志仍保留，未覆盖原始结果。

本轮为核对恢复重复执行了相同回放，均复用同一50＋20的标签，未多揭示其他独立候选。每个逻辑运行仍扣自己的20次历史查询；不能将读取历史缓存称作新增GNN训练，也不能将回放耗时称作真实搜索训练耗时。

## 已验证命令

以下命令均在服务器实际执行成功。新建输出目录要求不存在；重新完整运行时更换结果子目录，不覆盖已完成运行。

```bash
cd /home/fangxy/D-VAE
/home/fangxy/.conda/envs/nas/bin/python -m pytest -q tests/test_fuzzy_metric.py tests/test_fuzzy_metric_replay.py
/home/fangxy/.conda/envs/nas/bin/python -m py_compile surrogate/fuzzy_metric.py analyse/fuzzy_metric_replay.py tests/test_fuzzy_metric.py tests/test_fuzzy_metric_replay.py
git diff --check
```

实际数据准备：

```bash
/home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_replay.py --prepare --prepared results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2
```

实际50＋20回放：

```bash
/home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_replay.py --prepared results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2 --output results/posthoc/fuzzy-metric-stage01-20260909-v1/replay_final
```

实际中断/恢复检查：

```bash
/home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_replay.py --prepared results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2 --output results/posthoc/fuzzy-metric-stage01-20260909-v1/resume_final --pause-before-reveal-batch 2
/home/fangxy/.conda/envs/nas/bin/python analyse/fuzzy_metric_replay.py --prepared results/posthoc/fuzzy-metric-stage01-20260909-v1/prepared_v2 --output results/posthoc/fuzzy-metric-stage01-20260909-v1/resume_final --resume
```

最后一条对已完成运行再次执行只保留完成状态，不重复查询。阶段1入口可从其他当前目录启动，但建议按上面的项目根目录命令使用。

## 结果位置、差异与限制

完整结果根：`/home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage01-20260909-v1/`。

主要文件：阶段0 `data_audit.json`、`input_documents.json`、`workspace_before.json`；准备数据 `prepared_v2/`；正式工程回放 `replay_final/`；恢复副本 `resume_final/`；最终测试 `tests_final.log`；恢复比较 `real_resume_comparison_final.json`；最终差异 `workspace_after_comparison.json`、`new_files.patch`、`task_document_revision.diff`、`delivery_manifest.json`。

本轮修复的是可验证的工程边界：补齐完整HP/载入白名单，保证揭示已落盘而GP未更新时恢复重新拟合，增加源码一致性检查。最初关于固定噪声状态尺寸可能崩溃的推测没有在当前GPyTorch复现，不将其报告为已复现故障。测试断言及并行修改时的合成fixture不匹配已修正，原日志保留。

阶段0/1完成不代表邻域有效或搜索优于基线。阶段2前仍需冻结架构分组、开发参数比较与评判线；阶段3仍需F0/R1/T1/B4/F1及一维GP/D1等对照；确认数据/运行尚未指定。现有P2是架构身份配额而非连续结构距离，这一语义差异已公开记录。VAE历史训练输入的当时字节哈希缺失仍是溯源限制。旧评估缓存键未含训练轮数/patience，本轮独立配方校验规避此缺口，未来真实训练前需补齐。没有进入阶段2–6，没有改变后续研究要求或预算。
