# R1 固定五折开发复核（2026-09-11）

## 完成状态

沿用原 A 的五折和 15 个仅回归 100 步起点，完成 30 条各新增 200 步的固定轨迹。有效更新和实际更新均为 **6000**，废弃、重复、不确定更新均 **0**，300 步工程余量未使用。新增 GNN、VAE、GP 训练均 **0**。完成重入增加 **0** 步。

**当前R1跨折收益证据不足。** 相对R0-8D，整体顺序C增加0.001535，但额外邻居差E增加0.028777个百分点，只有第0、2折同时改善两者（2/5）；相对R0-1D，C降低0.004342、E减少0.177753个百分点，没有一折同时改善两者（0/5）。局部10顺序对两个对照均略降。预测头误差增加0.016202个百分点，近邻预测误差相对R0-8D增加0.012787、相对R0-1D减少0.277269个百分点。候选等权与架构等权的总体方向一致。

本轮使用已参与开发的 A 验证成绩，属于开发复核；B 和测试成绩未读取。未新增通过门槛，没有选参数、换划分或延长终点。不能把小幅预测误差变化直接解释为最终 GNN 精度变化。

## 实际位置与原始材料

主机 `ismc36`；项目 `/home/fangxy/D-VAE`；分支 `feature/accuracy-bottleneck-diagnostics`；完整 HEAD `64a44c14316f11debaf01cf8583426acfe1410e6`。
Python `/home/fangxy/.conda/envs/nas/bin/python`（3.10.20），PyTorch 2.5.1+cu121，NumPy 2.2.6；CUDA 可用、2 张 GPU。本轮单线程 CPU、float32，GPU 隐藏。完整环境和未提交文件列表见 `environment.json`。
开始时已有的 `bo_phase4.py`、`eval_utils.py` 修改和既有未跟踪研究文件作为基线保留；未 pull、切分支、reset、提交或推送。
独立结果目录：`/home/fangxy/D-VAE/results/posthoc/relative-metric-r1-fivefold-20260911-v2`。初次零更新启动目录 `/home/fangxy/D-VAE/results/posthoc/relative-metric-r1-fivefold-20260911-v1` 原样保留。

本轮原始指令来自 `/Users/sherlockmore/Downloads/Codex_Server_R1_Fivefold_Validation_Prompt.md`，项目存档为 `/home/fangxy/D-VAE/results/posthoc/relative-metric-r1-fivefold-20260911-v2/input_originals/Codex_Server_R1_Fivefold_Validation_Prompt.md`，SHA256 `39e38b2176b249d0a4a1c5ad1757ee52260882f6aa9b2dd6e9a3b7079538835b`。
已读取项目适用 `AGENTS.md`、最新阶段任务文档、R1 单划分报告及冻结协议、阶段2协议/划分/任务矩阵、N2 五折模型清单。原文副本和绝对路径/完整哈希见 `input_originals/`、`input_sources.json`。
冻结协议身份 `16ccd885144ea351d7fd9f2fcbf9c73badd7730acacc0165765c6a1fb0ccf508`；方法核心 `surrogate/relative_metric.py` 和原小实验入口均未改动。

## 要求—代码—证据

| 要求 | 实际实现及本轮变化 | 验证依据 |
|---|---|---|
| 原五折及15起点 | 新增 fivefold.audit；复用 closure.completed_state，交叉核对 started/completed/清单/哈希 | warmstart_checks.json：15项，无缺失；训练表示、距离最大差0；预测头、标签目标和SE平方与旧存档一致 |
| 原损失、原采样不变 | 复用 relative_metric 的 validate_batch/build_plan/order_terms/objective；不修改核心 | 13项零更新检查；141/143/144/145独立标量、旧112前向一致；plan_checks.json |
| 逐折标签隔离 | 五折独立 worker；只允许自身输入和输出文件，拒绝其他折/留出标签路径 | 实际文件访问拦截测试；每折子进程；全部封存后 evaluation_label_access.json |
| 同起点和新Adam | make_branches/fresh_adam；原Adam有100步但两边均重置 | actual_initial_objects.json；150节点实际加载模型、动量、RNG和采样位置 |
| 固定6000步和恢复 | 新增 intents/receipts/latest 原子账本；完成重入跳过 | update_ledger.json：6000项；runtime_validation.json；invocations/run02.json：0新增 |
| 全局/局部诊断 | 新增 probe_groups 只读包装；两组原样各半 | 150探针，独立复算w/间隔/损失及分组占比；损失占比不表示梯度贡献 |
| 三方法逐折评价 | 新增 fivefold_report，复用 evaluate_query；R0-1D用预测值作坐标 | 1611留出、6444训练分开；训练自排除；共同有效查询与架构等权 |
| 独立复算和便携证据 | 新增 fivefold_recompute，只有Python标准库 | 8055行、150探针及全部分折/总体配对复算；independent_recompute.json |

## 起点、划分和尺度

| 折 | 训练候选 | 留出候选 | 训练架构 | 留出架构 |
|---|---|---|---|---|
| fold0 | 141 | 38 | 76 | 20 |
| fold1 | 143 | 36 | 77 | 19 |
| fold2 | 143 | 36 | 77 | 19 |
| fold3 | 145 | 34 | 77 | 19 |
| fold4 | 144 | 35 | 77 | 19 |

五折留出恰好覆盖 A 的179候选、96架构一次。每折训练/留出架构互斥。起点均来自原阶段2外层 regression 的有效完成状态；旧100步权重、HP4顺序、候选完整配置身份、fit_ids、标签目标和噪声存档逐一核验。所有15个起点均有旧预测头参照，无参照缺失。
| 折 | 种子 | 有效原模型 | SHA256前12位 |
|---|---|---|---|
| 0 | 0 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold0-refit-regression-s0/attempt01/model.pt | 88841e26e87d |
| 0 | 1 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold0-refit-regression-s1/attempt01/model.pt | d66c48f688ab |
| 0 | 2 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold0-refit-regression-s2/attempt01/model.pt | 774ec2c89f2e |
| 1 | 0 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold1-refit-regression-s0/attempt01/model.pt | 6b6c231b7f6d |
| 1 | 1 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold1-refit-regression-s1/attempt01/model.pt | d414ff2381fd |
| 1 | 2 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold1-refit-regression-s2/attempt01/model.pt | 67d418cbd655 |
| 2 | 0 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold2-refit-regression-s0/attempt01/model.pt | 019f751933b3 |
| 2 | 1 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold2-refit-regression-s1/attempt01/model.pt | 2d25057f7411 |
| 2 | 2 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold2-refit-regression-s2/attempt01/model.pt | 4919e48db8f5 |
| 3 | 0 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold3-refit-regression-s0/attempt01/model.pt | 7e8aecdb37d4 |
| 3 | 1 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold3-refit-regression-s1/attempt01/model.pt | 8a045706c12b |
| 3 | 2 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold3-refit-regression-s2/attempt01/model.pt | f8b08fcfcc09 |
| 4 | 0 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold4-refit-regression-s0/attempt01/model.pt | 2c9a2d491b5f |
| 4 | 1 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold4-refit-regression-s1/attempt01/model.pt | c322ca68e9f6 |
| 4 | 2 | /home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/tasks/fold4-refit-regression-s2/attempt01/model.pt | 2b1fc135d0f7 |

完整哈希、完成记录、初始化身份、旧表示/距离/预测误差以及标准化数值见 `warmstart_checks.json`。`warmstarts/` 仅抽取原模型和原尺度；未保留旧优化器或GP对象用于训练。
标签只读取原 `A_validation.json` 白名单：同配方种子0/1/2，逐候选重新核验均值和 $SE=std(ddof=1)/\sqrt{3}$。每折使用自己的旧训练尺度，未重新拟合。`inputs/fold*/` 仅含本折训练标签及无标签特征；留出标签单独置于 `evaluation_labels/`。

## 固定设置与执行边界

冻结VAE缓存的12维编码与HP4进入原联合映射，输出8维表示及线性预测头。每个原起点深拷贝R0/R1，新建Adam：lr=0.001、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad/foreach/fused=False。全量本折训练；预测MSE逐点平均，大小权重0.001；顺序项R0=0、R1=0.01。无新增归一化、调度、早停、裁剪。
每折每种子单独生成200步完整采样计划，PCG64(20260911+模型种子)，不加入折号。合法方向严格按绝对性能差之差>1e-12；全局/局部10每组无放回抽8个，不足全取、空组为0。每组mean(w×hinge)，两组各半，再对当前折全部中心等权平均；原噪声降权、间隔与分母均未改变。
首步包含在各自200步预算内。0/1/50/100/200步均保存固定首计划探针；探针仅作记录，不参与选点、调参或终点选择。每步有更新意图、optimizer返回收据和原子保存状态；保存全局ID、随机状态、完整Adam与计划位置。
全部轨迹封存时间 `2026-09-11T13:44:47.422735+00:00`；统一留出评价开始 `2026-09-11T13:46:11.357912+00:00`。推理及精确距离近邻均先于成绩关联封存。

## 留出主结果

每种子先拼接五折179条唯一候选，候选等权平均，再平均3种子。顺序分数用原0–1单位；误差和邻居差用百分点。相对变化均为R1减对照，顺序正值较好，误差负值较好。
| 指标 | R0-8D | R0-1D | R1-8D |
|---|---|---|---|
| 整体顺序 C ↑ | 0.684533 | 0.690410 | 0.686068 |
| 局部10顺序 ↑ | 0.521709 | 0.520872 | 0.520254 |
| 实际邻居差 De（百分点）↓ | 7.743886 | 7.950416 | 7.772663 |
| 最佳邻居差 Dstar（百分点）↓ | 0.769609 | 0.769609 | 0.769609 |
| 额外差 E（百分点）↓ | 6.974277 | 7.180807 | 7.003054 |
| 预测头误差（百分点）↓ | 7.536112 | 7.536112 | 7.552314 |
| 近邻均值预测误差（百分点）↓ | 6.810751 | 7.100807 | 6.823538 |

| 指标 | R1−R0-8D | R1−R0-1D |
|---|---|---|
| 整体顺序 C ↑ | +0.001535 | -0.004342 |
| 局部10顺序 ↑ | -0.001456 | -0.000618 |
| 实际邻居差 De（百分点）↓ | +0.028777 | -0.177753 |
| 最佳邻居差 Dstar（百分点）↓ | +0.000000 | +0.000000 |
| 额外差 E（百分点）↓ | +0.028777 | -0.177753 |
| 预测头误差（百分点）↓ | +0.016202 | +0.016202 |
| 近邻均值预测误差（百分点）↓ | +0.012787 | -0.277269 |

## 每折及每种子配对变化

| 对照 | 整体顺序提高且额外差降低的折数 | 对应折 |
|---|---|---|
| R0-8D | 2/5 | 0,2 |
| R0-1D | 0/5 | 无 |

| 折 | 对照 | Δ整体C | Δ局部10 | ΔE百分点 | Δ头误差百分点 | Δ近邻误差百分点 |
|---|---|---|---|---|---|---|
| 0 | R0-8D | +0.001917 | +0.000741 | -0.049240 | -0.002496 | +0.025848 |
| 0 | R0-1D | -0.004024 | -0.009793 | -0.710058 | -0.002496 | -0.559649 |
| 1 | R0-8D | +0.001384 | -0.001546 | +0.151852 | +0.046222 | +0.122963 |
| 1 | R0-1D | -0.008550 | +0.003030 | +0.389506 | +0.046222 | +0.272469 |
| 2 | R0-8D | +0.000740 | -0.001508 | -0.071235 | -0.005799 | -0.155185 |
| 2 | R0-1D | +0.000048 | +0.006086 | +0.086543 | -0.005799 | -0.245556 |
| 3 | R0-8D | +0.001544 | -0.003403 | +0.109935 | +0.026139 | +0.048758 |
| 3 | R0-1D | -0.007026 | +0.010977 | -0.213987 | +0.026139 | -0.396209 |
| 4 | R0-8D | +0.002084 | -0.001802 | +0.010921 | +0.018600 | +0.023111 |
| 4 | R0-1D | -0.002267 | -0.012567 | -0.419937 | +0.018600 | -0.453206 |

| 种子 | 对照 | Δ整体C | Δ局部10 | ΔE百分点 | Δ头误差百分点 | Δ近邻误差百分点 |
|---|---|---|---|---|---|---|
| 0 | R0-8D | +0.001469 | -0.000452 | -0.050205 | +0.023155 | -0.078361 |
| 0 | R0-1D | -0.004397 | -0.002532 | -0.140857 | +0.023155 | -0.413780 |
| 1 | R0-8D | +0.001685 | +0.000380 | +0.089460 | +0.030097 | +0.113296 |
| 1 | R0-1D | -0.005511 | -0.004512 | -0.053035 | +0.030097 | +0.004171 |
| 2 | R0-8D | +0.001451 | -0.004295 | +0.047076 | -0.004647 | +0.003426 |
| 2 | R0-1D | -0.003118 | +0.005191 | -0.339367 | -0.004647 | -0.422197 |

误差变化并非各折一致：预测头误差在第1折（+0.046222百分点）和第3折（+0.026139）增加较多，第0、2折略降；相对R0-8D的近邻误差仅第2折下降（−0.155185），其余四折增加，第1折增加最多（+0.122963）。三个种子对R0-8D的C均增加，但只有种子0的E下降；头误差仅种子2下降，近邻误差仅种子0下降。相对R0-1D，近邻误差下降见于第0、2、3、4折，第1折反而增加0.272469百分点。局部排序未随整体C的小幅提高而改善，不能据此推断已学好性能接近候选的关系。
各折×种子的完整变化见 `evaluation/paired.csv`；不是将45方法状态或8055行当作独立重复。折间训练候选重叠，三个模型种子共享同一批候选；本轮没有统计显著性结论。

## 架构等权辅助结果

每种子先对同架构候选求均值，再对架构等权，最后平均3种子。训练诊断的重复折/候选条目也先在架构内平均；只作辅助诊断。
| 对照 | 指标 | 候选等权变化 | 架构等权变化 | 方向相同 |
|---|---|---|---|---|
| R0-8D | 整体顺序 C ↑ | +0.001535 | +0.001495 | True |
| R0-8D | 局部10顺序 ↑ | -0.001456 | -0.001419 | True |
| R0-8D | 实际邻居差 De（百分点）↓ | +0.028777 | +0.040231 | True |
| R0-8D | 最佳邻居差 Dstar（百分点）↓ | +0.000000 | +0.000000 | True |
| R0-8D | 额外差 E（百分点）↓ | +0.028777 | +0.040231 | True |
| R0-8D | 预测头误差（百分点）↓ | +0.016202 | +0.017868 | True |
| R0-8D | 近邻均值预测误差（百分点）↓ | +0.012787 | +0.027222 | True |
| R0-1D | 整体顺序 C ↑ | -0.004342 | -0.004626 | True |
| R0-1D | 局部10顺序 ↑ | -0.000618 | -0.002410 | True |
| R0-1D | 实际邻居差 De（百分点）↓ | -0.177753 | -0.150579 | True |
| R0-1D | 最佳邻居差 Dstar（百分点）↓ | +0.000000 | +0.000000 | True |
| R0-1D | 额外差 E（百分点）↓ | -0.177753 | -0.150579 | True |
| R0-1D | 预测头误差（百分点）↓ | +0.016202 | +0.017868 | True |
| R0-1D | 近邻均值预测误差（百分点）↓ | -0.277269 | -0.280903 | True |


## 训练查询诊断（不混入主结果）

五折训练候选合计716个折/候选条目，每查询排除自身；三方法三种子共6444行。存在重复候选，不能理解为6444个独立样本。
| 指标 | R0-8D | R0-1D | R1-8D |
|---|---|---|---|
| 整体顺序 C ↑ | 0.718791 | 0.729547 | 0.720987 |
| 局部10顺序 ↑ | 0.532486 | 0.537162 | 0.534223 |
| 实际邻居差 De（百分点）↓ | 5.421173 | 5.093004 | 5.401148 |
| 最佳邻居差 Dstar（百分点）↓ | 0.786220 | 0.786220 | 0.786220 |
| 额外差 E（百分点）↓ | 4.634953 | 4.306785 | 4.614929 |
| 预测头误差（百分点）↓ | 3.600239 | 3.600239 | 3.586282 |
| 近邻均值预测误差（百分点）↓ | 4.093662 | 3.958765 | 4.067678 |


## 探针与分组损失

完整150节点记录见 `probe_details.csv` 和各轨迹 `probes.json`：总顺序损失、激活比例、表示范数、共享映射梯度比/余弦均保留；全局和局部各自保存w五分位、均值和加权损失占比。下表为15起点对应轨迹的均值，w来自各自固定首计划。**损失占比不是梯度贡献比例。**
| 分支 | 新增步 | 组 | w均值 | 半权损失 | 损失占比 | 组内激活比例 |
|---|---|---|---|---|---|---|
| R0 | 0 | global | 0.441560 | 0.116445 | 0.959974 | 0.505960 |
| R0 | 0 | local10 | 0.017754 | 0.004838 | 0.040026 | 0.501353 |
| R0 | 1 | global | 0.441560 | 0.117623 | 0.959314 | 0.504614 |
| R0 | 1 | local10 | 0.017754 | 0.004972 | 0.040686 | 0.500367 |
| R0 | 50 | global | 0.441560 | 0.070714 | 0.956447 | 0.495592 |
| R0 | 50 | local10 | 0.017754 | 0.003241 | 0.043553 | 0.497839 |
| R0 | 100 | global | 0.441560 | 0.056953 | 0.962405 | 0.482022 |
| R0 | 100 | local10 | 0.017754 | 0.002268 | 0.037595 | 0.489557 |
| R0 | 200 | global | 0.441560 | 0.037655 | 0.961765 | 0.445427 |
| R0 | 200 | local10 | 0.017754 | 0.001541 | 0.038235 | 0.483553 |
| R1 | 0 | global | 0.441560 | 0.116445 | 0.959974 | 0.505960 |
| R1 | 0 | local10 | 0.017754 | 0.004838 | 0.040026 | 0.501353 |
| R1 | 1 | global | 0.441560 | 0.117348 | 0.959347 | 0.504734 |
| R1 | 1 | local10 | 0.017754 | 0.004956 | 0.040653 | 0.500367 |
| R1 | 50 | global | 0.441560 | 0.070001 | 0.956900 | 0.496170 |
| R1 | 50 | local10 | 0.017754 | 0.003169 | 0.043100 | 0.498124 |
| R1 | 100 | global | 0.441560 | 0.056543 | 0.963085 | 0.483415 |
| R1 | 100 | local10 | 0.017754 | 0.002209 | 0.036915 | 0.491603 |
| R1 | 200 | global | 0.441560 | 0.037035 | 0.963670 | 0.446830 |
| R1 | 200 | local10 | 0.017754 | 0.001437 | 0.036330 | 0.481694 |

局部w较低属于固定噪声降权的结果，没有因此增大权重、改分母或去除降权。探针不用于事后改参数。

## 验证、故障与预算

运行前13项检查全部通过，包含变长分母、独立标量公式、并列/空组、R0无顺序梯度、错折拒绝、实际文件边界、旧112前向一致和命令行冲突拒绝，优化更新0。实际重新加载150个节点，模型预测/探针、Adam步数、随机状态均核验；首步损失及实际参数变化精确复现，6000条更新ID与收据一一对应。
独立复算 8055 行和 150 探针；顺序/ID/计数精确匹配，浮点差异见 `independent_recompute.json`。留出1611行、训练6444行；全局NA=0，局部NA=0，配对排除累计=0，微小负E裁剪=2。负E低于−1e-12原单位会报错，不填0掩盖。
工程启动曾在v1被自身目录fsync的文件访问检查拦截，实际优化0；修复仅允许本折输出目录本身，跨折与留出路径仍拒绝。自动审批拒绝移动旧目录后，采用新的v2目录继续，v1原样保留。另有两次初始只读探查引号错误，均0更新。训练期间没有故障或状态重算。报告首次汇总时总激活率与分组激活率同名，改为分组前缀后继续，报告修复优化0。详情 `prior_zero_update_startup.json` 和v1 `failures/`、`logs/`。
恢复验证再次执行完整入口，30条已完成任务全部跳过，新增0步；正式实际6000、有效6000、废弃0、重复0、不确定0、修复余量消耗0，总上限6300。未使用合成数据训练，工程标量小例只作零更新检查。

## 已执行且验证可运行的入口

工作目录 `/home/fangxy/D-VAE`，解释器 `/home/fangxy/.conda/envs/nas/bin/python`。以下命令均实际执行；正式目录再次 `--run` 已验证不会新增更新。
```sh
cd /home/fangxy/D-VAE
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --audit
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --check
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --config configs/relative_metric_r1_fivefold_v1.json --run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold.py --report
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python analyse/relative_metric_fivefold_verify.py
/home/fangxy/.conda/envs/nas/bin/python -S analyse/relative_metric_fivefold_recompute.py /home/fangxy/D-VAE/results/posthoc/relative-metric-r1-fivefold-20260911-v2
```

日志：`logs/audit.log`、`checks.log`、`run.log`、`run01-fold0..4.log`、`completed_reentry.log`、`runtime_validation.log`、`evaluation.log`、`independent_recompute.log`。独立复算只用标准库，可复制到无项目环境，按证据包README运行。

## 交付文件与停止范围

新增代码：`analyse/relative_metric_fivefold.py`（审查/检查/五折训练/恢复）、`relative_metric_fivefold_report.py`（只读评价）、`relative_metric_fivefold_verify.py`（节点和账本检查）、`relative_metric_fivefold_recompute.py`（独立复算）、`relative_metric_fivefold_write_report.py`（报告）；新增配置 `configs/relative_metric_r1_fivefold_v1.json` 和测试 `tests/test_relative_metric_fivefold.py`。未修改原R1核心及小实验入口。
新报告保存到 `docs/Fuzzy_Metric_R1_Fivefold_20260911.md`；阶段任务文档只追加第15节。`source_snapshot/` 保存冻结训练源码、配置和测试；`delivery_sources/`、`source_changes.patch` 保存其余新增代码及最终差异；`warmstart_checks.json` 保存15起点，`trajectories/` 保存30轨迹，`intents/receipts/update_ledger.json`保存全部更新证据。
阶段3–6、GP/搜索及旧19项缺口均未启动或补做；原N2五折收尾与R1单划分结果未改写。不存在本轮关键材料阻塞。本轮到此结束；下一轮如需进入GP或搜索，应先明确新研究任务、预算及继续保留B/测试的边界，不能把本次开发结果当独立测试结论。
