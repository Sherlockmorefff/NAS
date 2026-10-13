# A/C 开放搜索与固定冠军 LayerNorm：实际源码审查

本目录索引 2026-10-12/13 两轮实验的实际源码。源码按服务器的 SOURCE_MANIFEST / freeze / 交付修复回执逐文件核对，原冻结代码按字节保留。本次同步不运行实验，不修改模型、算法、配置、预算、分数或已封存判断。

## 阅读顺序

1. 仓库根目录 `SOURCE_SYNC_MANIFEST.json`：每个文件的 SHA256、源路径、冻结所属实验及独立修复/监督工具的角色。
2. 仓库根目录 `GIT_SOURCE_SYNC_REPORT.md`：远端基底比较、核验、同步范围和原工作区保留记录。
3. `eval_utils.py` 的 `DynamicGNN`：实际模型及 baseline/layer_norm 分支；`deterministic_runtime.py`、`dataset_utils.py`、`hp_modes.py`：确定性、数据与实际 HP。
4. `analyse/ac_open_graph_mix/{prepare,graphs,gp,search,optimize,runtime,scoring,report}.py`：开放搜索科学路径和调度。
5. `analyse/ac_champions_ln_reuse_none/{prepare,checks,training,runtime,scoring,report}.py`：严格配对的 None 复用与 LN 从头训练。
6. `scripts/review/ac_ln_source_sync/{watchdog,finalize_delivery}.py`：实际使用的独立监督/报告交付工具，按原字节单独保存；不属于冻结科学源码。

## 实验及调用关系

### A/C 开放搜索

协议 `AC_open_graph_mix_v1_20261012`。A/C 各五组，每组 50 初始化 + 250 在线查询，共 6300 项逻辑训练（搜索3000、Top10×30复评3000、冠军×30final300），B 不运行。A 使用 RAW16 GP、多起点梯度采集与冻结 VAE 落地；C 使用完整稳定 WL token 的 W-I GP、随机合法图/变异及固定图混合 HP 优化，在线 VAE encoder/decoder 调用为 0。各轨迹仅使用本轨迹已见 validation 标签。

入口和原调用结构如下，仅作审查/复现说明，本次同步没有执行 prepare/run/worker/score：

```text
python -B -m analyse.ac_open_graph_mix prepare --prompt <冻结任务文档> --run-dir <本协议结果目录>
bash analyse/ac_open_graph_mix/start_tmux.sh
python -u -B -m analyse.ac_open_graph_mix run --run-dir <本协议结果目录>
```

`start_tmux.sh` 的真实服务器路径保持原样；`__main__.py` 分发控制器、GPU/GP worker、评分和验收器。原控制器/监督器自动从搜索推进至复评、final、统一模型封存、独立 float32 test 和报告。C 的完整历史特征扩展与 HP 梯度路径见 `graphs.py`、`search.py` 和 `optimize.py`。

### 十个冻结冠军：300 新 LN + 300 历史 None

协议 `ac_champions_ln_reuse_none_20261013v2`。A/C 各五个冠军，各30个原final seeds；只有300项新LN训练。None validation/test 从开放搜索的原 FINAL_SEED_RESULTS.csv 按 method/group/CID/seed 一对一读取，无新None训练或重复评分，无历史LN缓存，无旧权重warm start，无NAS/GP/VAE。

```text
python -B -m analyse.ac_champions_ln_reuse_none prepare --prompt <冻结任务文档>
bash analyse/ac_champions_ln_reuse_none/start_tmux.sh
python -u -B -m analyse.ac_champions_ln_reuse_none run --root <本协议结果目录>
```

模型只启用原 `DynamicGNN(component_variant='layer_norm', activation_variant='relu')` 分支：非Identity卷积输出 → 独立LayerNorm(hidden_dim, eps=1e-5, affine=True) → 原ReLU → 原dropout；Identity、分类头和GIN内部MLP没有新增LN。LN gamma/beta仍接受原Adam全部参数weight decay。

架构、边、HP、300 epoch/patience80、Adam、CosineAnnealingLR和first-strict-best validation checkpoint规则保持。训练视图屏蔽test标签/掩码，训练test计算为0。全部300模型及validation工程建议先封存，独立评分器才用正确LN变体和strict=True加载权重统一评分。None validation来自final逐seed CSV，不是Top10复评均值。

初始化证据是从冻结None代码按原seed重建未训练None/LN，比较共同参数与Python/NumPy/Torch CPU/CUDA训练前RNG，验证新增LN常数初始化。历史任务未保存真实初始化哈希，不能称为历史训练当时的逐对初始权重核验。

## 环境与外部输入

原环境：Python 3.10；Torch 2.5.1+cu121、PyG 2.7.0、BoTorch 0.16.1、GPyTorch 1.15.2、NumPy 2.2.6、SciPy 1.15.3；同步时实测 linear-operator 0.6.1、igraph 1.0.0、OGB 1.3.6。后面三项为同步时环境实测值，不冒称它们全部写入了原冻结环境清单。

Cora Planetoid public固定划分、NormalizeFeatures、原全图float32训练。数据指纹 `373d00c7962dc1ab0cc86974b1542eb76a768d2878013ceee0691b3932d062c7`，划分指纹 `bcf2020a2d127eb2b4a91fb698062a3b22106a11b752490924c1adbe5dba9e09`。

Git仓库不包含数据、模型权重、原768池、任务数据库、完整候选档案或结果包。真实入口需要服务器上对应的封存输入和结果来源；仅clone源码无法凭空恢复这些输入。入口/启动器原绝对路径保留，未为审查同步重写路径或生成一个不同协议的实现。

运行/结果根目录：

- `/home/fangxy/D-VAE/results/ac_open_graph_mix_20261012v1/`
- `/home/fangxy/D-VAE/results/ac_champions_ln_reuse_none_20261013v1/`

开放搜索还核验旧 B/AC 来源：`wi_schur_B_pool300_20261011v1` 与 `wi_schur_AC_pool300_20261012v1`。因此对应冻结身份依赖也收录；它们不会被本次审查运行。旧 `structured_kernel_followup.validation_train.run_task` 独立CLI助手引用的整套历史followup runner/vae_label_expansion 不属于本轮调用路径，也没有整体复制；本轮使用其 `train_validation_only` / `epoch_core` 等核函数，所有相关入口与导入已核验。`analyse` 保留原namespace package形态，没有伪造或添加其原来不存在的 `__init__.py`。

## 源码身份与交付工具

两轮主冻结身份：

- 开放搜索：`66bf42f94961b5fbfa1255e59b719db0953d92363cd5e060ac533b1d020ddb71`
- LN：`0de8d569e77ca4a6330c3171781971d3d061d956a1025c73c15e2568945fe2c0`

`identities/` 保留两轮原 SOURCE_MANIFEST.json / PROTOCOL.json 的精确小文件副本，以及独立监督和交付修复回执。逐源码SHA256与路径映射见根清单。

LN的冻结 `report.py` 产生NumPy布尔标量，原JSON输出因此失败。**该冻结源码保留原样**。真实修复工具 `finalize_delivery.py` 仅在报告输出边界转换NumPy标量为同值Python标量，重新生成报告/验收/证据包；没有重新训练、评分或修改validation判断。原执行路径为 `artifacts/ac_champions_ln_reuse_none_20261013v1/finalize_delivery.py`，原命令为 `python -u -B -m artifacts.ac_champions_ln_reuse_none_20261013v1.finalize_delivery`。本仓库单独保存的review副本SHA256完全相同，原路径/hash见 DELIVERY_REPAIR 回执。它是有副作用的历史交付工具，本次只编译检查，未导入或执行。

开放搜索的独立 `watchdog.py` 为封存后追加的心跳/同身份恢复监督工具，也以原hash单独保存，不能当作冻结科学模块。开放搜索曾出现控制器/验收器并发打包的临时文件竞争，后续使用未修改冻结代码进行串行重新打包；没有伪造一个新增的“训练版本”。相关完整交付记录仍保留在服务器。

## 本次检查边界

逐SHA核验、源码/脚本语法、67个模块的CPU导入、两个新入口的--help、git diff --check、staged白名单及差异审查。已有SANITY/训练/评分验收作为科学核验来源，本次不重跑真实GNN、NAS、GP拟合或test。两个补充工具仅编译，不导入执行。完整运行结果及历史test/seed暴露限制仍以服务器报告为准；源代码同步不构成追加实验。

默认staged空白检查的唯一发现是原冻结LN training.py末尾空行（第157行），为保留实际SHA未删除。其余文件默认检查与该文件其它空白类别检查通过；清单记录原始返回码和明确例外。
