# A/C 与 LayerNorm 实际源码同步审查报告

本提交只同步实际运行源码、必要依赖和小型身份说明；不改变科学实现、实验结果或原服务器工作区。目标为 `Sherlockmorefff/NAS` 的独立 `review/ac-ln-source-sync-20261013` 分支，不合并 main、不 force push。

## 基底和原工作区

服务器原目录 `/home/fangxy/D-VAE`，原分支 `feature/accuracy-bottleneck-diagnostics`，原 HEAD `64a44c14316f11debaf01cf8583426acfe1410e6`。原工作区有297项porcelain记录，包括 bo_phase4.py / eval_utils.py 的既有未提交修改。没有reset、clean、checkout原工作区或停止运行任务。

实际 git fetch 后 main 为 `9a1ea429df658cdb4642dca36e4fa794523ba49f`（2026-06-23）。对远端分支逐文件比较后，以最接近的 `origin/review/dkl-sync-20260929T170121+0800` / `1ab3901a13673ae3751caff88f530559f9419f30` 建立独立worktree `/home/fangxy/D-VAE-review-ac-ln-20261013`。模型 eval_utils.py 已与该基底逐字节一致，因此没有改动模型文件。比较详见 SOURCE_SYNC_MANIFEST.json 的 remote_source_comparison。

## 逐文件来源核对

两轮 SOURCE_MANIFEST 与各自 freeze.source 一致，共57个不同实际冻结路径；当前服务器文件全部匹配预期SHA256。无当前树后来修改冲突，也无两轮共享文件冲突。57个文件包括开放搜索/LN入口与模型、数据、确定性、HP、代理、WL、CPU科学检查和原validation-only核函数。

另收录13个旧B/AC冻结身份依赖：开放搜索的verify会读取其文件hash，故不能只复制两个新包而缺少它们。这些旧源码不意味着本轮运行B，也不意味着本次执行旧搜索。独立监督watchdog.py与LN报告finalize_delivery.py各有自己的实际执行回执，按原SHA256作为两个补充工具保存；不会冒充冻结科学模块。

源码均原样保留，特别是LN report.py的NumPy布尔序列化错误也未在冻结实现中悄悄修正。实际交付修复仅转换报告输出边界的标量类型，未重训/重评；原修复工具、原路径及sha在 scripts/review/ac_ln_source_sync/ 和 docs/ac_ln_source_sync/identities/。

## 调用与运行语义

完整调用/环境说明见 [docs/ac_ln_source_sync/README.md](docs/ac_ln_source_sync/README.md)。开放搜索入口为 `python -m analyse.ac_open_graph_mix`，LN入口为 `python -m analyse.ac_champions_ln_reuse_none`，各自start_tmux.sh保持真实路径和参数。

开放A/C各五组、每轨迹50+250，6300逻辑任务；LN仅300项新训练，严格复用300条原None final逐seed validation/test，无None新训/重评、无旧权重warm start。原架构/HP/数据/预算/Adam/调度不变，checkpoint只按validation严格改善选择；全部LN模型与validation判断先封存，再独立strict layer_norm float32评分。初始化为冻结None代码按原seed重建证据，不伪称旧训练存有初始hash。

Cora public划分与NormalizeFeatures，Torch 2.5.1+cu121、PyG 2.7.0；原冻结GP环境及数据/划分指纹在清单和原protocol小文件中。两轮结果仍在 `/home/fangxy/D-VAE/results/ac_open_graph_mix_20261012v1/` 与 `/home/fangxy/D-VAE/results/ac_champions_ln_reuse_none_20261013v1/`。

## 实际执行的检查

- 70份冻结/历史身份源码逐SHA256一致；两份独立补充工具与其回执及服务器快照一致。
- 72份源码/脚本语法检查：Python compile 与 shell bash -n；67个模块CPU导入路径均指向独立review worktree。
- 两个实验入口 --help 通过；未调用prepare/run/worker/score、真实数据加载、GNN/NAS、GP拟合或test评分。补充工具仅编译，未导入执行。
- 既有科学SANITY/验收的来源和hash已记录，复用原核验，不再次产生实验数据。
- git diff --check（未暂存部分）通过；默认 git diff --cached --check 唯一发现为冻结LN training.py:157原有EOF空行。保留原SHA与字节，不改源码；其他文件默认严格检查通过，该文件关闭blank-at-eof类别后其它空白检查通过。此已存在风格例外不冒称默认全量检查PASS。
- staged差异和逐文件白名单/内容hash核查完成；原HEAD、原分支、完整porcelain和tracked diff已核验保持。

未复制未使用的历史followup独立run_task/vae_label_expansion整套pipeline；本轮实际调用的train_validation_only/epoch_core及新入口依赖已完整导入核验。仓库仍需服务器外部封存池/数据/权重/结果来源，源码clone不等于输入数据也已打包。

## 提交范围及交付

只显式stage本清单中的新增源码、两份补充工具、六份小身份/协议回执，以及本报告/总清单/README。没有git add .。权重、数据集、结果包、逐成绩表、运行日志、环境目录、凭据及其他原工作区修改不提交；基底本已有的历史代码不作为本次新增内容。

本提交没有科学代码字节修改。push后以 git ls-remote、fetch新分支和固定commit的Git blob逐SHA复核；最终分支/commit与文件链接、远端核验回执由交付消息提供。文档不把自身commit写进自身hash，冻结身份保持原值。
