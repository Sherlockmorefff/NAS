# 复算说明

本包使用服务器现有 `/home/fangxy/.conda/envs/nas/bin/python`，Python3.10.20、PyTorch2.5.1+cu121（仅CPU）、GPyTorch1.15.2、NumPy2.2.6。没有安装或升级依赖。当前C原文件已包含于包内，原样SHA256记录于manifest；BoTorch仅用于历史审计，本次训练不导入它。

## 不训练，独立重算已有结果

在证据包解压根目录执行：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/fangxy/.conda/envs/nas/bin/python code/verify_artifacts.py --root "$PWD"
```

它只读取获准20条快照与9个小型终点检查点，不读服务器历史库；重算选样数据尺度、SE换算、后验核/均值/方差、独立Cholesky NLL和C目标梯度等价性。只写本次重算检查结果。检查点由本任务生成，包含模型与优化器/RNG状态。

## 完整重放短程检查

```sh
PYTHON_BIN=/home/fangxy/.conda/envs/nas/bin/python sh reproduce.sh /absolute/path/to/a-new-empty-run
```

目标目录必须不存在。脚本复制完整新代码、固定协议及允许20条数据快照到新目录，运行合成检查、两个种子及seed0重放，然后重算。不会访问历史标签源、启动GNN/VAE或进入阶段2。所有真实训练步骤及验证副本在同一次运行的1200秒计时器下执行。因实现修复重跑时须保留旧尝试、继承累计预算；本交付原始真实运行没有失败/重跑。

复算使用固定ID清单，不重新接触整池标签。需要审计选样来源时，数据适配器 `audit/select/release/mock` 各子命令以及当前库原位置、公共文件hash、源标签文件hash均保存在data审计和selection_manifest中。重新访问源库不属于离线复算所需步骤。

## 版本和差异

原仓库HEAD `64a44c14316f11debaf01cf8583426acfe1410e6`，工作区未提交文件清单保存在audit/git_status_before.txt。实际依赖C源文件直接从脏工作区复制，未用HEAD替代。所有新实现完整包含；主线及旧入口没有修改，必要差异就是本包新增文件，`code_manifest.json`记录各新文件及原样依赖hash，`implementation.patch`提供新增实现文本差异。历史机制源快照在audit/source/，其hash关系在audit/source_evidence.json。

测试记录：stage1_checks.json、synthetic/checks.json、saved_artifact_checks.json、audit/validation_final.json。正式阶段1trace与额外验证trace分开，合计583次真实优化器更新。真实训练计时记录包括工程探针；读取/报告/离线重算不列入训练耗时。

任何后续科学比较必须另行确定数据分组、指标/门槛和预算，`next_stage_authorized=false`。
