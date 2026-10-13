# 复现说明

三种操作彼此独立。原始交付目录保持不变，纯评分也应在新副本运行。评分程序只用Python标准库，不导入模型或训练器。

服务器Python为 `/home/fangxy/.conda/envs/nas/bin/python`。模型复现环境见 `environment_frozen.json`：CPU、float64、1线程；不需要GPU、Cora下载、GNN或VAE权重。

## 1. 仅复算A诊断统计与分流

这条路径使用已封存的几何/近邻/预测，重新计算E1/E5、误差、区间与固定门槛；不重新前向，不更新模型。

```bash
DKL_BUNDLE=/home/fangxy/D-VAE/artifacts/dkl-residual-constraint-v1-20260924T024011Z
DKL_PY=/home/fangxy/.conda/envs/nas/bin/python
DKL_A_COPY=/tmp/dkl-residual-A-recompute-$(date -u +%Y%m%dT%H%M%SZ)
cp -a "$DKL_BUNDLE" "$DKL_A_COPY"
"$DKL_PY" "$DKL_A_COPY/code/score_residual.py" historical --root "$DKL_A_COPY" --labels-from "$DKL_BUNDLE/scoring_A/labels_A.json"
"$DKL_PY" "$DKL_A_COPY/audit/verify_residual_scoring.py" historical --root "$DKL_A_COPY"
```

核对 `diagnostic_gate.json` 的四组逐折值、均值、正折数及route。时间戳和本次评分程序来源字段可以不同；科学数值应在1e−12浮点容差内一致。若需要从历史检查点重新生成无标签几何，可使用第3节新目录准备步骤，完成A无标签诊断和评分后停止，不调用B。

## 2. 仅复算B固定终点评分

只能在已有统一封存清单的副本上执行，不调用训练器；不能用该命令补缺终点。

```bash
DKL_B_COPY=/tmp/dkl-residual-B-recompute-$(date -u +%Y%m%dT%H%M%SZ)
cp -a "$DKL_BUNDLE" "$DKL_B_COPY"
"$DKL_PY" "$DKL_B_COPY/code/score_residual.py" formal --root "$DKL_B_COPY" --labels-from "$DKL_BUNDLE/scoring_B/labels_A.json"
"$DKL_PY" "$DKL_B_COPY/audit/verify_residual_scoring.py" formal --root "$DKL_B_COPY"
```

输出逐单元、抽样/模型种子、折、配对比较、随机参考、区间评分和成本表。RAW16为20个实际终点；配对引用不变成60次独立拟合。A/B在这里表示本任务的两个执行部分，两个评分目录均仅包含合法A池标签，没有B池/test标签。

## 3. 在新目录按同一协议重新训练

这会产生新的模型计算成本，**不属于已完成实验的自动后续操作**。以下命令仅供明确需要复现时手动执行；本次交付没有再次运行它。

`prepare_bundle_rerun.py`仅从本包复制必要代码、合法输入和历史源状态，不读取历史评分表。它运行合成预检、累计验证成本、冻结完整新目录；不改变科学参数。后续A仍必须通过相同门槛才能启动完整B。

```bash
DKL_NEW=/home/fangxy/D-VAE/artifacts/dkl-residual-constraint-reproduce-$(date -u +%Y%m%dT%H%M%SZ)
"$DKL_PY" "$DKL_BUNDLE/audit/prepare_bundle_rerun.py" --bundle "$DKL_BUNDLE" --output "$DKL_NEW"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$DKL_PY" "$DKL_NEW/code/run_historical_diagnostics.py" --root "$DKL_NEW"
"$DKL_PY" "$DKL_NEW/code/score_residual.py" historical --root "$DKL_NEW" --labels-from "$DKL_BUNDLE/scoring_A/labels_A.json"
DKL_ROUTE=$("$DKL_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["route"])' "$DKL_NEW/admission.json")
if [ "$DKL_ROUTE" = RUN_RESIDUAL_ABLATION ]; then
  mkdir "$DKL_NEW/formal_run_once.lock" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$DKL_PY" "$DKL_NEW/code/run_residual.py" --root "$DKL_NEW"
  "$DKL_PY" "$DKL_NEW/code/score_residual.py" formal --root "$DKL_NEW" --labels-from "$DKL_BUNDLE/scoring_B/labels_A.json"
fi
```

若STOP_DIAGNOSTIC_ONLY，结束于A，不运行替代实验；若BLOCKED/INVALID，保留原始失败并停止；预算为1200秒，不能扩大或删矩阵。评分器拒绝INVALID封存。不要直接使用原始runner的 `--resume`：强杀后最后1–99步可能尚未落盘，无法确认已消耗成本。永久运行锁用于拒绝再次进入同一正式目录。完整Adam检查点的合成恢复已验证，但不把它解释为允许漏计未保存的强杀成本。

B完成或终止封存且正式评分曝光后，不再训练、补跑或延长原实验。只读检查脚本不会运行forward/NLL/SVD或优化器。

## 证据与完整性

`FILES.sha256`覆盖包内文件；外层 `evidence.tar.gz.sha256`校验压缩包。`source_manifest.json`记录实际使用的历史源码、输入、状态与预测身份；`frozen_manifest.json`和`code_manifest.json`记录A评分前的冻结身份。原训练子集scaler与候选顺序原样复用。

```bash
cd "$DKL_BUNDLE"
sha256sum -c evidence.tar.gz.sha256
sha256sum -c FILES.sha256
```

本轮独立复核、实际命令和检查结果见 `audit/` 与 `verification/`。不以重新运行全仓测试或完整历史训练矩阵作为复算前提。
