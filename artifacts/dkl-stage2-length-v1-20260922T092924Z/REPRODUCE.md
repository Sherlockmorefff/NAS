# 训练长度诊断：复算与重新续训

当前目录是独立的开发性补充实验，原阶段2结果保持不变。所有完整重跑必须使用新目录；本说明不授权阶段3、扩展步数或等预算控制组比较。

## 仅复算稳定性与评分

证据包包含全部必要的A输入、选中训练子集、原60个带Adam的DKL源检查点、原封存预测、固定控制组预测、120个新检查点、每100步的紧凑诊断与无损float64核、独立评分标签。稳定性和评分复算不执行模型训练或重新预测。评分使用Python标准库；稳定性复算另需NumPy，不导入PyTorch。

在服务器执行如下命令；本地解包后只需替换 `artifact` 和可用Python路径。

```sh
set -eu
artifact=/home/fangxy/D-VAE/artifacts/dkl-stage2-length-v1-20260922T092924Z
python_bin=/home/fangxy/.conda/envs/nas/bin/python
result_dir=$(mktemp -d /tmp/dkl-length-recompute-XXXXXX)
"$python_bin" "$artifact/code/recompute_stability.py" --root "$artifact" --output-dir "$result_dir/stability"
"$python_bin" "$artifact/code/score_length.py" --root "$artifact" --output-dir "$result_dir/scoring"
```

输出目录使用全新临时目录，避免覆盖封存证据和原评分。结果时间/来源字段可变化；不同Python/NumPy版本可能有末位舍入差异，数值核对采用绝对/相对容差1e−12。只能评分注册的1000/2000/4000时刻，不计算其他诊断点的留出成绩。

在证据根目录运行 `sha256sum -c FILES.sha256` 检查全文件；macOS使用 `shasum -a 256 -c FILES.sha256`。外层包hash见 `evidence.tar.gz.sha256`。全文件清单排除自身、压缩包及外层PACKAGING.json，避免自引用；PACKAGING.json单列打包与回读校验耗时。

## 从包内原1000步状态重新续训

需要Python3.10.20、PyTorch2.5.1+cu121、GPyTorch1.15.2、NumPy2.2.6；计算实际仅使用CPU、float64、1线程。代码、输入和检查点已经包含在包内，不依赖原阶段2目录或任何GNN/VAE权重。环境安装不包含在包内。以下命令在现有服务器仓库的 `artifacts/` 下新建目录，以满足预检中只读 `git diff --check` 的工作目录要求。

```sh
set -eu
cd /home/fangxy/D-VAE
python_bin=/home/fangxy/.conda/envs/nas/bin/python
bundle=/home/fangxy/D-VAE/artifacts/dkl-stage2-length-v1-20260922T092924Z
new_run="/home/fangxy/D-VAE/artifacts/dkl-stage2-length-v1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$new_run"
cp -R "$bundle/code" "$new_run/code"
cp -R "$bundle/source" "$new_run/source"
cp "$bundle/source_manifest.json" "$bundle/worklist.json" "$new_run/"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" "$new_run/code/preflight_length.py" --root "$new_run"
"$python_bin" "$new_run/code/prepare_length.py" freeze --root "$new_run"
"$python_bin" "$new_run/code/run_length.py" --root "$new_run" > "$new_run/run_stdout.log" 2> "$new_run/run_stderr.log"
mkdir -p "$new_run/audit"
cp "$bundle/audit/verify_length_saved.py" "$new_run/audit/verify_length_saved.py"
"$python_bin" "$new_run/audit/verify_length_saved.py" --root "$new_run"
"$python_bin" "$new_run/code/recompute_stability.py" --root "$new_run"
"$python_bin" "$new_run/code/prepare_length.py" release-scoring --root "$new_run" --labels-from "$bundle/scoring/labels_A.json"
"$python_bin" "$new_run/code/score_length.py" --root "$new_run"
```

`mkdir` 拒绝已有目录。只复制原1000步来源和实现，不复制已续训的 `latest/`、`checkpoints/`、资源账本或新评分。原scheduler保留planned1000/completed1000；恢复Adam后实际LR固定为映射1e−4、GP1e−3，2000保存不重建优化器，全部轨迹继续至4000。新任务的协议hash因目录名和冻结时间变化，源状态hash保持不变。

新运行的合成预检会直接检验已修复实现，不需重复本次冻结前失败与定位尝试。因此验证更新数和耗时可与本证据不同；正式新增更新数固定180000。只有在完整矩阵下才运行完整状态核验；若预算到限，按PARTIAL保留已有结果并统一封存后评分，不扩大预算或选点补跑。正式运行非零退出时先检查 `failure.json`，不要无条件执行后续评分。

如果进程意外中断且尚未形成 `predictions_manifest.json`，可用同一命令追加 `--resume`。恢复必须保留完整 `latest/`、诊断、源清单和资源账本。若账本步数与检查点不一致，运行器会阻断而不是静默重跑；需保存原记录后明确处理。达到预算并封存或已曝光新评分后，本轮不能恢复补跑。

## 证据入口

- `protocol.json` / `frozen_manifest.json` / `source_manifest.json` / `code_manifest.json`：更新开始前冻结的协议、代码和全部来源。
- `source/`：只含本任务需要的原DKL源检查点、原预测、固定对照无标签预测、A训练快照和查询输入；没有B/test。
- `verification/`：1071次合成更新及冻结前失败/定位/复测记录；60条原1000步复现检查。
- `checkpoints/`：2000/4000带Adam检查点；`latest/`：各轨迹可恢复的最终进度。
- `diagnostics/` / `diagnostics_manifest.json`：每100步更新后NLL、预测、核与漂移；均无查询y/SE。
- `stability_summary/`：1860时刻、1800窗口和各规模/检查点稳定数量、失败项、ID、范围与分位数。
- `predictions_manifest.json` / `predictions/`：全部三个预定科学时刻统一封存。
- `scoring/`：封存后释放的A标签、regret5/gain5、随机参考、校准、固定对照、逐轨迹/种子/折统计。
- `resource_ledger.json` / `trajectories/`：本轮新增180000更新与验证分开；原阶段2耗时不重复计费。
- `audit/`：独立状态/评分复核，以及本次冻结前修正说明。

本轮为A池原结果曝光后的开发性补充。稳定性不是数学收敛；4000步DKL与旧控制组不是等预算比较；本任务没有进行阶段3或搜索。
