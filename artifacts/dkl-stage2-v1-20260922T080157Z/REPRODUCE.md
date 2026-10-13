# 复算与重跑说明

本目录是已完成的阶段2证据，禁止在这里继续训练或覆盖原评分。所有命令只使用合法A验证数据，不需要B/test、数据集下载或VAE/GNN权重。下述完整重跑命令仅供复现；本次没有执行第二轮或阶段3。

## 1. 仅复算评分：不触发训练

评分器仅使用Python标准库，读取已封存预测及其终点文件hash，再读取独立的 `scoring/labels_A.json`。不加载PyTorch、不进行模型预测或优化。证据包已经包括所有必需数据；本地或服务器均可运行。

在服务器执行：

```sh
set -eu
artifact=/home/fangxy/D-VAE/artifacts/dkl-stage2-v1-20260922T080157Z
score_out=$(mktemp -d /tmp/dkl-stage2-score-XXXXXX)
python3 "$artifact/code/score_stage2.py"   --artifact-dir "$artifact" --output-dir "$score_out"
```

本地解包后将 `artifact` 改为实际解包目录。预计输出 `COMPLETE_VALID`、200终点、n20“未见相对继续回归的收益”、n50“局部或混合信号”。逐单元、逐折、成本、校准CSV及随机参考应与原输出数值一致；不同Python版本可出现约1e-15量级末位舍入差异，复核使用1e-12绝对/相对容差；`summary.json` 和 `score_provenance.json` 的评分时间、来源字段可以不同。绝不要将输出目录指向原 `scoring/`。

文件完整性检查：在证据根目录运行 `sha256sum -c FILES.sha256`；macOS可用 `shasum -a 256 -c FILES.sha256`。外层压缩包的hash见同名 `.sha256` 文件。`FILES.sha256` 不包含自身及外层包，避免自引用。

## 2. 按原协议重跑：必须使用全新任务目录

原环境：Python3.10.20，PyTorch2.5.1+cu121，GPyTorch1.15.2，NumPy2.2.6，Linux；实际计算仅CPU、float64、1线程。服务器现有解释器为 `/home/fangxy/.conda/envs/nas/bin/python`。跨库/平台版本的浮点结果可能不同，不保证不同版本逐字节一致。

训练适配器保留了严格的原始A来源身份检查。重跑依赖服务器原始来源：

- `/home/fangxy/D-VAE/results/posthoc/fuzzy-scheduler-regularization-v1-20260921T064826Z/` 中的公开输入、协议及协议指定的A验证标签；标签字节hash为 `d6823a6dbe62507f4fdc53de01a69e33cdd3ec8de4fc0aae0cca89563766fa58`。
- 原架构分组 `/home/fangxy/D-VAE/results/posthoc/fuzzy-metric-stage2-20260910-v1/partition.json`；只使用其中A的ID与分组元数据。文件存在但身份错误时停止，只有确实不存在时协议才允许备用分组；此时不能宣称与本次逐折复现相同。

精简快照包括179条A输入、20个训练子集的独立标签/标准化快照及封存后的A评分标签。快照可独立复算评分、核对训练输入；原训练入口仍要求原始来源存在，不能把完整A评分字典直接传给拟合函数。

在服务器仓库内运行下列流程。`mkdir` 会拒绝已存在目录，避免覆盖证据。先完成新增风险预检，再冻结协议和代码，之后才释放正式训练标签。完整流程固定200终点、344000正式更新，含验证的累计拟合/预测预算7200秒；预检合成更新和耗时单列。不要同时启动多个矩阵进程。

```sh
set -eu
cd /home/fangxy/D-VAE
python_bin=/home/fangxy/.conda/envs/nas/bin/python
original=/home/fangxy/D-VAE/artifacts/dkl-stage2-v1-20260922T080157Z
new_run="/home/fangxy/D-VAE/artifacts/dkl-stage2-v1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$new_run"
cp -R "$original/code" "$new_run/code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" "$new_run/code/stage2_data.py" prepare --root "$new_run"
"$python_bin" "$new_run/code/preflight_stage2.py" --root "$new_run"
"$python_bin" "$new_run/code/freeze_stage2.py" --root "$new_run"
"$python_bin" "$new_run/code/run_stage2.py" --root "$new_run"   > "$new_run/run_stdout.log" 2> "$new_run/run_stderr.log"
"$python_bin" "$new_run/code/stage2_data.py" release-scoring   --root "$new_run" --seal "$new_run/predictions_manifest.json"
"$python_bin" "$new_run/code/score_stage2.py" --artifact-dir "$new_run"
```

预检或正式运行非零退出时，先检查错误与封存状态；不要无条件执行后续评分。完整矩阵才适用预注册分类，缺失项如实标为PARTIAL。评分曝光后不得恢复训练。若仅训练进程中断且尚未封存，可使用同一运行器 `--resume` 恢复完整参数/优化器/步数/学习率阶段/RNG；不手动重置预算或选择种子。

新目录时间戳、冻结时间、资源时间及这些文件的hash会变化；科学配方、源码、分组、子集及固定更新数应保持一致。当前证据的独立矩阵核验程序默认固定当前协议hash，验证新运行时必须显式传入新 `protocol.json` 的SHA256：

```sh
mkdir -p "$new_run/audit"
cp "$original/audit/verify_matrix.py" "$new_run/audit/verify_matrix.py"
protocol_hash=$(sha256sum "$new_run/protocol.json" | cut -d ' ' -f 1)
"$python_bin" "$new_run/audit/verify_matrix.py"   --root "$new_run" --expected-protocol-sha256 "$protocol_hash"
```

该核验只检查保存状态、分组、步数和账本，不训练、不重新预测、不读评分标签；默认针对完整矩阵。建议正式复现实验在评分释放前执行核验。

## 3. 证据入口

- `README.md` / `status.json`：科学结论与执行状态。
- `frozen_manifest.json` / `code_manifest.json`：标签释放前冻结的协议和源码身份。
- `public_inputs.json` / `folds.json` / `subsets.json` / `worklist.json`：仅A的输入和固定矩阵。
- `training/`：每次允许的20/50条训练记录与标准化器。
- `states/`：60共同状态与200固定终点；`checkpoints/`：400块的最终恢复状态，无每步模型堆积。
- `predictions_manifest.json` / `predictions/`：统一封存以及不含查询真值的预测。
- `scoring/`：独立评分标签、原始指标、三层汇总、配对、随机参考、校准及成本。
- `training_traces.csv` / `block_summaries.csv` / `resource_ledger.json`：3440个诊断点、400块和344000次正式更新账本。
- `verification/` / `audit/`：合成预检、冻结前修正记录、独立矩阵/评分/训练诊断复核。

本次所有正式训练仅运行一次；冻结前修正与针对性预检有保留，合计1437次合成更新。全新复现采用已修正源码直接预检，不需重复已经解决的问题的旧版检查，也不加载阶段01训练权重。
