"""Evidence reports and an explicit-whitelist, A-only share package."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import shlex
import tarfile

from .io import read_json, write_json, sha256, now
from .score import METHODS


def _read_optional(root, name, default):
    path = root / name
    return read_json(path) if path.exists() else default


def _number(value, digits=4):
    return "缺失" if value is None else f"{float(value):.{digits}f}"


def _table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows])


def _starts(root):
    records = []
    for path in sorted((root / "settings").glob("*/start[012].json")):
        value = read_json(path)
        records.append({"setting_id": path.parent.name, **value})
    return records


def _commands(root):
    path = root / "commands.jsonl"
    commands = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            argv = row.get("argv", [])
            # __main__.py appears as argv[0]; module syntax is portable to the saved repo.
            if argv and str(argv[0]).endswith("__main__.py"):
                argv = ["-B", "-m", "analyse.structured_kernel_phase1", *argv[1:]]
            command = shlex.join([row["executable"], *argv])
            commands.append(f"- {row.get('at', 'time unavailable')}，目录 `{row.get('cwd', '')}`\n\n  ```sh\n  {command}\n  ```")
    return "\n\n".join(commands) or "尚无命令记录；不得声称已执行。"


def build_report(run_dir, repo):
    root, repo = Path(run_dir).resolve(), Path(repo).resolve()
    status = read_json(root / "status.json")
    gate = _read_optional(root, "engineering_gate.json", {"status": "INCOMPLETE", "selected_method": None, "all_settings_complete": False})
    summary = _read_optional(root, "metrics_summary.json", [])
    legacy = _read_optional(root, "legacy_summary.json", None)
    sources = _read_optional(root, "source_manifest.json", {})
    features = _read_optional(root, "feature_manifest.json", {})
    baseline = _read_optional(root, "baseline.json", {})
    progress = _read_optional(root, "progress.json", {})
    starts = _starts(root)
    entries = status.get("settings", {})
    counts = {state: sum(row["state"] == state for row in entries.values())
              for state in ("COMPLETE", "FAILED", "BLOCKED", "RUNNING", "PENDING")}
    successful_starts = sum(row.get("success") is True for row in starts)
    warning_starts = sum(bool(row.get("convergence_warning")) for row in starts)
    adam_updates = sum(row.get("adam_updates", 0) for row in starts)
    lbfgs_iterations = sum(row.get("lbfgs_iterations", 0) for row in starts)
    lbfgs_evaluations = sum(row.get("lbfgs_evaluations", 0) for row in starts)
    lbfgs_over125 = sum(row.get("lbfgs_evaluations", 0) > 125 for row in starts)
    lbfgs_max_evaluations = max((row.get("lbfgs_evaluations", 0) for row in starts), default=0)
    complete = counts["COMPLETE"] == 100 and len(starts) == 300 and successful_starts == 300 and gate.get("all_settings_complete") is True
    result_status = gate.get("status", "INCOMPLETE") if complete else "INCOMPLETE"
    lines = ["# 第一轮 E0–E2 证据报告", "", f"生成时间：{now()}。结果状态：**{result_status}**。",
             "本报告仅涉及 A 上的探索性代理比较；未进行 B/test 确认或 KG 闭环。",
             "", "## 1. 已核实事实与仍缺失的证据", "",
             f"执行仓库：`{repo}`；branch `{baseline.get('branch', '缺失')}`；commit `{baseline.get('commit', '缺失')}`。用户原有修改记录见 `baseline.json`。",
             f"E0 完整来源清单见 `source_manifest.json`，已核验 A 验证行数 {sources.get('A_validation_rows_verified', '缺失')}，训练上下文 {sources.get('training_contexts_verified', '缺失')}。",
             "配方为 Cora/public/NormalizeFeatures/full-batch/global4/No-LN，300 epochs、patience 80，GNN seeds 0/1/2 验证均值。SE 已是三种子均值标准误，known_var=SE²；未再除 3。",
             "原 5 fold 与两个抽样 draw 恢复，n20 是 n50 的有序前缀；同架构 HP 组合不跨 fold。详情与逐行核验见 `preflight_report.md`、`candidate_manifest.csv`、`split_manifest.json`。",
             f"有向算子 WL 仅使用 A 的无标签结构，H=2；字典分层大小 {features.get('dictionary_sizes', '缺失')}，phi={features.get('phi_dimension', '缺失')}，psi={features.get('psi_dimension', '缺失')}，碰撞架构比例 {features.get('collision_architecture_fraction', '缺失')}。没有把 HP4 重复放入 WL 标签。",
             "VAE 的历史日志、训练源码及迁移记录支持无性能监督的重建/KL训练；原始训练 pkl 的字节身份和精确图成员没有历史完整记录，不能宣称这些已经验证。",
             f"来源审计记录 B 标签解码数={sources.get('B_label_rows_decoded', '缺失')}，节点 test 分数解码数={sources.get('test_scores_decoded', '缺失')}。",
             "", "## 2. 完成数量、优化预算与数值状态", "",
             _table(["COMPLETE", "FAILED", "BLOCKED", "RUNNING", "PENDING"], [[counts[key] for key in counts]]), "",
             f"静态 GP 设置：{counts['COMPLETE']}/100 完整成功；起点文件 {len(starts)}/300，其中成功 {successful_starts}，失败 {len(starts)-successful_starts}。",
             f"记录 Adam 更新 {adam_updates}（满预算 90000）；L-BFGS 实际迭代合计 {lbfgs_iterations}、函数评估合计 {lbfgs_evaluations}。L-BFGS 每起点配置 max_iter=100/max_eval=125，不要求耗尽预算才可完成。",
             f"PyTorch strong_wolfe 的 max_eval=125 是配置停止阈值；一次线搜索结束后计数可能越过阈值，并非硬上限。实际函数评估>125 的起点数={lbfgs_over125}，单起点最大值={lbfgs_max_evaluations}；全部方法使用同一规则，保留实际计数，没有自动扩展配置。",
             f"convergence_warning 起点数 {warning_starts}。达到冻结预算且目标/参数/posterior有限表示预算内完成，不证明收敛；没有自动延长训练或追加起点。",
             "每个设置仅使用最终训练 NLL 最小的起点；三起点不是独立实验重复，也不组成 ensemble。所有起点摘要/曲线与选中模型身份保留在 `settings/` 和 `fit_log.jsonl`。",
             "预测封存后由独立 scorer 读取 Q 标签；`scorer_integrity.json` 核对模型/预测/起点和冻结文件未被评分修改。若 latent 方差只存在 [-1e-8,0) 数值舍入值，保留原值，latent 区间明确缺失；前瞻/事后方差按原值加噪声评分，不裁剪。"]
    if not complete:
        lines += ["", "**尚未满足 100 设置/300 起点完整成功条件，不能宣布任何方法通过。**",
                  f"当前进程记录 PID={status.get('pid', progress.get('pid', '无记录'))}，运行状态={status.get('run_state', '未记录')}；此记录不是实时进程存活证明。"]
    failures = gate.get("failures", [])
    if failures:
        lines += ["", _table(["setting", "失败或阻塞原因"], [[row.get("setting_id", ""), row.get("reason", "")] for row in failures])]
    lines += ["", "## 3. 匹配静态 GP 结果", "",
              "主指标 MAE、width、IS90 的单位为百分点；NLPD 是 raw accuracy（0–1）密度上的 nat，可以为负。coverage 为比例。主分布方差使用 latent+extra_noise+median_T(known_var)，不使用 Q 的真实 SE。",
              "先在同 fold/n 内平均两个 draw，再对 5 fold 等权平均。缺少任一 draw 时该 fold 不做部分平均；缺 fold 时不报告该方法/n 的总体均值。",
              _table(["n", "方法", "MAE pp", "前瞻 NLPD nat", "IS90 pp", "coverage90", "width90 pp"],
                     [[row["n"], row["method"], *[_number(row.get(key)) for key in ("mae_pp", "nlpd", "is90_pp", "coverage90", "width90_pp")]]
                      for row in sorted(summary, key=lambda row: (-row["n"], METHODS.index(row["method"])))]),
              "", "逐上下文/逐 fold 指标、latent 区间、retrospective 噪声分解、Spearman、真实前10%偏差、预测前5偏差及 regret@5 见 `metrics_by_context.csv`、`metrics_by_fold.csv`、`predictions.csv`。retrospective 字段只能用于事后噪声解释。",
              "", "## 4. 成对消融的支持范围", ""]
    comparison_rows = []
    for row in gate.get("comparisons", []):
        if row["status"] != "COMPLETE":
            comparison_rows.append([row["n"], f"{row['target']} vs {row['baseline']}", "缺失", "缺失", "缺失"])
        else:
            comparison_rows.append([row["n"], f"{row['target']} vs {row['baseline']}", _number(row["mae_improvement_pp"]),
                                    _number(row["nlpd_change_nat"]), _number(row["is90_change_pp"])])
    lines += [_table(["n", "比较", "MAE改善 pp（正为好）", "NLPD变化（正为坏）", "IS90变化（正为坏）"], comparison_rows), "",
              "L-I vs L-A 检验这一固定 latent 表示下加入显式乘积项的效果；W-A vs L-A 检验这一加法核下表示替换的效果；W-I vs W-A 检验 WL 表示下的显式乘积项。差值反映本轮固定条件，不能推广成所有表示/核族的因果结论。",
              "WL 更好不证明 VAE 已丢失信息；交互更好不证明原整体核完全没有交互。重叠 fold/draw 和架构内 HP 不是独立样本，这里不计算简单独立样本显著性。"]
    for target, baseline_method in (("L-I", "L-A"), ("W-A", "L-A"), ("W-I", "W-A")):
        row = next((r for r in gate.get("comparisons", []) if r["target"] == target and r["baseline"] == baseline_method and r["n"] == 50), None)
        if row is None or row["status"] != "COMPLETE":
            lines += [f"n50 {target} vs {baseline_method}：数据不完整，不能给出配对总体结论。"]
            continue
        gain = row["mae_improvement_pp"]
        direction = "降低" if gain > 0 else ("增加" if gain < 0 else "不变")
        evidence = "支持此固定条件下的均值误差改善" if gain > 0 else "没有显示此固定条件下的均值误差改善"
        lines += [f"n50 {target} vs {baseline_method}：MAE {direction} {_number(abs(gain))} pp；前瞻 NLPD 变化 {_number(row['nlpd_change_nat'])} nat，IS90 变化 {_number(row['is90_change_pp'])} pp。该配对结果{evidence}；是否达到完整工程门槛须同时看与 RAW16 的比较。"]
    lines += ["", "## 5. 旧 DKL 机制证据", ""]
    if legacy is None:
        lines += ["**E1 未完成：缺少 legacy_summary.json。未声称完成旧 checkpoint 合法性审计，也不能据此判断 DKL 机制。**"]
    else:
        lines += [f"E1 状态：{legacy.get('status', '缺失')}；新增 optimizer updates={legacy.get('new_optimizer_updates', '缺失')}；diagnostic rows={legacy.get('diagnostic_rows', '缺失')}；缺失记录数={legacy.get('missing_rows', '缺失')}。",
                  "旧族为 stage2/residual 的 Matérn5/2、constant mean、FixedNoise+extra noise。生产 DKL 的 ARD RBF/不同网络属于另一族；历史优化预算也不同，旧 RAW 不替代本轮匹配 RAW16。",
                  "合法性/训练快照/架构隔离审计见 `legacy_summary.json` 的 checkpoint_audit；来源 hash 与缺失轨迹见 `legacy_source_manifest.json`、`legacy_missing.csv`。",
                  _table(["family", "method", "n", "step", "Q MAE pp", "前瞻 NLPD", "T有效秩"],
                         [[row.get("family"), row.get("method"), row.get("n"), row.get("step"), _number(row.get("q_mae_pp")),
                           _number(row.get("q_nlpd_prospective")), _number(row.get("T_effective_rank"))]
                          for row in legacy.get("endpoint_aggregates", [])]), "",
                  _table(["family/method/n", "step", "Δtrain NLL", "ΔQ MAE pp", "Δ前瞻 NLPD", "ΔT有效秩", "三联MAE证据folds"],
                         [[f"{row.get('family')}/{row.get('method')}/{row.get('n')}", f"{row.get('from_step')}→{row.get('to_step')}",
                           *[_number(row.get(key)) for key in ("delta_train_nll", "delta_q_mae_pp", "delta_q_nlpd_prospective", "delta_T_effective_rank")],
                           f"{row.get('supporting_folds_NLL_improves_MAE_worsens_T_rank_drops', '缺失')}/{row.get('fold_count', '缺失')}"]
                          for row in legacy.get("mechanisms", {}).get("means", [])]),
                  "核有效秩 r_eff=(tr R)²/tr(R²)，R 是指定 T 或 T∪Q 无标签输入上的归一化先验核相关矩阵；两种样本集的诊断字段分别标记，不能混读。",
                  "事实是表中的已保存检查点差值；训练 NLL 改善、Q 恶化与有效秩下降同时出现，才支持过拟合/过度相关解释，仍不是因果证明。只有映射距离缩小不构成坍缩证据，因为 lengthscale 可补偿；覆盖率提高或区间变宽也不等于均值改善。",
                  "没有用 Q 曲线挑选最佳 checkpoint，没有为补齐 0/10/50/100/300/1000 轨迹重训。缺失点在 missing 清单保留。"]
        for row in legacy.get("mechanisms", {}).get("means", []):
            if row.get("n") != 50:
                continue
            changes = [row.get(key) for key in ("delta_train_nll", "delta_q_mae_pp", "delta_T_effective_rank")]
            known = all(value is not None for value in changes)
            joint = known and changes[0] < 0 and changes[1] > 0 and changes[2] < 0
            verdict = "同时出现" if joint else ("未同时出现" if known else "缺失，无法核实")
            lines += [f"n50 {row.get('family')}/{row.get('method')}，step {row.get('from_step')}→{row.get('to_step')}：总体均值变化中，训练 NLL 改善＋Q MAE 恶化＋T 核有效秩下降{verdict}；逐 fold 同时满足为 {row.get('supporting_folds_NLL_improves_MAE_worsens_T_rank_drops', '缺失')}/{row.get('fold_count', '缺失')}，逐 draw-context 为 {row.get('supporting_draw_contexts_NLL_improves_MAE_worsens_T_rank_drops', '缺失')}/{row.get('draw_context_count', '缺失')}。这限定了过拟合/过度相关解释的实际支持范围。"]
    selected = gate.get("selected_method") if complete else None
    lines += ["", "## 6. 工程门槛与唯一候选", "",
              "门槛：n50 MAE 相对 RAW16 至少降低 0.3 pp，至少4/5 fold改善，前瞻 NLPD 恶化≤0.1 nat，IS90恶化≤5%；n20 MAE恶化≤0.3 pp；全部科学设置完整成功。它们是项目工程门槛，不是理论或显著性保证。",
              _table(["方法", "通过", "n50 MAE改善 pp", "改善fold数"],
                     [[row["method"], row.get("passed", False), _number(row.get("n50_mae_improvement_pp")), row.get("n50_positive_fold_count", "缺失")]
                      for row in gate.get("candidates", [])]),
              f"唯一待第二轮确认方法：**{selected if selected else '无'}**。完整门槛逐项结果和 <0.05 pp/<0.01 nat 的固定并列裁决见 `engineering_gate.json`。",
              "", "## 7. 下一轮建议与停止边界", ""]
    if not complete or legacy is None:
        recommendation = "先补齐当前明确缺失或失败的 E0–E2 工程证据，保持科学规格；不据不完整结果启动容量、CLML、KG或B确认。"
    elif selected:
        recommendation = f"最小下一步是对 {selected} 与匹配 RAW16 制定单独的第二轮确认及同采集规则比较；当前生产入口实际为 qLogEI，若下一轮要求 KG，先单独冻结 KG 实现与预算。本报告只推荐，不执行。"
    else:
        recommendation = "本轮为 NO_STABLE_GAIN，停止继续增加代理复杂度或扫参。若旧证据的 NLL/Q/有效秩三联变化明确，可另行审批一个最小容量或 CLML 判别实验；本轮证据不自动授权任一分支。"
    lines += [recommendation, "", "## 8. 执行与复现", "",
              "所有新静态 GP 使用 CPU/float64、每进程单线程，当前 workers=1。计时来自实际起点日志；剩余估计只用已完成设置时间，见 `progress.json`，不是预先保证。",
              "本轮代码及 diff、真实执行命令、冻结 protocol/来源/feature/split 身份，均随证据包保留。checkpoint 权重留在服务器原运行目录；分享包默认不含权重或任何 B/test 文件。",
              "复核结果从 `metrics_summary.csv`、`paired_comparisons.csv`、`engineering_gate.json`、`legacy_summary.json` 与 `mapping_diagnostics.csv` 开始。",
              "", "### 真实执行命令", "", _commands(root)]
    (root / "README_RESULTS.md").write_text("\n\n".join(lines) + "\n")
    request = ["# 后续 GPT 分析请求", "", "请分析本证据包内的第一轮 E0–E2 结果。只依据实际文件，引用具体数值、表名/字段和文件路径。",
               "先核实 status.json 的 100 设置与300起点完整性、engineering_gate.json、preflight_report.md 及 E1 缺失清单；不把缺失或失败记录排除后宣称通过。",
               "分别讨论事实、解释、尚待检验的假说。n50 为主，n20 完整报告；主配套指标只用前瞻方差，retrospective 另列。讨论三个成对消融和各组 vs 匹配RAW16。",
               "WL 更好不能解释为已证明 VAE 丢信息；交互更好不能解释为原模型完全没有交互。A 的探索性收益不是 B/test 验证，更不是搜索成功。",
               "DKL 过拟合/过度相关解释要同时检查 train NLL 改善、Q恶化和有效秩下降；说明 R 的样本集，不能只看距离缩小、coverage增加或区间变宽。不要按 Q 曲线选择 checkpoint。",
               "当前生产采集是 qLogEI；KG未执行。推荐证据支持的最小下一步：同规则搜索确认、一个容量/CLML判别实验，或停止增加复杂度。不要重启已无稳定收益的 fuzzy metric、ARCH_FLOOR、延长训练、参数扫描路线。",
               f"当前自动门槛状态：{result_status}；待确认候选：{selected or '无'}。当前建议：{recommendation}"]
    (root / "GPT_ANALYSIS_REQUEST.md").write_text("\n\n".join(request) + "\n")
    (root / "actual_commands.md").write_text("# 真实执行命令\n\n" + _commands(root) + "\n")
    report_status = {"generated_at": now(), "E2_complete": bool(complete), "E1_present": legacy is not None,
                     "result_status": result_status, "selected_method": selected, "counts": counts,
                     "starts_recorded": len(starts), "starts_successful": successful_starts,
                     "convergence_warnings": warning_starts, "new_GNN_trainings": 0, "KG_runs": 0}
    write_json(root / "report_status.json", report_status)
    return report_status


def package_run(run_dir, repo):
    """Whitelist this run's summaries, A-only features/labels and exact code only."""
    root, repo = Path(run_dir).resolve(), Path(repo).resolve()
    freeze = read_json(root / "frozen_manifest.json")
    allowed_root = {
        "PLAN.md", "PROMPT.txt", "protocol.json", "candidate_manifest.csv", "split_manifest.json",
        "feature_manifest.json", "source_manifest.json", "public_features.json", "preflight_report.md",
        "status.json", "settings.json", "baseline.json", "environment.json", "progress.json", "pilot_validation.json",
        "frozen_manifest.json", "prediction_seal.json", "scorer_integrity.json", "fit_log.jsonl", "commands.jsonl",
        "predictions.csv", "metrics_by_context.csv", "metrics_by_fold.csv", "metrics_summary.csv", "metrics_summary.json",
        "engineering_gate.json", "paired_comparisons.json", "paired_comparisons.csv", "README_RESULTS.md",
        "GPT_ANALYSIS_REQUEST.md", "actual_commands.md", "report_status.json", "legacy_summary.json",
        "legacy_source_manifest.json", "mapping_diagnostics.csv", "mapping_diagnostics_missing.csv",
        "mapping_diagnostics_missing.json", "legacy_access_audit.json", "legacy_checkpoint_audit.json", "legacy_missing.csv",
        "verification.json", "test_output.txt", "runner.log", "execution_ledger.json",
    }
    files = [(path, path.name) for path in sorted(root.iterdir()) if path.is_file() and path.name in allowed_root]
    files.extend((path, path.name) for path in sorted(root.glob("error_*.json")) if path.is_file())
    for name in ("training", "queries"):
        files.extend((path, str(path.relative_to(root))) for path in sorted((root / name).glob("*.json")))
    allowed_setting = {"identity.json", "prediction.json", "metrics.json", "failure.json"}
    allowed_setting |= {f"start{start}.{suffix}" for start in range(3) for suffix in ("json", "jsonl")}
    for path in sorted((root / "settings").glob("*/*")):
        if path.is_file() and path.name in allowed_setting:
            files.append((path, str(path.relative_to(root))))
    diff = []
    for relative, expected in freeze["code_files"].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or sha256(path) != expected:
            raise ValueError("Cannot package changed or escaped source " + relative)
        files.append((path, "code/" + relative))
        diff.extend(difflib.unified_diff([], path.read_text().splitlines(keepends=True), fromfile="/dev/null", tofile="b/" + relative))
    (root / "code_diff.patch").write_text("".join(diff))
    files.append((root / "code_diff.patch", "code_diff.patch"))
    manifest = {"created_at": now(), "files": {arc: sha256(path) for path, arc in files},
                "scope": "Only current-run A-only artifacts and frozen source whitelist",
                "excluded": ["all .pt weights (retained on server)", "B labels/SE", "node test scores", "credentials", "unrelated results"]}
    write_json(root / "package_manifest.json", manifest)
    files.append((root / "package_manifest.json", "package_manifest.json"))
    output = root / "evidence_package.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        for path, arc in files:
            if path.is_symlink():
                raise ValueError("Refuse to package symlink " + str(path))
            archive.add(path, arcname=arc, recursive=False)
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        if sorted(names) != sorted(arc for _, arc in files) or any(name.endswith(".pt") for name in names):
            raise AssertionError("Package whitelist verification failed")
    result = {"path": str(output), "sha256": sha256(output), "files": len(files), "bytes": output.stat().st_size,
              "excluded_B_and_node_test_sources": True}
    write_json(root / "package_result.json", result)
    return result
