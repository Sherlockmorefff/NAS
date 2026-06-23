"""Generate a static audit report for the NAS + HPO pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_BUNDLE_FILES = [
    "search_runs_summary.csv",
    "final_eval_summary.csv",
    "nas_hpo_contribution_summary.csv",
    "manifest.json",
    "analysis_notes.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NAS + HPO pipeline audit report")
    parser.add_argument("--results_bundle", type=str, default="results_analysis_bundle")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output", type=str, default="results_analysis_bundle/nas_pipeline_audit_report.md")
    parser.add_argument("--include_source_static_summary", action="store_true")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def read_csv(path: Path, warnings: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        warnings.append(f"warning: missing CSV: {path}")
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        warnings.append(f"warning: failed to read CSV {path}: {exc}")
        return []


def read_json(path: Path, warnings: list[str]) -> Any:
    if not path.exists():
        warnings.append(f"warning: missing JSON: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"warning: failed to read JSON {path}: {exc}")
        return None


def read_text(path: Path, warnings: list[str]) -> str:
    if not path.exists():
        warnings.append(f"warning: missing text file: {path}")
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        warnings.append(f"warning: failed to read text file {path}: {exc}")
        return ""


def metric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = safe_float(row.get(key))
        if value is not None:
            return value
    return None


def best_row(rows: list[dict[str, Any]], *keys: str) -> dict[str, Any] | None:
    scored = []
    for row in rows:
        value = metric(row, *keys)
        if value is not None:
            scored.append((value, row))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def fmt(value: Any) -> str:
    num = safe_float(value)
    if num is None:
        return "not available"
    return f"{num:.4f}"


def table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def artifact_inventory(bundle: Path, warnings: list[str]) -> list[str]:
    lines = []
    if not bundle.exists():
        warnings.append(f"warning: results_bundle does not exist: {bundle}")
        return ["- results bundle: not available"]
    for expected in EXPECTED_BUNDLE_FILES:
        path = bundle / expected
        if not path.exists():
            warnings.append(f"warning: missing expected bundle file: {path}")
    for path in sorted(bundle.glob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md"}:
            lines.append(f"- {path.name}")
    return lines or ["- no CSV/JSON/MD artifacts detected"]


def candidate_name(row: dict[str, Any]) -> str:
    return str(row.get("candidate_name") or row.get("name") or "")


def group_name(row: dict[str, Any]) -> str:
    return str(row.get("group") or "")


def find_candidate(rows: list[dict[str, Any]], logical_name: str) -> dict[str, Any] | None:
    for row in rows:
        name = candidate_name(row)
        group = group_name(row)
        if logical_name == "manual_searched" and (name == "Manual_2xGCN_searchedHP" or group == "hp_on_manual"):
            return row
        if logical_name == "nas_full" and (name == "NAS_full" or group == "full"):
            return row
        if logical_name == "manual_default" and ("BASE_2xGCN_defaultHP" in name or group == "baseline"):
            return row
    return None


def contribution_finding(contrib_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if contrib_rows:
        found_comparison = False
        for row in contrib_rows:
            label = f"{row.get('hp_mode')}/{row.get('source_run_id') or 'unlinked'}"
            nas_vs_manual = safe_float(row.get("nas_vs_manual_under_searched_hp"))
            hpo_gain = safe_float(row.get("hpo_gain"))
            nas_gain = safe_float(row.get("nas_arch_gain_default_hp"))
            if nas_vs_manual is not None:
                found_comparison = True
                if nas_vs_manual <= 0:
                    lines.append(
                        f"- {label}: NAS_full does not exceed Manual_2xGCN_searchedHP "
                        f"({fmt(nas_vs_manual)} delta). Current results more strongly support searched HP, "
                        "not a claim that searched NAS architecture beats manual 2xGCN + searched HP."
                    )
                else:
                    lines.append(
                        f"- {label}: NAS_full exceeds Manual_2xGCN_searchedHP by {fmt(nas_vs_manual)}; "
                        "interpret with seed validity and metric caveats."
                    )
            if hpo_gain is not None and nas_gain is not None and hpo_gain > nas_gain + 0.005:
                lines.append(f"- {label}: HPO gain is larger than architecture-only gain.")
        if not found_comparison:
            lines.append("- NAS_full versus Manual_2xGCN_searchedHP: not available.")
        return lines

    manual = find_candidate(final_rows, "manual_searched")
    full = find_candidate(final_rows, "nas_full")
    if manual and full:
        manual_test = metric(manual, "test_mean_valid_only", "test_mean")
        full_test = metric(full, "test_mean_valid_only", "test_mean")
        delta = None if manual_test is None or full_test is None else full_test - manual_test
        if delta is not None and delta <= 0:
            lines.append(
                "- NAS_full does not exceed Manual_2xGCN_searchedHP. Current results more strongly support "
                "searched HP; they do not prove searched NAS architecture beats manual 2xGCN + searched HP."
            )
        else:
            lines.append(f"- NAS_full versus Manual_2xGCN_searchedHP delta: {fmt(delta)}.")
    else:
        lines.append("- NAS_full versus Manual_2xGCN_searchedHP: not available.")
    return lines


def source_static_summary(project_root: Path, warnings: list[str]) -> list[str]:
    files = [
        "hp_modes.py",
        "train_joint.py",
        "bo_phase3.py",
        "bo_phase4.py",
        "bo_phase4_tpe.py",
        "weighted_diag_gmm_init.py",
        "final_eval.py",
        "collect_experiment_results.py",
    ]
    lines = []
    for name in files:
        text = read_text(project_root / name, warnings)
        if not text:
            lines.append(f"- {name}: not available")
            continue
        lines.append(
            f"- {name}: {len(text.splitlines())} lines; "
            f"mentions hp_mode={text.count('hp_mode')}, "
            f"mentions valid={text.count('valid')}, "
            f"mentions gmm={text.lower().count('gmm')}."
        )
    return lines


def current_pipeline_table() -> list[str]:
    rows = [
        [
            "hp_mode semantic layer",
            "mode name and ops",
            "HP vector schema and masks",
            "hp_modes.py helpers",
            "select global/conditional dimensions",
            "inactive conditional HP can still appear in vectors",
        ],
        [
            "JointSpaceVAE training",
            "architecture and HP vectors",
            "joint latent model",
            "JointSpaceVAE",
            "learn z_arch and HP representation",
            "latent quality may mix architecture and HP effects",
        ],
        [
            "Phase3 joint search",
            "z_search history",
            "candidate evaluations",
            "BO-style search",
            "evaluate promising z_search",
            "validation objective differs from final test summary",
        ],
        [
            "weighted diagonal GMM initialization",
            "high-performing history",
            "warm-start z_search samples",
            "p(z_search | high-performing history)",
            "sample candidates near historical winners",
            "density is not predicted accuracy",
        ],
        [
            "Phase4 BO / TPE search",
            "history plus warm starts",
            "best_z_final and history",
            "BO-GP or TPE surrogate",
            "choose next real evaluations",
            "surrogate quality needs independent diagnosis",
        ],
        [
            "final_eval",
            "selected candidates",
            "multi-seed val/test summaries",
            "train_and_eval_arch",
            "retrain and summarize candidates",
            "invalid seed reporting must be visible",
        ],
        [
            "result aggregation",
            "existing results artifacts",
            "analysis bundle",
            "CSV/JSON/MD parser",
            "link search and final evaluation rows",
            "missing files or duplicate baselines can affect interpretation",
        ],
    ]
    lines = [
        table_row(["stage", "input", "output", "model", "decision logic", "potential risk"]),
        table_row(["---"] * 6),
    ]
    lines.extend(table_row(row) for row in rows)
    return lines


def safe_commands() -> list[str]:
    return [
        "python -m py_compile nas_pipeline_audit_report.py analyze_nas_hpo_contributions.py final_eval.py collect_experiment_results.py",
        "python analyze_nas_hpo_contributions.py --final_eval_summary results_analysis_bundle/final_eval_summary.csv --search_summary results_analysis_bundle/search_runs_summary.csv --output_csv results_analysis_bundle/nas_hpo_contribution_summary.csv --output_md results_analysis_bundle/nas_hpo_contribution_summary.md",
        "python nas_pipeline_audit_report.py --results_bundle results_analysis_bundle --results_root results --output results_analysis_bundle/nas_pipeline_audit_report.md",
        "python collect_experiment_results.py --results_root results --output results_analysis_bundle",
        "python final_eval.py --help",
    ]


def write_report(args: argparse.Namespace) -> None:
    project_root = Path.cwd()
    bundle = Path(args.results_bundle)
    results_root = Path(args.results_root)
    output = Path(args.output)
    warnings: list[str] = []

    inventory = artifact_inventory(bundle, warnings)
    if not results_root.exists():
        warnings.append(f"warning: results_root does not exist: {results_root}")

    search_rows = read_csv(bundle / "search_runs_summary.csv", warnings)
    final_rows = read_csv(bundle / "final_eval_summary.csv", warnings)
    contrib_rows = read_csv(bundle / "nas_hpo_contribution_summary.csv", warnings)
    manifest = read_json(bundle / "manifest.json", warnings)

    best_search = best_row(search_rows, "best_val_acc")
    best_final = best_row(final_rows, "test_mean_valid_only", "test_mean")
    contribution_lines = contribution_finding(contrib_rows, final_rows)

    lines = [
        "# NAS + HPO Pipeline Audit Report",
        "",
        "## 1. Executive Summary",
        "",
        "- Current system is joint NAS + HPO, not pure NAS. The search vector combines z_arch with HP dimensions.",
        "- Interpretation must separate architecture gain, HPO gain, and joint gain.",
    ]
    lines.extend(contribution_lines)
    lines.extend(
        [
            "- GMM density must be read as warm-start density over z_search, not predicted accuracy.",
            "- BO-GP validation-accuracy predictions are online surrogate quantities, not final test accuracy.",
            "",
            "Data availability:",
        ]
    )
    lines.extend(inventory)
    if isinstance(manifest, dict):
        lines.append(f"- manifest reports n_search_runs={manifest.get('n_search_runs', 'not available')}")
        lines.append(f"- manifest reports n_final_eval_rows={manifest.get('n_final_eval_rows', 'not available')}")

    lines.extend(["", "## 2. Current Pipeline Overview", ""])
    lines.extend(current_pipeline_table())

    lines.extend(
        [
            "",
            "## 3. Search Space Definition",
            "",
            "- global4: search_dim = 16 = 12 architecture dims + [lr, dropout, hidden_dim, L2].",
            "- hybrid_cond7: search_dim = 19 = 12 architecture dims + global4 + [gat_heads, sage_aggr, gin_eps].",
            "- layer_cond19: search_dim = 31 = 12 architecture dims + global4 + per-layer conditional HP.",
            "- For pure GCN architectures, GAT/SAGE/GIN conditional parameters are theoretically inactive in hybrid_cond7 and layer_cond19.",
            "- If inactive HP are not canonicalized, they may affect GMM density, BO GP kernel distance, TPE parameter distribution, duplicate detection, and result interpretation.",
            "",
            "## 4. GMM vs BO-GP",
            "",
            "- Weighted diagonal GMM models p(z_search | high-performing history).",
            "- Its role is to generate warm-start candidates near historically strong search vectors.",
            "- It must not be described as predicting val_acc for a specific z_search.",
            "- BO-GP models p(val_acc | z_search, observed data).",
            "- Its role is to use posterior mean/std and Expected Improvement to choose the next candidate for real evaluation.",
            "- BO-GP validation predictions must not be equated with final test accuracy.",
            "",
            "## 5. Final Evaluation Interpretation",
            "",
        ]
    )
    if best_search:
        lines.append(
            f"- best search run: {best_search.get('run_id', 'not available')} "
            f"hp_mode={best_search.get('hp_mode', 'not available')} best_val_acc={fmt(best_search.get('best_val_acc'))}"
        )
    else:
        lines.append("- best search run: not available")
    if best_final:
        lines.append(
            f"- best final-eval candidate: {candidate_name(best_final) or 'not available'} "
            f"group={group_name(best_final) or 'not available'} test_mean={fmt(metric(best_final, 'test_mean_valid_only', 'test_mean'))}"
        )
    else:
        lines.append("- best final-eval candidate: not available")
    lines.extend(contribution_lines)
    lines.append("- searched HP effect on manual 2xGCN: see hpo_gain in nas_hpo_contribution_summary.csv when available.")
    lines.append("- independent NAS architecture gain: see nas_arch_gain_default_hp when available.")

    lines.extend(
        [
            "",
            "## 6. Suspected Issues",
            "",
            "1. Current flow is joint NAS + HPO, so gains cannot all be attributed to NAS.",
            "2. If BASE_2xGCN_defaultHP is far below common Cora 2-layer GCN levels, run a sanity check.",
            "3. Inactive HP may change hybrid_cond7 / layer_cond19 trajectories.",
            "4. GMM is not an accuracy predictor and needs density diagnosis.",
            "5. BO-GP is an online surrogate and needs separate prediction-quality diagnosis.",
            "6. final_eval needs valid/invalid seed reporting.",
            "7. HPO gain, NAS gain, and joint gain need to be split before making claims.",
            "",
            "## 7. Recommended Modification Plan",
            "",
            "- P0: standard 2xGCN sanity check; three hp_mode fixed 2xGCN equivalence tests; final_eval valid/invalid seed report.",
            "- P1: NAS/HPO contribution summary.",
            "- P2: inactive HP canonicalization diagnostic-only.",
            "- P3: GMM density diagnosis.",
            "- P4: BO-GP surrogate diagnosis.",
            "- P5: performance-aware latent / MFA, delayed.",
            "",
            "## 8. Safe Next Commands",
            "",
        ]
    )
    for command in safe_commands():
        lines.append(f"- `{command}`")

    if args.include_source_static_summary:
        lines.extend(["", "## Source Static Summary", ""])
        lines.extend(source_static_summary(project_root, warnings))

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in dict.fromkeys(warnings):
            lines.append(f"- {warning}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote audit report: {output}")


def main() -> None:
    args = parse_args()
    write_report(args)


if __name__ == "__main__":
    main()
