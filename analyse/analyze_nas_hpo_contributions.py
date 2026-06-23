"""Analyze whether final NAS gains come from architecture search or HPO."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


LOGICAL_FIELDS = [
    "manual_default",
    "manual_official",
    "manual_searched",
    "nas_default",
    "nas_global_hp",
    "nas_cond_hp",
    "nas_full",
]

OUTPUT_FIELDS = [
    "hp_mode",
    "source_run_id",
    "manual_default_test_mean",
    "manual_official_test_mean",
    "manual_searched_test_mean",
    "nas_default_test_mean",
    "nas_global_hp_test_mean",
    "nas_cond_hp_test_mean",
    "nas_full_test_mean",
    "hpo_gain",
    "nas_arch_gain_default_hp",
    "joint_gain",
    "nas_vs_manual_under_searched_hp",
    "global_hp_gain_on_nas",
    "cond_hp_gain_on_nas",
    "full_minus_global_hp",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize NAS vs HPO final-eval contributions")
    parser.add_argument("--final_eval_summary", type=str, default="results_analysis_bundle/final_eval_summary.csv")
    parser.add_argument("--search_summary", type=str, default="results_analysis_bundle/search_runs_summary.csv")
    parser.add_argument("--output_csv", type=str, default="results_analysis_bundle/nas_hpo_contribution_summary.csv")
    parser.add_argument("--output_md", type=str, default="results_analysis_bundle/nas_hpo_contribution_summary.md")
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


def parse_listish(value: Any) -> list[Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
    return None


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


def get_name(row: dict[str, Any]) -> str:
    return str(row.get("candidate_name") or row.get("name") or "")


def get_group(row: dict[str, Any]) -> str:
    return str(row.get("group") or "")


def source_run_id_from_row(row: dict[str, Any]) -> str:
    source_run_id = str(row.get("source_run_id") or "").strip()
    if source_run_id:
        return source_run_id
    source_best_z = str(row.get("source_best_z") or "").strip()
    if source_best_z:
        return Path(source_best_z).parent.name
    return ""


def classify_candidate(row: dict[str, Any]) -> str | None:
    group = get_group(row)
    name = get_name(row)

    if group == "baseline_official_hp" or "BASE_2xGCN_officialHP" in name:
        return "manual_official"
    if group == "hp_on_manual" or "Manual_2xGCN_searchedHP" in name:
        return "manual_searched"
    if group == "arch_only" or "NAS_arch_defaultHP" in name:
        return "nas_default"
    if group == "arch_global_hp" or "NAS_arch_globalHPOnly" in name:
        return "nas_global_hp"
    if group == "cond_only" or "NAS_arch_condHPOnly" in name:
        return "nas_cond_hp"
    if group == "full" or name == "NAS_full":
        return "nas_full"
    if group == "baseline" or "BASE_2xGCN_defaultHP" in name:
        return "manual_default"
    return None


def row_values(row: dict[str, Any]) -> tuple[list[float], str]:
    seed_value = safe_float(row.get("test_acc"))
    if seed_value is not None:
        return [seed_value], "per_seed_test_acc"

    for key in ("test_mean_valid_only", "test_mean"):
        value = safe_float(row.get(key))
        if value is not None:
            return [value], key

    parsed = parse_listish(row.get("test_list"))
    if parsed is not None:
        values = [safe_float(item) for item in parsed]
        values = [value for value in values if value is not None]
        if values:
            return values, "test_list"

    return [], "missing_test_metric"


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def subtract(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def dedupe_manual_default_rows(rows: list[dict[str, Any]], notes: list[str]) -> list[dict[str, Any]]:
    named_2gcn = [row for row in rows if "BASE_2xGCN_defaultHP" in get_name(row)]
    if named_2gcn and len(named_2gcn) != len(rows):
        ignored = sorted({get_name(row) or "<blank>" for row in rows if row not in named_2gcn})
        notes.append("manual_default_disambiguated_to_BASE_2xGCN_defaultHP")
        notes.append(f"ignored_baseline_group_rows={','.join(ignored)}")
        return named_2gcn
    return rows


def aggregate_final_rows(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[str]]:
    notes: list[str] = []
    grouped_rows: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        logical = classify_candidate(row)
        if logical is None:
            continue
        hp_mode = str(row.get("hp_mode") or "unknown")
        source_run_id = source_run_id_from_row(row)
        grouped_rows[(hp_mode, source_run_id, get_name(row), get_group(row), logical)].append(row)

    candidate_aggs: list[dict[str, Any]] = []
    for (hp_mode, source_run_id, name, group, logical), group_rows in grouped_rows.items():
        values: list[float] = []
        value_sources: set[str] = set()
        for row in group_rows:
            row_metric_values, source = row_values(row)
            values.extend(row_metric_values)
            value_sources.add(source)
        candidate_aggs.append(
            {
                "hp_mode": hp_mode,
                "source_run_id": source_run_id,
                "candidate_name": name,
                "group": group,
                "logical": logical,
                "value": mean(values),
                "n_rows": len(group_rows),
                "n_values": len(values),
                "value_sources": ",".join(sorted(value_sources)),
            }
        )

    logical_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for agg in candidate_aggs:
        logical_groups[(agg["hp_mode"], agg["source_run_id"], agg["logical"])].append(agg)

    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, aggs in logical_groups.items():
        hp_mode, source_run_id, logical = key
        selected = aggs
        if logical == "manual_default":
            selected = dedupe_manual_default_rows(aggs, notes)
        values = [agg["value"] for agg in selected if agg["value"] is not None]
        value = mean(values)
        names = sorted({agg["candidate_name"] or "<blank>" for agg in selected})
        groups = sorted({agg["group"] or "<blank>" for agg in selected})
        agg_notes: list[str] = []
        if len(selected) > 1:
            agg_notes.append("duplicate_candidate_rows_averaged")
            if logical in {"manual_default", "manual_official"}:
                agg_notes.append("duplicate_baseline_averaged")
        out[key] = {
            "hp_mode": hp_mode,
            "source_run_id": source_run_id,
            "logical": logical,
            "value": value,
            "candidate_names": names,
            "groups": groups,
            "notes": ";".join(agg_notes),
        }
    return out, notes


def search_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys = set()
    for row in rows:
        hp_mode = str(row.get("hp_mode") or row.get("source_hp_mode") or "unknown")
        run_id = str(row.get("run_id") or row.get("source_run_id") or "")
        if hp_mode != "unknown" or run_id:
            keys.add((hp_mode, run_id))
    return keys


def lookup_metric(
    aggregates: dict[tuple[str, str, str], dict[str, Any]],
    hp_mode: str,
    source_run_id: str,
    logical: str,
    notes: list[str],
) -> float | None:
    exact = aggregates.get((hp_mode, source_run_id, logical))
    if exact is not None:
        if exact.get("notes"):
            notes.append(f"{logical}:{exact['notes']}")
        return exact.get("value")

    if source_run_id and logical in {"manual_default", "manual_official"}:
        fallback = aggregates.get((hp_mode, "", logical))
        if fallback is not None:
            notes.append(f"{logical}:used_hp_mode_baseline_fallback")
            if fallback.get("notes"):
                notes.append(f"{logical}:{fallback['notes']}")
            return fallback.get("value")
    return None


def build_summary_rows(
    aggregates: dict[tuple[str, str, str], dict[str, Any]],
    search_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = {(hp_mode, source_run_id) for hp_mode, source_run_id, _ in aggregates}
    keys.update(search_keys(search_rows))
    rows: list[dict[str, Any]] = []

    for hp_mode, source_run_id in sorted(keys):
        notes: list[str] = []
        metrics = {
            logical: lookup_metric(aggregates, hp_mode, source_run_id, logical, notes)
            for logical in LOGICAL_FIELDS
        }
        row = {
            "hp_mode": hp_mode,
            "source_run_id": source_run_id,
            "manual_default_test_mean": metrics["manual_default"],
            "manual_official_test_mean": metrics["manual_official"],
            "manual_searched_test_mean": metrics["manual_searched"],
            "nas_default_test_mean": metrics["nas_default"],
            "nas_global_hp_test_mean": metrics["nas_global_hp"],
            "nas_cond_hp_test_mean": metrics["nas_cond_hp"],
            "nas_full_test_mean": metrics["nas_full"],
            "hpo_gain": subtract(metrics["manual_searched"], metrics["manual_default"]),
            "nas_arch_gain_default_hp": subtract(metrics["nas_default"], metrics["manual_default"]),
            "joint_gain": subtract(metrics["nas_full"], metrics["manual_default"]),
            "nas_vs_manual_under_searched_hp": subtract(metrics["nas_full"], metrics["manual_searched"]),
            "global_hp_gain_on_nas": subtract(metrics["nas_global_hp"], metrics["nas_default"]),
            "cond_hp_gain_on_nas": subtract(metrics["nas_cond_hp"], metrics["nas_default"]),
            "full_minus_global_hp": subtract(metrics["nas_full"], metrics["nas_global_hp"]),
            "notes": ";".join(dict.fromkeys(notes)),
        }
        rows.append(row)
    return rows


def csv_value(value: Any) -> Any:
    if value is None:
        return "NaN"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in OUTPUT_FIELDS})


def md_value(value: Any) -> str:
    num = safe_float(value)
    if num is None:
        return "not available"
    return f"{num:.4f}"


def table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def mapping_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No candidate rows detected."]
    seen = []
    for row in rows:
        item = (
            str(row.get("hp_mode") or "unknown"),
            source_run_id_from_row(row),
            get_name(row),
            get_group(row),
            classify_candidate(row) or "unmapped",
        )
        if item not in seen:
            seen.append(item)
    lines = [
        table_row(["hp_mode", "source_run_id", "candidate_name", "group", "logical_candidate"]),
        table_row(["---"] * 5),
    ]
    for item in seen:
        lines.append(table_row(list(item)))
    return lines


def interpretation_lines(summary_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "- If hpo_gain is clearly larger than nas_arch_gain_default_hp, the current improvement is mainly from HPO.",
        "- If nas_vs_manual_under_searched_hp <= 0, do not claim that the NAS architecture beats manual 2xGCN under searched HP.",
        "- search best_val_acc and final test_mean are different metrics and should not be compared as equivalent quantities.",
    ]
    if not summary_rows:
        lines.append("- No final-eval rows are available, so contribution claims are not available.")
        return lines

    for row in summary_rows:
        label = f"{row.get('hp_mode')}/{row.get('source_run_id') or 'unlinked'}"
        hpo_gain = safe_float(row.get("hpo_gain"))
        nas_gain = safe_float(row.get("nas_arch_gain_default_hp"))
        nas_vs_manual = safe_float(row.get("nas_vs_manual_under_searched_hp"))
        if hpo_gain is not None and nas_gain is not None and hpo_gain > nas_gain + 0.005:
            lines.append(f"- {label}: HPO gain is larger than default-HP architecture gain.")
        if nas_vs_manual is not None and nas_vs_manual <= 0:
            lines.append(
                f"- {label}: Current results do not prove searched NAS architecture is better than "
                "manual 2xGCN + searched HP."
            )
    if not any(safe_float(row.get("nas_vs_manual_under_searched_hp")) is not None for row in summary_rows):
        lines.append("- NAS_full versus Manual_2xGCN_searchedHP is not available in the current data.")
    return lines


def write_markdown(
    path: Path,
    final_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    warnings: list[str],
    aggregate_notes: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NAS vs HPO Contribution Summary",
        "",
        "## 1. Candidate mapping",
        "",
    ]
    lines.extend(mapping_lines(final_rows))
    lines.extend(
        [
            "",
            "## 2. Gain definitions",
            "",
            "- hpo_gain = Manual_2xGCN_searchedHP - BASE_2xGCN_defaultHP.",
            "- nas_arch_gain_default_hp = NAS_arch_defaultHP - BASE_2xGCN_defaultHP.",
            "- joint_gain = NAS_full - BASE_2xGCN_defaultHP.",
            "- nas_vs_manual_under_searched_hp = NAS_full - Manual_2xGCN_searchedHP.",
            "- global_hp_gain_on_nas = NAS_arch_globalHPOnly - NAS_arch_defaultHP.",
            "- cond_hp_gain_on_nas = NAS_arch_condHPOnly - NAS_arch_defaultHP.",
            "- full_minus_global_hp = NAS_full - NAS_arch_globalHPOnly.",
            "",
            "## 3. Per-run contribution table",
            "",
        ]
    )
    if summary_rows:
        columns = [
            "hp_mode",
            "source_run_id",
            "hpo_gain",
            "nas_arch_gain_default_hp",
            "joint_gain",
            "nas_vs_manual_under_searched_hp",
            "nas_full_test_mean",
            "manual_searched_test_mean",
            "notes",
        ]
        lines.append(table_row(columns))
        lines.append(table_row(["---"] * len(columns)))
        for row in summary_rows:
            lines.append(
                table_row(
                    [
                        row.get("hp_mode", ""),
                        row.get("source_run_id", ""),
                        md_value(row.get("hpo_gain")),
                        md_value(row.get("nas_arch_gain_default_hp")),
                        md_value(row.get("joint_gain")),
                        md_value(row.get("nas_vs_manual_under_searched_hp")),
                        md_value(row.get("nas_full_test_mean")),
                        md_value(row.get("manual_searched_test_mean")),
                        row.get("notes", ""),
                    ]
                )
            )
    else:
        lines.append("not available")

    lines.extend(["", "## 4. Interpretation", ""])
    lines.extend(interpretation_lines(summary_rows))

    if aggregate_notes:
        lines.extend(["", "## Aggregation notes", ""])
        for note in dict.fromkeys(aggregate_notes):
            lines.append(f"- {note}")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_detected_candidates(rows: list[dict[str, Any]]) -> None:
    print("Detected candidate names and groups:")
    if not rows:
        print("  <none>")
        return
    seen = []
    for row in rows:
        item = (get_name(row) or "<blank>", get_group(row) or "<blank>")
        if item not in seen:
            seen.append(item)
    for name, group in seen:
        print(f"  name={name} group={group}")


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    final_rows = read_csv(Path(args.final_eval_summary), warnings)
    search_rows = read_csv(Path(args.search_summary), warnings)

    print_detected_candidates(final_rows)

    aggregates, aggregate_notes = aggregate_final_rows(final_rows)
    summary_rows = build_summary_rows(aggregates, search_rows)
    write_summary_csv(Path(args.output_csv), summary_rows)
    write_markdown(Path(args.output_md), final_rows, summary_rows, warnings, aggregate_notes)
    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
