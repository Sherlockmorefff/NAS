"""Diagnostic-only canonicalization helpers for z_search records.

This module is intentionally not wired into BO, TPE, GMM, or final evaluation.
It is for offline diagnostics of inactive HP dimensions in joint NAS + HPO
search vectors.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_ARCH_NZ = 12
DEFAULT_INACTIVE_VALUE = 0.5
ZERO_INACTIVE_WARNING = (
    "inactive_value=0.0 is allowed, but it is a legal boundary value and may "
    "affect distance or density diagnostics."
)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _listish(value: Any) -> list[Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "detach") and callable(value.detach):
        try:
            return value.detach().cpu().flatten().tolist()
        except Exception:
            return None
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            out = value.tolist()
            return out if isinstance(out, list) else [out]
        except Exception:
            return None
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(value)
            except Exception:
                continue
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
    return None


def _numeric_list(value: Any) -> list[float] | None:
    items = _listish(value)
    if items is None:
        return None
    out: list[float] = []
    for item in items:
        num = _safe_float(item)
        if num is None:
            return None
        out.append(num)
    return out


def _warn_zero_inactive_value(inactive_value: float) -> list[str]:
    if float(inactive_value) == 0.0:
        warnings.warn(ZERO_INACTIVE_WARNING, RuntimeWarning, stacklevel=2)
        return [ZERO_INACTIVE_WARNING]
    return []


def canonicalize_z_search(
    z_search,
    condition_mask_vector,
    arch_nz: int = DEFAULT_ARCH_NZ,
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
):
    """Return a canonicalized z_search list without modifying the input.

    z_arch dimensions are preserved. Active HP dimensions are preserved.
    Inactive HP dimensions are set to inactive_value.
    """

    _warn_zero_inactive_value(inactive_value)
    z_values = _numeric_list(z_search)
    mask_values = _numeric_list(condition_mask_vector)
    if z_values is None:
        raise ValueError("z_search is missing or is not a numeric vector")
    if mask_values is None:
        raise ValueError("condition_mask_vector is missing or is not a numeric vector")

    arch_nz = int(arch_nz)
    expected_len = arch_nz + len(mask_values)
    if len(z_values) != expected_len:
        raise ValueError(
            f"z_search length mismatch: expected {expected_len} from arch_nz={arch_nz} "
            f"and mask length={len(mask_values)}, got {len(z_values)}"
        )

    out = list(z_values)
    for hp_idx, active in enumerate(mask_values):
        if float(active) == 0.0:
            out[arch_nz + hp_idx] = float(inactive_value)
    return out


def _vector_from_condition_mask(condition_mask: Any) -> list[float] | None:
    if not isinstance(condition_mask, dict):
        return None
    for key in ("vector", "mask"):
        values = _numeric_list(condition_mask.get(key))
        if values is not None:
            return values
    return None


def _ops_from_record(record: dict[str, Any]) -> list[str] | None:
    ops = _listish(record.get("operations"))
    if ops is None:
        return None
    return [str(op) for op in ops]


def _call_mask_function(func, ops: list[str], hp_mode: str):
    try:
        return func(ops, hp_mode)
    except TypeError:
        return func(ops)


def _extract_condition_mask_vector_with_warnings(
    record: dict[str, Any],
    hp_mode: str | None = None,
    arch_nz: int = DEFAULT_ARCH_NZ,
) -> tuple[list[float] | None, list[str]]:
    del arch_nz
    warnings_out: list[str] = []

    direct = _numeric_list(record.get("condition_mask_vector"))
    if direct is not None:
        return direct, warnings_out

    from_mask = _vector_from_condition_mask(record.get("condition_mask"))
    if from_mask is not None:
        return from_mask, warnings_out

    resolved_hp_mode = hp_mode or record.get("hp_mode")
    if not resolved_hp_mode and isinstance(record.get("condition_mask"), dict):
        resolved_hp_mode = record["condition_mask"].get("hp_mode")
    ops = _ops_from_record(record)
    if not ops or not resolved_hp_mode:
        warnings_out.append(
            "cannot infer condition mask because operations or hp_mode is missing"
        )
        return None, warnings_out

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import hp_modes  # type: ignore
    except ImportError as exc:
        warnings_out.append(f"cannot infer condition mask because hp_modes import failed: {exc}")
        return None, warnings_out

    for func_name in ("condition_mask_vector_from_ops", "condition_mask_dict_from_ops"):
        func = getattr(hp_modes, func_name, None)
        if not callable(func):
            continue
        try:
            result = _call_mask_function(func, ops, str(resolved_hp_mode))
        except Exception as exc:
            warnings_out.append(f"{func_name} failed while inferring condition mask: {exc}")
            continue
        if isinstance(result, dict):
            vector = _vector_from_condition_mask(result)
        else:
            vector = _numeric_list(result)
        if vector is not None:
            return vector, warnings_out

    warnings_out.append(
        "cannot infer condition mask because no compatible hp_modes function was found"
    )
    return None, warnings_out


def extract_condition_mask_vector(
    record: dict[str, Any],
    hp_mode: str | None = None,
    arch_nz: int = DEFAULT_ARCH_NZ,
):
    """Extract or infer a condition mask vector for a history/final-eval record."""

    vector, warning_messages = _extract_condition_mask_vector_with_warnings(
        record,
        hp_mode=hp_mode,
        arch_nz=arch_nz,
    )
    for message in warning_messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return vector


def canonicalize_history_record(
    record: dict[str, Any],
    arch_nz: int = DEFAULT_ARCH_NZ,
    inactive_value: float = DEFAULT_INACTIVE_VALUE,
) -> dict[str, Any]:
    """Return a new record with raw and canonical z_search diagnostics added."""

    out = deepcopy(record)
    warning_messages = _warn_zero_inactive_value(inactive_value)
    z_raw = _numeric_list(record.get("z_search"))
    out["z_search_raw"] = None if z_raw is None else list(z_raw)
    out["z_search_canonical"] = None
    out["canonicalization_applied"] = False
    out["inactive_hp_count"] = 0
    out["active_hp_count"] = 0

    if z_raw is None:
        warning_messages.append("z_search missing or not numeric; canonicalization skipped")
        out["canonicalization_warning"] = "; ".join(warning_messages)
        return out

    mask_vector, mask_warnings = _extract_condition_mask_vector_with_warnings(
        record,
        hp_mode=record.get("hp_mode"),
        arch_nz=arch_nz,
    )
    warning_messages.extend(mask_warnings)
    if mask_vector is None:
        out["canonicalization_warning"] = "; ".join(warning_messages)
        return out

    out["active_hp_count"] = sum(1 for value in mask_vector if float(value) != 0.0)
    out["inactive_hp_count"] = sum(1 for value in mask_vector if float(value) == 0.0)

    expected_len = int(arch_nz) + len(mask_vector)
    if len(z_raw) != expected_len:
        warning_messages.append(
            f"z_search length mismatch: expected {expected_len}, got {len(z_raw)}"
        )
        out["canonicalization_warning"] = "; ".join(warning_messages)
        return out

    try:
        out["z_search_canonical"] = canonicalize_z_search(
            z_raw,
            mask_vector,
            arch_nz=arch_nz,
            inactive_value=inactive_value,
        )
        out["canonicalization_applied"] = True
    except Exception as exc:
        warning_messages.append(f"canonicalization failed: {exc}")
    out["canonicalization_warning"] = "; ".join(warning_messages)
    return out


def _history_records(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        for key in ("history", "records", "trials", "results"):
            value = obj.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if isinstance(obj.get("z_search"), (list, tuple, str)):
            return [obj]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize inactive HP dimensions for diagnostics")
    parser.add_argument("--input_json", type=str, default="")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--arch_nz", type=int, default=DEFAULT_ARCH_NZ)
    parser.add_argument("--inactive_value", type=float, default=DEFAULT_INACTIVE_VALUE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_json:
        print("No --input_json provided; import this module or pass a history JSON to canonicalize.")
        return

    input_path = Path(args.input_json)
    output_path = Path(args.output_json) if args.output_json else input_path.with_name(
        input_path.stem + "_canonicalized.json"
    )
    try:
        obj = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"status": "skipped", "warning": f"failed to read input JSON: {exc}"}, indent=2),
            encoding="utf-8",
        )
        return

    records = _history_records(obj)
    payload = {
        "status": "ok",
        "input_json": input_path.as_posix(),
        "arch_nz": int(args.arch_nz),
        "inactive_value": float(args.inactive_value),
        "n_records": len(records),
        "records": [
            canonicalize_history_record(
                record,
                arch_nz=args.arch_nz,
                inactive_value=args.inactive_value,
            )
            for record in records
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote canonicalized diagnostics: {output_path}")


if __name__ == "__main__":
    main()
