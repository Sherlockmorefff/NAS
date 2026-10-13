#!/usr/bin/env python3
"""One-time worker for the actual frozen-validation-checkpoint READY bundle.

Validation and CLI help never import a dataset loader.  The real-data entry
point validates the freeze-stage manifest, full source closure and checkpoint
bytes, then creates the single fixed claim under ``reveal_claim/``.  Only after
that irreversible claim does it import and load the built-in Cora adapter.
There is no CLI-selectable adapter and no configurable claim/result path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence


FORMAT_VERSION = "frozen_validation_checkpoints_global4_v1"
READY_SCHEMA = "frozen-checkpoint-reveal-ready-v1"
READY_ROOT_SCHEME = "canonical-json-excluding-immutable-ready-root-v1"
METHOD_LEVEL_STATE = "VALIDATION_CONFIRMED_HIGH_REGIME_PRIOR"
EXPECTED_SEEDS = tuple(range(100000, 100030))
ROLES = ("P", "Q")
WORKER_RELATIVE = "analyse/reveal_frozen_checkpoints_once.py"
ANALYSIS_RELATIVE = "analyse/analyse_final_checkpoint_reveal.py"
ADAPTER_RELATIVE = "analyse/frozen_cora_benchmark_reveal_adapter.py"
CLAIM_RELATIVE = "reveal_claim/ONE_TIME_REVEAL_CLAIM.json"
RESULT_RELATIVE = "reveal_claim/ONE_TIME_REVEAL_RESULT.json"
FAILURE_RELATIVE = "reveal_claim/ONE_TIME_REVEAL_TERMINAL_FAILURE.json"


class RevealProtocolError(RuntimeError):
    pass


class RevealAlreadyClaimed(RevealProtocolError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def compute_ready_root(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("immutable_ready_root", None)
    return stable_hash(payload)


def seal_ready_document(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(document)
    sealed.setdefault("ready_schema", READY_SCHEMA)
    sealed.setdefault("ready_root_scheme", READY_ROOT_SCHEME)
    sealed["immutable_ready_root"] = compute_ready_root(sealed)
    return sealed


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RevealProtocolError(f"path must be relative: {relative!r}")
    root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RevealProtocolError(f"path escapes ready root: {relative!r}") from exc
    return resolved


def _verified_bytes(path: Path, row: Mapping[str, Any], label: str) -> bytes:
    if not path.is_file():
        raise RevealProtocolError(f"{label} missing: {path}")
    value = path.read_bytes()
    if row.get("size_bytes") != len(value) or row.get("sha256") != sha256_bytes(value):
        raise RevealProtocolError(f"{label} identity mismatch: {path}")
    return value


@dataclass(frozen=True)
class ValidatedReadyBundle:
    root_dir: Path
    ready: Mapping[str, Any]
    ready_manifest: Mapping[str, Any]
    immutable_ready_root: str
    candidates: Mapping[str, Mapping[str, Any]]
    logical_records: tuple[Mapping[str, Any], ...]
    checkpoint_blobs: Mapping[str, bytes]
    adapter_source_bytes: bytes
    candidates_identical: bool


def validate_ready_bundle(ready_path: Path | str, *, expected_ready_root: str) -> ValidatedReadyBundle:
    """Validate actual READY artifacts and retain the exact hashed checkpoint bytes."""
    path = Path(ready_path).resolve()
    if path.name != "TEST_REVEAL_READY.json" or not path.is_file():
        raise RevealProtocolError("ready path must be TEST_REVEAL_READY.json")
    root = path.parent
    ready = _load_json(path)
    if not isinstance(ready, dict) or ready.get("format_version") != FORMAT_VERSION:
        raise RevealProtocolError("ready format mismatch")
    checks = (
        (ready.get("ready_schema") == READY_SCHEMA, "schema"),
        (ready.get("ready_root_scheme") == READY_ROOT_SCHEME, "root scheme"),
        (ready.get("status") == "TEST_REVEAL_READY", "status"),
        (ready.get("classification") == "TEST_REVEAL_READY", "classification"),
        (ready.get("method_level_state") == METHOD_LEVEL_STATE, "method state alias"),
        (ready.get("method_level_status") == METHOD_LEVEL_STATE, "method state"),
        (ready.get("fresh_seeds") == list(EXPECTED_SEEDS), "fresh seeds alias"),
        (ready.get("fresh_paired_seeds") == list(EXPECTED_SEEDS), "paired seeds"),
        (ready.get("all_checkpoint_file_and_tensor_hashes_verified") is True,
         "checkpoint verification flag"),
        (ready.get("real_benchmark_access_count") == 0, "prior benchmark access"),
        (ready.get("reveal_claim_directory_empty") is True, "claim directory state"),
    )
    for passed, label in checks:
        if not passed:
            raise RevealProtocolError(f"ready {label} mismatch")
    ready_root = ready.get("immutable_ready_root")
    if not isinstance(ready_root, str) or ready_root != expected_ready_root:
        raise RevealProtocolError("externally expected ready root mismatch")
    if compute_ready_root(ready) != ready_root:
        raise RevealProtocolError("self-hashed READY document root mismatch")
    expected_paths = {
        "claim": CLAIM_RELATIVE, "result": RESULT_RELATIVE, "failure": FAILURE_RELATIVE,
    }
    if ready.get("fixed_reveal_paths") != expected_paths:
        raise RevealProtocolError("fixed reveal path registry mismatch")

    manifest = _load_json(root / "ready_immutable_manifest.json")
    if not isinstance(manifest, dict):
        raise RevealProtocolError("ready manifest malformed")
    manifest_root = manifest.get("manifest_root_hash")
    if not isinstance(manifest_root, str) or manifest_root != ready.get("ready_manifest_root"):
        raise RevealProtocolError("ready manifest root mismatch")
    files, frozen_sources = manifest.get("files"), manifest.get("frozen_future_sources")
    if not isinstance(files, list) or not isinstance(frozen_sources, list):
        raise RevealProtocolError("ready manifest lists are malformed")
    compact_files = [{
        "relative_path": row.get("relative_path"), "size_bytes": row.get("size_bytes"),
        "sha256": row.get("sha256"),
    } for row in sorted(files, key=lambda item: str(item.get("relative_path")))]
    compact_sources = [{
        "relative_path": row.get("relative_path"), "sha256": row.get("sha256"),
        "size_bytes": row.get("size_bytes"),
    } for row in frozen_sources]
    if stable_hash({"files": compact_files, "frozen_future_sources": compact_sources}) != manifest_root:
        raise RevealProtocolError("ready manifest root recomputation failed")

    file_rows: dict[str, Mapping[str, Any]] = {}
    file_bytes: dict[str, bytes] = {}
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
            raise RevealProtocolError("ready manifest file row malformed")
        relative = str(row["relative_path"])
        if relative in file_rows:
            raise RevealProtocolError(f"duplicate ready file: {relative}")
        file_rows[relative] = row
        file_bytes[relative] = _verified_bytes(_safe_relative(root, relative), row, "ready file")

    dependency_raw = file_bytes.get("source_dependency_manifest.json")
    if dependency_raw is None:
        raise RevealProtocolError("source_dependency_manifest.json not frozen")
    dependency = json.loads(dependency_raw.decode("utf-8"))
    dependency_rows = dependency.get("files") if isinstance(dependency, dict) else None
    if not isinstance(dependency_rows, list):
        raise RevealProtocolError("source dependency manifest malformed")
    dependency_by_relative = {}
    for row in dependency_rows:
        if not isinstance(row, dict) or not isinstance(row.get("relative_path"), str):
            raise RevealProtocolError("source dependency row malformed")
        relative = str(row["relative_path"])
        if relative in dependency_by_relative:
            raise RevealProtocolError(f"duplicate source dependency: {relative}")
        dependency_by_relative[relative] = row
        _verified_bytes(Path(str(row.get("path", ""))).resolve(), row, "source dependency")
    source_by_relative = {
        str(row.get("relative_path")): row for row in frozen_sources if isinstance(row, dict)
    }
    for relative in (WORKER_RELATIVE, ANALYSIS_RELATIVE, ADAPTER_RELATIVE):
        frozen, dependency_row = source_by_relative.get(relative), dependency_by_relative.get(relative)
        if frozen is None or dependency_row is None:
            raise RevealProtocolError(f"future source not frozen: {relative}")
        if (frozen.get("sha256"), frozen.get("size_bytes")) != (
            dependency_row.get("sha256"), dependency_row.get("size_bytes")
        ):
            raise RevealProtocolError(f"future source identity conflict: {relative}")
    frozen_adapter = ready.get("frozen_benchmark_adapter")
    adapter_row = source_by_relative.get(ADAPTER_RELATIVE)
    if frozen_adapter != {
        "relative_path": ADAPTER_RELATIVE,
        "sha256": adapter_row.get("sha256") if adapter_row else None,
        "size_bytes": adapter_row.get("size_bytes") if adapter_row else None,
    }:
        raise RevealProtocolError("frozen benchmark adapter READY identity mismatch")
    if sha256_file(Path(__file__).resolve()) != dependency_by_relative[WORKER_RELATIVE].get("sha256"):
        raise RevealProtocolError("executing worker differs from frozen source")

    inventory_raw = file_bytes.get("checkpoint_inventory.json")
    if inventory_raw is None:
        raise RevealProtocolError("checkpoint inventory not frozen")
    inventory = json.loads(inventory_raw.decode("utf-8"))
    if not isinstance(inventory, list):
        raise RevealProtocolError("checkpoint inventory malformed")
    p_fp, q_fp = ready.get("P_joint_config_fingerprint"), ready.get("Q_joint_config_fingerprint")
    if not isinstance(p_fp, str) or not isinstance(q_fp, str):
        raise RevealProtocolError("candidate fingerprints missing")
    identical = p_fp == q_fp
    ready_candidates = ready.get("candidates")
    if not isinstance(ready_candidates, dict) or {
        role: ready_candidates.get(role, {}).get("joint_fingerprint")
        if isinstance(ready_candidates.get(role), dict) else None
        for role in ROLES
    } != {"P": p_fp, "Q": q_fp}:
        raise RevealProtocolError("READY candidate aliases are inconsistent")
    if ready.get("final_candidates_identical") is not identical:
        raise RevealProtocolError("READY identical-candidate flag is inconsistent")
    expected_physical = 30 if identical else 60
    if (
        ready.get("checkpoint_count") != expected_physical
        or manifest.get("checkpoint_count") != expected_physical
        or len(inventory) != 60
    ):
        raise RevealProtocolError("checkpoint inventory cardinality mismatch")
    ready_records = ready.get("checkpoint_records")
    if not isinstance(ready_records, list) or len(ready_records) != 60:
        raise RevealProtocolError("READY checkpoint_records cardinality mismatch")
    ready_record_by_key = {
        (row.get("method_role"), row.get("seed")): row
        for row in ready_records if isinstance(row, dict)
    }
    if len(ready_record_by_key) != 60:
        raise RevealProtocolError("READY checkpoint_records are duplicated")
    candidates = {}
    for role in ROLES:
        relative = f"frozen_candidate_{role}.json"
        raw_candidate = file_bytes.get(relative)
        if raw_candidate is None:
            raise RevealProtocolError(f"candidate {role} is absent from ready manifest")
        candidate = json.loads(raw_candidate.decode("utf-8"))
        if not isinstance(candidate, dict):
            raise RevealProtocolError(f"candidate {role} document malformed")
        candidates[role] = candidate
    for role, fingerprint in (("P", p_fp), ("Q", q_fp)):
        if candidates[role].get("joint_config_fingerprint") != fingerprint:
            raise RevealProtocolError(f"candidate {role} identity mismatch")

    records, blobs, keys = [], {}, set()
    for row in inventory:
        if not isinstance(row, dict):
            raise RevealProtocolError("checkpoint row malformed")
        role, seed = row.get("role"), row.get("seed")
        if role not in ROLES or not isinstance(seed, int) or (role, seed) in keys:
            raise RevealProtocolError("checkpoint role/seed malformed or duplicated")
        keys.add((role, seed))
        expected_fp = p_fp if role == "P" else q_fp
        if row.get("joint_config_fingerprint") != expected_fp:
            raise RevealProtocolError("checkpoint fingerprint mismatch")
        relative = row.get("checkpoint_relative_path")
        if not isinstance(relative, str) or relative not in file_rows:
            raise RevealProtocolError("checkpoint not bound into ready manifest")
        manifest_row = file_rows[relative]
        if (row.get("checkpoint_file_sha256"), row.get("checkpoint_size_bytes")) != (
            manifest_row.get("sha256"), manifest_row.get("size_bytes")
        ):
            raise RevealProtocolError("checkpoint inventory/manifest conflict")
        ready_record = ready_record_by_key.get((role, seed))
        if ready_record is None or (
            ready_record.get("joint_fingerprint"), ready_record.get("relative_path"),
            ready_record.get("checkpoint_sha256"), ready_record.get("checkpoint_size")
        ) != (
            row.get("joint_config_fingerprint"), relative,
            row.get("checkpoint_file_sha256"), row.get("checkpoint_size_bytes")
        ):
            raise RevealProtocolError("READY checkpoint record/inventory conflict")
        blobs[relative] = file_bytes[relative]
        records.append(dict(row))
    if keys != {(role, seed) for role in ROLES for seed in EXPECTED_SEEDS}:
        raise RevealProtocolError("checkpoint role/seed grid incomplete")
    if len(blobs) != expected_physical:
        raise RevealProtocolError("physical checkpoint count inconsistent")
    if identical:
        by_key = {(row["role"], row["seed"]): row for row in records}
        for seed in EXPECTED_SEEDS:
            p, q = by_key[("P", seed)], by_key[("Q", seed)]
            if (p["checkpoint_relative_path"], p["checkpoint_file_sha256"]) != (
                q["checkpoint_relative_path"], q["checkpoint_file_sha256"]
            ):
                raise RevealProtocolError("identical candidates do not share physical checkpoint")
    adapter_path = Path(str(dependency_by_relative[ADAPTER_RELATIVE]["path"])).resolve()
    adapter_bytes = _verified_bytes(
        adapter_path, dependency_by_relative[ADAPTER_RELATIVE], "frozen benchmark adapter"
    )
    return ValidatedReadyBundle(root, ready, manifest, ready_root, candidates,
                                tuple(records), blobs, adapter_bytes, identical)


def _atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json_bytes(value) + b"\n"
    if exclusive:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError as exc:
            raise RevealAlreadyClaimed(f"one-time reveal already claimed: {path}") from exc
        try:
            os.write(fd, data); os.fsync(fd)
        finally:
            os.close(fd)
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            with temporary.open("xb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists(): temporary.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)


def create_fixed_atomic_claim(bundle: ValidatedReadyBundle) -> Mapping[str, Any]:
    paths = [bundle.root_dir / value for value in (CLAIM_RELATIVE, RESULT_RELATIVE, FAILURE_RELATIVE)]
    claim_directory = (bundle.root_dir / "reveal_claim").resolve()
    existing_claim_artifacts = (
        [path for path in claim_directory.rglob("*") if path.is_file()]
        if claim_directory.exists() else []
    )
    if existing_claim_artifacts or any(path.exists() for path in paths):
        raise RevealAlreadyClaimed("fixed reveal-claim namespace is not empty")
    claim = {
        "claim_schema": "one-time-frozen-checkpoint-reveal-claim-v1",
        "ready_immutable_root": bundle.immutable_ready_root,
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "irreversible": True, "fixed_claim_relative_path": CLAIM_RELATIVE,
    }
    _atomic_json(paths[0], claim, exclusive=True)
    return claim


class _SingleForward:
    def __init__(self, model: Any): self.model, self.calls = model, 0
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.calls: raise RevealProtocolError("model received more than one forward")
        self.calls += 1
        return self.model(*args, **kwargs)


def _member(adapter: Any, name: str) -> Callable[..., Any]:
    value = adapter.get(name) if isinstance(adapter, Mapping) else getattr(adapter, name, None)
    if not callable(value): raise RevealProtocolError(f"adapter missing {name}")
    return value


def _checkpoint_load(torch_module: Any, value: bytes) -> Mapping[str, Any]:
    try: payload = torch_module.load(io.BytesIO(value), weights_only=True, map_location="cpu")
    except TypeError as exc: raise RevealProtocolError("weights_only loader required") from exc
    if not isinstance(payload, Mapping): raise RevealProtocolError("checkpoint payload malformed")
    return payload


def _load_frozen_cora_adapter(bundle: ValidatedReadyBundle, cora_root: Path) -> Mapping[str, Any]:
    """Execute the exact pre-claim-hashed adapter bytes, strictly post-claim."""
    module = types.ModuleType("_frozen_cora_benchmark_reveal_adapter")
    module.__file__ = ADAPTER_RELATIVE
    code = compile(bundle.adapter_source_bytes, ADAPTER_RELATIVE, "exec")
    exec(code, module.__dict__)
    factory = getattr(module, "build_adapter", None)
    if not callable(factory):
        raise RevealProtocolError("frozen adapter has no build_adapter factory")
    return factory(bundle, cora_root)


def _evaluate(bundle: ValidatedReadyBundle, claim: Mapping[str, Any], adapter: Any,
              torch_module: Any) -> Mapping[str, Any]:
    view = _member(adapter, "load_benchmark_view")()
    build_model, score_output = _member(adapter, "build_model"), _member(adapter, "score_output")
    if not isinstance(view, Mapping): raise RevealProtocolError("benchmark view malformed")
    args, kwargs = view.get("model_args", ()), view.get("model_kwargs", {})
    if not isinstance(args, (list, tuple)) or not isinstance(kwargs, Mapping):
        raise RevealProtocolError("model arguments malformed")
    records_by_physical: dict[str, list[Mapping[str, Any]]] = {}
    for row in bundle.logical_records:
        records_by_physical.setdefault(str(row["checkpoint_relative_path"]), []).append(row)
    physical_scores: dict[str, float] = {}
    physical_forward_counts: dict[str, int] = {}
    for relative in sorted(records_by_physical):
        row = sorted(
            records_by_physical[relative],
            key=lambda value: (int(value["seed"]), ROLES.index(str(value["role"]))),
        )[0]
        payload = _checkpoint_load(torch_module, bundle.checkpoint_blobs[relative])
        model = build_model(payload, row); model.eval()
        if getattr(model, "training", False): raise RevealProtocolError("eval mode not active")
        forward = _SingleForward(model)
        with torch_module.no_grad(): output = forward(*args, **dict(kwargs))
        value = score_output(output, view)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RevealProtocolError("accuracy must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise RevealProtocolError("accuracy must be a proportion in [0,1]")
        physical_scores[relative] = value
        physical_forward_counts[relative] = forward.calls

    scores = []
    for row in sorted(
        bundle.logical_records,
        key=lambda value: (int(value["seed"]), ROLES.index(str(value["role"]))),
    ):
        relative = str(row["checkpoint_relative_path"])
        scores.append({"method_role": row["role"], "seed": row["seed"],
                       "joint_fingerprint": row["joint_config_fingerprint"],
                       "checkpoint_sha256": row["checkpoint_file_sha256"],
                       "score": physical_scores[relative],
                       "physical_forward_count": physical_forward_counts[relative]})
    lookup = {(row["method_role"], row["seed"]): row["score"] for row in scores}
    pairs = [{"seed": seed, "P_score": lookup[("P", seed)], "Q_score": lookup[("Q", seed)],
              "delta": lookup[("P", seed)] - lookup[("Q", seed)]} for seed in EXPECTED_SEEDS]
    result: MutableMapping[str, Any] = {
        "result_schema": "one-time-frozen-checkpoint-reveal-result-v1", "status": "REVEAL_COMPLETE",
        "evidence_name": "prospective-after-freeze benchmark reveal",
        "project_level_benchmark_was_untouched": False, "method_level_state": METHOD_LEVEL_STATE,
        "immutable_ready_root": bundle.immutable_ready_root, "claim": dict(claim),
        "final_candidates_identical": bundle.candidates_identical,
        "candidates": {role: {"joint_fingerprint": bundle.ready[f"{role}_joint_config_fingerprint"]}
                       for role in ROLES},
        "checkpoint_scores": scores, "pairs": pairs,
        "physical_checkpoint_count": len(bundle.checkpoint_blobs),
        "logical_checkpoint_count": len(scores),
        "all_physical_forward_counts_one": bool(physical_forward_counts) and all(
            count == 1 for count in physical_forward_counts.values()
        ),
    }
    result["result_sha256"] = stable_hash(result)
    return result


def _execute_after_validation(bundle: ValidatedReadyBundle, adapter_loader: Callable[[], Any],
                              torch_module: Any) -> Mapping[str, Any]:
    claim = create_fixed_atomic_claim(bundle)
    try:
        result = _evaluate(bundle, claim, adapter_loader(), torch_module)
        _atomic_json(bundle.root_dir / RESULT_RELATIVE, result)
        return result
    except BaseException as exc:
        failure = {"failure_schema": "one-time-frozen-checkpoint-reveal-terminal-failure-v1",
                   "ready_immutable_root": bundle.immutable_ready_root, "terminal": True,
                   "retry_permitted": False, "stage": "post_claim_reveal",
                   "error_type": type(exc).__name__,
                   "error_text_or_benchmark_value_persisted": False}
        failure_path = bundle.root_dir / FAILURE_RELATIVE
        if not failure_path.exists(): _atomic_json(failure_path, failure)
        raise


def execute_one_time_reveal(*, ready_path: Path | str, expected_ready_root: str,
                            cora_root: Path | str) -> Mapping[str, Any]:
    bundle = validate_ready_bundle(ready_path, expected_ready_root=expected_ready_root)
    import torch
    return _execute_after_validation(bundle,
        lambda: _load_frozen_cora_adapter(bundle, Path(cora_root)), torch)


def _execute_with_synthetic_adapter_for_tests(*, ready_path: Path | str,
        expected_ready_root: str, adapter_loader: Callable[[], Any], torch_module: Any) -> Mapping[str, Any]:
    bundle = validate_ready_bundle(ready_path, expected_ready_root=expected_ready_root)
    return _execute_after_validation(bundle, adapter_loader, torch_module)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-once", action="store_true")
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--expected-ready-root", required=True)
    parser.add_argument("--cora-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.execute_once or args.authorization_token != "CLAIM_AND_REVEAL_ONCE":
        raise SystemExit("exact explicit one-time authorization is required")
    execute_one_time_reveal(ready_path=args.ready, expected_ready_root=args.expected_ready_root,
                            cora_root=args.cora_root)
    return 0


if __name__ == "__main__":  # pragma: no cover - never run in checkpoint-freeze goal
    raise SystemExit(main())
