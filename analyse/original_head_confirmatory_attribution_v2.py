#!/usr/bin/env python3
"""Freeze the Original Head confirmatory attribution experiment (global4 v2).

This module has no GNN-training or performance-label input path.  It creates
24 fresh candidate pools, frozen Original Head predictions, and the B/M/C
selection batches before any new validation label can be connected.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
import torch
import torch_geometric


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse import original_head_deterministic_batch_pilot as v1


FORMAT_VERSION = "original_head_confirmatory_attribution_global4_v2"
OUTPUT_RELATIVE = "results/posthoc/original-head-confirmatory-attribution-global4-v2"
V1_RELATIVE = "results/posthoc/original-head-deterministic-batch-pilot-global4-v1"
STRICT_RELATIVE = "results/posthoc/performance-aware-vae-strict-regression-global4-v2"
BASE_CHECKPOINT = REPO_ROOT / "checkpoints/joint_model_pipeline_global4_best.pth"
MASTER_SEED = 20260825
POOL_COUNT = 24
ARCHITECTURES_PER_POOL = 64
HP_REGIMES = dict(v1.HP_REGIMES)
INITIAL_DRAWS = 4096
MAX_DRAWS = 65536
DECODE_BATCH = 256
TRAINING_SEEDS = (0, 1, 2)
MAX_SYSTEM_RETRIES = 2
BOOTSTRAP_REPLICATES = 50000
STRATEGIES = (
    "HEAD_GLOBAL_TOP4", "MATCHED_HP_RANDOM_ARCH", "BLIND_JOINT_RANDOM",
)
FINAL_CLASSIFICATIONS = (
    "JOINT_AND_ARCH_CONFIRMED",
    "HP_FILTERING_ONLY",
    "JOINT_CONFIRMED_ATTRIBUTION_INCONCLUSIVE",
    "NO_HEAD_ADVANTAGE",
    "CONFIRMATORY_INCONCLUSIVE",
    "INSUFFICIENT_FRESH_CANDIDATES",
    "EXECUTION_OR_PROTOCOL_FAILURE",
)
EXPECTED_NEW_FILES = (
    "analyse/original_head_confirmatory_attribution_v2.py",
    "analyse/run_original_head_confirmatory_attribution_v2_labels.py",
    "analyse/analyse_original_head_confirmatory_attribution_v2.py",
    "tests/test_original_head_confirmatory_attribution_v2.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derived_seed(*namespace: Any) -> int:
    return int(stable_hash([MASTER_SEED, *namespace])[:8], 16) & 0x7FFFFFFF


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def dump_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row}) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(row.get(key), sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(key), (dict, list, tuple)) else row.get(key)
                    for key in fields
                })
    temporary.replace(path)


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def run_text(args: Sequence[str]) -> str:
    return subprocess.run(
        list(args), cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.rstrip()


def strict_test_key_hits(value: Any, path: str = "root") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if "test" in str(key).lower():
                hits.append(f"{path}.{key}")
            hits.extend(strict_test_key_hits(nested, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            hits.extend(strict_test_key_hits(nested, f"{path}[{index}]"))
    return hits


def _resolve_local_module(module: str) -> list[Path]:
    if not module:
        return []
    base = REPO_ROOT.joinpath(*module.split("."))
    candidates = [base.with_suffix(".py"), base / "__init__.py"]
    return [path.resolve() for path in candidates if path.is_file()]


def discover_local_source_closure(entries: Sequence[Path]) -> list[Path]:
    pending = [path.resolve() for path in entries]
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                if node.module:
                    modules.add(node.module)
                    for alias in node.names:
                        modules.add(f"{node.module}.{alias.name}")
        for module in modules:
            for candidate in _resolve_local_module(module):
                if candidate not in found:
                    pending.append(candidate)
    return sorted(found)


def _head_blob_identity(relative: str) -> tuple[bool, str | None]:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative], cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not tracked:
        return False, None
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return False, None
    return True, hashlib.sha256(result.stdout).hexdigest()


def source_dependency_manifest() -> dict[str, Any]:
    entries = [REPO_ROOT / relative for relative in EXPECTED_NEW_FILES]
    missing = [str(path) for path in entries if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing v2 source files: {missing}")
    closure = discover_local_source_closure(entries)
    rows = []
    for path in closure:
        relative = str(path.relative_to(REPO_ROOT))
        tracked, head_hash = _head_blob_identity(relative)
        current_hash = sha256_file(path)
        rows.append({
            **file_identity(path),
            "relative_path": relative,
            "tracked_by_git": tracked,
            "head_blob_sha256": head_hash,
            "recoverable_from_clean_head": bool(tracked and head_hash == current_hash),
            "requires_delivery_snapshot": not bool(tracked and head_hash == current_hash),
        })
    return {
        "created_at": utc_now(),
        "entry_files": list(EXPECTED_NEW_FILES),
        "dependency_count": len(rows),
        "snapshot_required_count": sum(row["requires_delivery_snapshot"] for row in rows),
        "files": rows,
    }


def environment_lock_text() -> str:
    sections = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_cuda_runtime": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_count": str(torch.cuda.device_count()),
        "nvidia_smi": run_text(["nvidia-smi"]),
        "nvidia_driver": run_text([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
        ]),
        "pip_freeze": run_text([sys.executable, "-m", "pip", "freeze"]),
    }
    return "\n\n".join(f"[{key}]\n{value}" for key, value in sections.items()) + "\n"


def source_identity(output: Path, dependency_manifest: Mapping[str, Any]) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout
    dump_text(output / "tracked_diff.binary.patch", diff.decode("utf-8", errors="surrogateescape"))
    dump_text(output / "environment_lock.txt", environment_lock_text())
    gpu_rows = run_text([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    return {
        "created_at": utc_now(),
        "branch": run_text(["git", "branch", "--show-current"]),
        "head": run_text(["git", "rev-parse", "HEAD"]),
        "status_short": run_text(["git", "status", "--short"]).splitlines(),
        "untracked_files": run_text([
            "git", "ls-files", "--others", "--exclude-standard",
        ]).splitlines(),
        "tracked_binary_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_binary_diff_size_bytes": len(diff),
        "gpu_rows": gpu_rows,
        "dependency_manifest_sha256": stable_hash(dependency_manifest),
        "experiment_identity": (
            "HEAD + tracked binary diff + source dependency manifest/current hashes"
        ),
        "working_tree_cleaned": False,
        "commit_created": False,
        "push_performed": False,
    }


def exposure_sources(output: Path) -> list[Path]:
    sources = set(v1.exposure_sources())
    sources.update({
        REPO_ROOT / V1_RELATIVE / "exposure_registry.json",
        REPO_ROOT / V1_RELATIVE / "frozen_candidate_pools.json",
        REPO_ROOT / V1_RELATIVE / "frozen_strategy_selections.json",
        REPO_ROOT / V1_RELATIVE / "selected_union_configs.json",
    })
    target = output.resolve()
    sources = {
        path.resolve() for path in sources
        if path.is_file() and target not in path.resolve().parents
    }
    return sorted(sources)


def build_exposure_registry(output: Path) -> tuple[set[str], list[dict[str, Any]], dict[str, Any]]:
    sources = exposure_sources(output)
    direct = {path: v1._fingerprints_from_text(path) for path in sources}
    pairs = {path: v1._config_fingerprint_pairs_from_text(path) for path in sources}
    references = {path: v1._reference_config_ids_from_text(path) for path in sources}
    lookup: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path, path_pairs in pairs.items():
        for config_id, fingerprint in path_pairs:
            lookup[config_id].append({
                "architecture_fingerprint": fingerprint,
                "source_path": str(path),
            })
    registry = set().union(*direct.values())
    unresolved = []
    ambiguous = []
    inventory = []
    for path in sources:
        resolved = []
        for config_id in references[path]:
            matches = lookup.get(config_id, [])
            fingerprints = sorted({row["architecture_fingerprint"] for row in matches})
            if not matches:
                unresolved.append({"path": str(path), "config_id": config_id})
                continue
            if len(fingerprints) != 1:
                ambiguous.append({"path": str(path), "config_id": config_id, "fingerprints": fingerprints})
                continue
            fingerprint = fingerprints[0]
            registry.add(fingerprint)
            resolved.append({
                "config_id": config_id,
                "architecture_fingerprint": fingerprint,
                "resolved_by": sorted({row["source_path"] for row in matches}),
            })
        inventory.append({
            **file_identity(path),
            "direct_fingerprint_count": len(direct[path]),
            "reference_config_id_count": len(references[path]),
            "resolved_reference_count": len(resolved),
            "resolved_references": resolved,
        })
    if unresolved or ambiguous:
        raise RuntimeError(f"exposure reference resolution failed: unresolved={unresolved}, ambiguous={ambiguous}")
    v1_candidates = load_json(REPO_ROOT / V1_RELATIVE / "frozen_candidate_pools.json")
    v1_fingerprints = {row["architecture_fingerprint"] for row in v1_candidates}
    missing_v1 = sorted(v1_fingerprints - registry)
    audit = {
        "passed": not missing_v1,
        "source_count": len(sources),
        "union_count": len(registry),
        "reference_resolution_passed": True,
        "unresolved_references": unresolved,
        "ambiguous_references": ambiguous,
        "v1_candidate_architecture_count": len(v1_fingerprints),
        "v1_candidate_fingerprints_covered": len(v1_fingerprints & registry),
        "missing_v1_candidate_fingerprints": missing_v1,
        "label_or_test_values_deserialized": False,
        "fingerprint_only_text_extraction": True,
    }
    if not audit["passed"]:
        raise RuntimeError("full v1 512-architecture pool is not covered by v2 exposure registry")
    return registry, [{"architecture_fingerprint": fp} for fp in sorted(registry)], {
        "audit": audit, "inventory": inventory,
    }


def _decode_pool(model: Any, pool_id: int, excluded: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = derived_seed("candidate_generation", pool_id)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    unique: dict[str, dict[str, Any]] = {}
    invalid = Counter()
    excluded_count = 0
    duplicate_count = 0
    processed = 0
    draw_target = INITIAL_DRAWS
    schedule = []
    while True:
        while processed < draw_target:
            size = min(DECODE_BATCH, draw_target - processed)
            z = torch.randn(size, 12, generator=generator)
            with torch.no_grad():
                graphs = model.arch_vae.decode(z, stochastic=False)
            for offset, graph in enumerate(graphs):
                try:
                    config = v1.config_from_igraph(model, graph)
                    fingerprint = v1.architecture_fingerprint(config)
                except (TypeError, ValueError, KeyError, IndexError) as error:
                    invalid[type(error).__name__] += 1
                    continue
                valid, reason = v1.validate_static_config(config)
                if not valid:
                    invalid[str(reason)] += 1
                    continue
                if fingerprint in excluded:
                    excluded_count += 1
                    continue
                if fingerprint in unique:
                    duplicate_count += 1
                    continue
                unique[fingerprint] = {
                    "pool_id": pool_id,
                    "architecture_fingerprint": fingerprint,
                    "operations": list(config["operations"]),
                    "edges": [list(map(int, edge)) for edge in config["edges"]],
                    "effective_layers": int(config["effective_layers"]),
                    "generation_index": processed + offset,
                    "architecture_source": "frozen_vae_standard_normal_deterministic_greedy",
                }
            processed += size
        schedule.append({"cumulative_draws": processed, "fresh_unique_count": len(unique)})
        if len(unique) >= ARCHITECTURES_PER_POOL or draw_target >= MAX_DRAWS:
            break
        draw_target = min(MAX_DRAWS, draw_target * 2)
    selected = sorted(
        unique.values(), key=lambda row: (row["generation_index"], row["architecture_fingerprint"])
    )[:ARCHITECTURES_PER_POOL]
    legal_count = processed - sum(invalid.values())
    return selected, {
        "pool_id": pool_id,
        "seed": seed,
        "draw_schedule": schedule,
        "draws_used": processed,
        "valid_unique_fresh_before_truncation": len(unique),
        "selected_architecture_count": len(selected),
        "legal_rate": legal_count / processed,
        "fresh_unique_rate": len(unique) / processed,
        "excluded_reject_count": excluded_count,
        "within_pool_duplicate_reject_count": duplicate_count,
        "invalid_counts": dict(sorted(invalid.items())),
        "passed": len(selected) == ARCHITECTURES_PER_POOL,
    }


def joint_config_fingerprint(architecture_fp: str, hp_fp: str) -> str:
    return stable_hash({"architecture_fingerprint": architecture_fp, "hp_fingerprint": hp_fp})


def generate_candidate_pools(exposure: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = v1.load_joint_vae(BASE_CHECKPOINT, "global4", device="cpu")
    model.arch_vae.eval()
    excluded = set(exposure)
    candidates = []
    audits = []
    for pool_id in range(POOL_COUNT):
        architectures, audit = _decode_pool(model, pool_id, excluded)
        audits.append(audit)
        if len(architectures) != ARCHITECTURES_PER_POOL:
            break
        hp_rng = np.random.default_rng(derived_seed("hp_regimes", pool_id))
        for architecture in architectures:
            excluded.add(str(architecture["architecture_fingerprint"]))
            for regime, (low, high) in HP_REGIMES.items():
                raw_hp = hp_rng.uniform(low, high, size=4).astype(np.float64).tolist()
                hp_fp = v1.hp_fingerprint(raw_hp)
                row = {
                    **architecture,
                    "hp_regime": regime,
                    "raw_hp_vector": raw_hp,
                    "hp_fingerprint": hp_fp,
                }
                row.update(v1.realize_global4_hp(row))
                row["joint_config_fingerprint"] = joint_config_fingerprint(
                    row["architecture_fingerprint"], hp_fp
                )
                row["config_id"] = (
                    f"v2_pool{pool_id}_{row['architecture_fingerprint'][:12]}_"
                    f"{regime}_{hp_fp[:8]}"
                )
                candidates.append(row)
    return candidates, audits


ARCHITECTURE_SCHEMA = {
    "pool_id", "architecture_fingerprint", "operations", "edges", "effective_layers",
}
B_SLOT_SCHEMA = {"source_b_slot", "b_architecture_group", "raw_hp_vector"}
CANDIDATE_HP_SCHEMA = {
    "pool_id", "config_id", "architecture_fingerprint", "hp_fingerprint",
    "joint_config_fingerprint", "hp_regime", "raw_hp_vector", "operations", "edges",
    "effective_layers", "lr", "dropout", "hidden_dim", "l2", "gat_heads",
    "sage_aggr", "gin_eps", "gat_heads_by_layer", "sage_aggr_by_layer", "gin_eps_by_layer",
}


def narrow_architecture(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ARCHITECTURE_SCHEMA}


def _validate_exact_schema(rows: Sequence[Mapping[str, Any]], allowed: set[str], name: str) -> None:
    for row in rows:
        extra = set(row) - allowed
        missing = allowed - set(row)
        if extra or missing:
            raise ValueError(f"{name} schema mismatch: extra={extra}, missing={missing}")


def select_replacement_architectures(
    b_slots: Sequence[Mapping[str, Any]], available_architectures: Sequence[Mapping[str, Any]],
    pool_id: int, random_batch: int,
) -> dict[str, dict[str, Any]]:
    _validate_exact_schema(b_slots, B_SLOT_SCHEMA, "matched-HP selector B slot")
    _validate_exact_schema(available_architectures, ARCHITECTURE_SCHEMA, "matched-HP architecture")
    groups = []
    for row in sorted(b_slots, key=lambda item: int(item["source_b_slot"])):
        group = str(row["b_architecture_group"])
        if group not in groups:
            groups.append(group)
    available = sorted(available_architectures, key=lambda row: row["architecture_fingerprint"])
    if len(available) < len(groups):
        raise RuntimeError("insufficient blind replacement architectures")
    rng = np.random.default_rng(derived_seed("replacement_architecture", pool_id, random_batch))
    indices = rng.choice(len(available), size=len(groups), replace=False)
    return {group: dict(available[int(index)]) for group, index in zip(groups, indices)}


def select_blind_hp_configs(
    b_slots: Sequence[Mapping[str, Any]], replacement_map: Mapping[str, Mapping[str, Any]],
    candidate_hp_rows: Sequence[Mapping[str, Any]], pool_id: int, random_batch: int,
) -> list[tuple[int, dict[str, Any]]]:
    _validate_exact_schema(b_slots, B_SLOT_SCHEMA, "blind-HP B slot")
    _validate_exact_schema(candidate_hp_rows, CANDIDATE_HP_SCHEMA, "blind-HP candidate")
    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_hp_rows:
        by_arch[str(row["architecture_fingerprint"])].append(dict(row))
    chosen_by_slot: dict[int, dict[str, Any]] = {}
    slots_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for slot in b_slots:
        slots_by_group[str(slot["b_architecture_group"])].append(slot)
    for group, slots in sorted(slots_by_group.items()):
        architecture_fp = str(replacement_map[group]["architecture_fingerprint"])
        choices = sorted(by_arch[architecture_fp], key=lambda row: (row["hp_regime"], row["config_id"]))
        if len(choices) != 4 or len(slots) > 4:
            raise RuntimeError("blind HP selection requires four frozen regimes and at most four slots")
        rng = np.random.default_rng(derived_seed("blind_hp", pool_id, random_batch, group))
        permutation = rng.permutation(len(choices))
        for slot, index in zip(sorted(slots, key=lambda row: row["source_b_slot"]), permutation):
            chosen_by_slot[int(slot["source_b_slot"])] = dict(choices[int(index)])
    rows = [(index, chosen_by_slot[index]) for index in sorted(chosen_by_slot)]
    order_rng = np.random.default_rng(derived_seed("blind_hp_budget_order", pool_id, random_batch))
    return [rows[int(index)] for index in order_rng.permutation(len(rows))]


def _selection_row(
    config: Mapping[str, Any], strategy: str, budget_order: int,
    random_batch: int | None, **metadata: Any,
) -> dict[str, Any]:
    keys = CANDIDATE_HP_SCHEMA | {"config_id", "joint_config_fingerprint"}
    return {
        **{key: config[key] for key in keys},
        "strategy": strategy,
        "random_batch": random_batch,
        "budget_order": int(budget_order),
        **metadata,
    }


def freeze_selections(
    candidates: Sequence[Mapping[str, Any]], head_means: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_pool: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_pool[int(row["pool_id"])].append(dict(row))
    selections = []
    mappings = []
    pool_audits = []
    for pool_id in range(POOL_COUNT):
        pool = by_pool[pool_id]
        ranked = sorted(pool, key=lambda row: (-float(head_means[row["config_id"]]), row["config_id"]))
        b_configs = ranked[:4]
        group_by_arch: dict[str, str] = {}
        for config in b_configs:
            fp = str(config["architecture_fingerprint"])
            group_by_arch.setdefault(fp, f"g{len(group_by_arch)}")
        b_slots = [
            {
                "source_b_slot": slot,
                "b_architecture_group": group_by_arch[str(config["architecture_fingerprint"])],
                "raw_hp_vector": [float(value) for value in config["raw_hp_vector"]],
            }
            for slot, config in enumerate(b_configs, 1)
        ]
        for slot, config in enumerate(b_configs, 1):
            selections.append(_selection_row(
                config, "HEAD_GLOBAL_TOP4", slot, None,
                source_b_slot=slot,
                b_architecture_group=group_by_arch[str(config["architecture_fingerprint"])],
                head_prediction_mean=float(head_means[config["config_id"]]),
            ))
        b_architectures = set(group_by_arch)
        all_architectures = {
            row["architecture_fingerprint"]: narrow_architecture(row) for row in pool
        }
        remaining = [row for fp, row in all_architectures.items() if fp not in b_architectures]
        used_replacements: set[str] = set()
        batch_maps = {}
        for random_batch in (0, 1):
            available = [row for row in remaining if row["architecture_fingerprint"] not in used_replacements]
            replacement_map = select_replacement_architectures(
                b_slots, available, pool_id, random_batch,
            )
            replacement_fps = {row["architecture_fingerprint"] for row in replacement_map.values()}
            used_replacements.update(replacement_fps)
            batch_maps[random_batch] = replacement_map
            m_by_slot = []
            for b_slot, b_config in zip(b_slots, b_configs):
                replacement = replacement_map[str(b_slot["b_architecture_group"])]
                raw_hp = [float(value) for value in b_slot["raw_hp_vector"]]
                hp_fp = v1.hp_fingerprint(raw_hp)
                m_config = {
                    **replacement,
                    "raw_hp_vector": raw_hp,
                    "hp_fingerprint": hp_fp,
                    "hp_regime": f"matched_from_{b_config['hp_regime']}",
                }
                m_config.update(v1.realize_global4_hp(m_config))
                m_config["joint_config_fingerprint"] = joint_config_fingerprint(
                    m_config["architecture_fingerprint"], hp_fp
                )
                m_config["config_id"] = (
                    f"v2_p{pool_id}_m{random_batch}_s{b_slot['source_b_slot']}_"
                    f"{m_config['architecture_fingerprint'][:12]}_{hp_fp[:8]}"
                )
                m_by_slot.append(m_config)
                selections.append(_selection_row(
                    m_config, "MATCHED_HP_RANDOM_ARCH", int(b_slot["source_b_slot"]), random_batch,
                    source_b_slot=int(b_slot["source_b_slot"]),
                    b_architecture_group=str(b_slot["b_architecture_group"]),
                    source_b_config_id=str(b_config["config_id"]),
                    source_b_hp_regime=str(b_config["hp_regime"]),
                    exact_b_raw_hp_reused=True,
                ))
            hp_candidates = [
                {key: row[key] for key in CANDIDATE_HP_SCHEMA}
                for row in pool if row["architecture_fingerprint"] in replacement_fps
            ]
            c_assignments = select_blind_hp_configs(
                b_slots, replacement_map, hp_candidates, pool_id, random_batch,
            )
            c_slot_assignment = {}
            for order, (source_slot, c_config) in enumerate(c_assignments, 1):
                b_slot = next(
                    slot for slot in b_slots if int(slot["source_b_slot"]) == source_slot
                )
                group = str(b_slot["b_architecture_group"])
                expected_architecture = replacement_map[group]["architecture_fingerprint"]
                if c_config["architecture_fingerprint"] != expected_architecture:
                    raise RuntimeError("blind HP slot lost its frozen replacement architecture")
                c_slot_assignment[source_slot] = c_config
                selections.append(_selection_row(
                    c_config, "BLIND_JOINT_RANDOM", order, random_batch,
                    source_b_slot=source_slot,
                    b_architecture_group=group,
                    blind_hp_selected=True,
                ))
            for group, replacement in sorted(replacement_map.items()):
                b_group_configs = [
                    config for config in b_configs
                    if group_by_arch[config["architecture_fingerprint"]] == group
                ]
                slots = [
                    int(slot["source_b_slot"]) for slot in b_slots
                    if slot["b_architecture_group"] == group
                ]
                mappings.append({
                    "pool_id": pool_id,
                    "random_batch": random_batch,
                    "b_architecture_group": group,
                    "b_architecture_fingerprint": b_group_configs[0]["architecture_fingerprint"],
                    "replacement_architecture_fingerprint": replacement["architecture_fingerprint"],
                    "source_b_slots": slots,
                    "matched_hp_config_ids": [m_by_slot[slot - 1]["config_id"] for slot in slots],
                    "blind_hp_config_ids": [c_slot_assignment[slot]["config_id"] for slot in slots],
                })
        counts = Counter(
            (row["strategy"], row["random_batch"])
            for row in selections if row["pool_id"] == pool_id
        )
        repeat_pattern = sorted(Counter(row["architecture_fingerprint"] for row in b_configs).values(), reverse=True)
        pool_audits.append({
            "pool_id": pool_id,
            "slot_counts": {f"{key[0]}:{key[1]}": value for key, value in counts.items()},
            "b_repeat_architecture_pattern": repeat_pattern,
            "b_unique_architecture_count": len(b_architectures),
            "random_batch_replacement_overlap": len(
                {row["architecture_fingerprint"] for row in batch_maps[0].values()}
                & {row["architecture_fingerprint"] for row in batch_maps[1].values()}
            ),
            "replacement_overlap_with_b": len(
                b_architectures & used_replacements
            ),
            "passed": (
                counts[("HEAD_GLOBAL_TOP4", None)] == 4
                and all(counts[(strategy, batch)] == 4 for strategy in (
                    "MATCHED_HP_RANDOM_ARCH", "BLIND_JOINT_RANDOM",
                ) for batch in (0, 1))
                and not (b_architectures & used_replacements)
                and len({row["architecture_fingerprint"] for row in batch_maps[0].values()}
                        & {row["architecture_fingerprint"] for row in batch_maps[1].values()}) == 0
            ),
        })
    audit = {
        "passed": len(selections) == POOL_COUNT * 20 and all(row["passed"] for row in pool_audits),
        "selection_slot_count": len(selections),
        "mapping_group_count": len(mappings),
        "pool_audits": pool_audits,
        "matched_selector_received_candidate_head": False,
        "matched_selector_received_head_std": False,
        "blind_joint_selector_received_any_head": False,
        "labels_or_trajectory_used": False,
    }
    return selections, mappings, audit


def make_union_tasks(selections: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in selections:
        groups[(int(row["pool_id"]), str(row["joint_config_fingerprint"]))].append(row)
    union = []
    tasks = []
    for (pool_id, joint_fp), rows in sorted(groups.items()):
        representative = dict(sorted(rows, key=lambda row: row["config_id"])[0])
        canonical_id = (
            f"v2_p{pool_id}_{representative['architecture_fingerprint'][:12]}_"
            f"{representative['hp_fingerprint'][:12]}"
        )
        assigned_gpu = derived_seed("gpu_shard", pool_id, joint_fp) % 2
        selected_by = sorted([
            {
                "strategy": row["strategy"],
                "random_batch": row["random_batch"],
                "budget_order": int(row["budget_order"]),
                "selection_config_id": row["config_id"],
            }
            for row in rows
        ], key=lambda row: (row["strategy"], -1 if row["random_batch"] is None else row["random_batch"], row["budget_order"]))
        config = {
            **{key: representative[key] for key in CANDIDATE_HP_SCHEMA},
            "config_id": canonical_id,
            "joint_config_fingerprint": joint_fp,
            "assigned_gpu": assigned_gpu,
            "selected_by": selected_by,
        }
        union.append(config)
        for seed in TRAINING_SEEDS:
            tasks.append({
                **config,
                "seed": seed,
                "task_key": (
                    f"{pool_id}::{canonical_id}::{representative['architecture_fingerprint']}::"
                    f"{representative['hp_fingerprint']}::{seed}"
                ),
            })
    return union, tasks


def protected_paths() -> list[Path]:
    paths = [
        REPO_ROOT / "bo_phase4.py",
        REPO_ROOT / "eval_utils.py",
        BASE_CHECKPOINT,
        REPO_ROOT / STRICT_RELATIVE / "task_plan.json",
        REPO_ROOT / STRICT_RELATIVE / "fixed_split_manifest.json",
    ]
    paths.extend(
        sorted((REPO_ROOT / STRICT_RELATIVE / "checkpoints/full/group_repeated").glob(
            "group_*/seed_*/frozen_head/checkpoint.pth"
        ))
    )
    paths.extend(path for path in (REPO_ROOT / V1_RELATIVE).rglob("*") if path.is_file())
    return sorted(set(path.resolve() for path in paths))


def write_failure(output: Path, classification: str, evidence: Mapping[str, Any]) -> None:
    if classification not in FINAL_CLASSIFICATIONS:
        raise ValueError(classification)
    dump_json(output / "final_classification.json", {
        "format_version": FORMAT_VERSION,
        "classification": classification,
        "created_at": utc_now(),
        "evidence": dict(evidence),
        "label_training_started": False,
        "next_stage_authorized": False,
    })


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    dependency_manifest = source_dependency_manifest()
    dump_json(output / "source_dependency_manifest.json", dependency_manifest)
    dump_json(output / "source_identity.json", source_identity(output, dependency_manifest))
    protected = {str(path): file_identity(path) for path in protected_paths()}
    dump_json(output / "protected_hashes_before.json", protected)

    strict_root = REPO_ROOT / STRICT_RELATIVE
    head_paths = sorted((strict_root / "checkpoints/full/group_repeated").glob(
        "group_*/seed_*/frozen_head/checkpoint.pth"
    ))
    if len(head_paths) != 150:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "Original Head member count", "found": len(head_paths)})
        return
    v1_root = REPO_ROOT / V1_RELATIVE
    new48_protocol = load_json(
        REPO_ROOT / "results/posthoc/h-vs-head-model-blind-prospective-global4-v1/protocol.json"
    )
    input_manifest = {
        "format_version": FORMAT_VERSION,
        "base_vae": file_identity(BASE_CHECKPOINT),
        "head_members": [file_identity(path) for path in head_paths],
        "head_feature_schema": "mu(architecture)[12] + fold-standardized global4 HP[4]",
        "strict_task_plan": file_identity(strict_root / "task_plan.json"),
        "strict_split_manifest": file_identity(strict_root / "fixed_split_manifest.json"),
        "cora_dataset_content_fingerprint": new48_protocol["dataset_content_fingerprint"],
        "cora_public_split_fingerprint": new48_protocol["split_fingerprint"],
        "v1_final_classification": file_identity(v1_root / "final_classification.json"),
        "v1_final_archive": file_identity(v1_root / "original-head-deterministic-batch-pilot-global4-v1-20260824.tar.gz"),
        "v1_final_archive_sidecar": file_identity(v1_root / "original-head-deterministic-batch-pilot-global4-v1-20260824.tar.gz.sha256"),
        "local_source_dependency_manifest": file_identity(output / "source_dependency_manifest.json"),
    }
    dump_json(output / "input_manifest.json", input_manifest)

    exposure, exposure_rows, exposure_metadata = build_exposure_registry(output)
    dump_json(output / "exposure_registry.json", exposure_rows)
    dump_json(output / "exposure_source_inventory.json", exposure_metadata)
    candidates, generation_audits = generate_candidate_pools(exposure)
    dump_json(output / "candidate_generation_audit.json", generation_audits)
    expected_candidates = POOL_COUNT * ARCHITECTURES_PER_POOL * len(HP_REGIMES)
    if len(candidates) != expected_candidates:
        write_failure(output, "INSUFFICIENT_FRESH_CANDIDATES", {
            "expected_candidate_count": expected_candidates,
            "actual_candidate_count": len(candidates),
            "generation_audits": generation_audits,
        })
        return
    candidate_fps = {row["architecture_fingerprint"] for row in candidates}
    by_pool_fps = {
        pool_id: {row["architecture_fingerprint"] for row in candidates if row["pool_id"] == pool_id}
        for pool_id in range(POOL_COUNT)
    }
    cross_overlaps = [
        {"left": left, "right": right, "count": len(by_pool_fps[left] & by_pool_fps[right])}
        for left in range(POOL_COUNT) for right in range(left + 1, POOL_COUNT)
    ]
    candidate_audit = {
        "passed": (
            len(candidate_fps) == POOL_COUNT * ARCHITECTURES_PER_POOL
            and len(candidate_fps & exposure) == 0
            and all(row["count"] == 0 for row in cross_overlaps)
            and not strict_test_key_hits(candidates)
        ),
        "candidate_configuration_count": len(candidates),
        "candidate_architecture_count": len(candidate_fps),
        "configurations_per_pool": dict(Counter(row["pool_id"] for row in candidates)),
        "architectures_per_pool": {pool: len(values) for pool, values in by_pool_fps.items()},
        "exposure_overlap_count": len(candidate_fps & exposure),
        "cross_pool_overlaps": cross_overlaps,
        "replicate_pool_claim": "non-overlapping replicate pools; not asserted IID",
    }
    dump_json(output / "candidate_pool_audit.json", candidate_audit)
    if not candidate_audit["passed"]:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "candidate audit failed", "audit": candidate_audit})
        return
    dump_json(output / "frozen_candidate_pools.json", candidates)
    dump_csv(output / "frozen_candidate_pools.csv", candidates)

    matrix, members, checkpoints = v1.head_member_predictions(candidates, strict_root)
    if matrix.shape != (150, len(candidates)):
        raise RuntimeError(f"unexpected Head member matrix shape: {matrix.shape}")
    prediction_rows = []
    for index, candidate in enumerate(candidates):
        values = matrix[:, index].astype(float).tolist()
        prediction_rows.append({
            "pool_id": candidate["pool_id"],
            "config_id": candidate["config_id"],
            "architecture_fingerprint": candidate["architecture_fingerprint"],
            "hp_fingerprint": candidate["hp_fingerprint"],
            "joint_config_fingerprint": candidate["joint_config_fingerprint"],
            "head_prediction_mean": float(np.mean(values)),
            "head_prediction_std_descriptive_only": float(np.std(values, ddof=1)),
            "head_member_predictions": values,
            "member_count": len(values),
        })
    dump_json(output / "frozen_head_predictions.json", prediction_rows)
    dump_csv(output / "frozen_head_predictions.csv", prediction_rows)
    dump_json(output / "frozen_head_member_identity.json", {"members": members, "checkpoints": checkpoints})
    head_means = {row["config_id"]: row["head_prediction_mean"] for row in prediction_rows}

    selections, mappings, selection_audit = freeze_selections(candidates, head_means)
    dump_json(output / "frozen_strategy_selections.json", selections)
    dump_csv(output / "frozen_strategy_selections.csv", selections)
    dump_json(output / "frozen_replacement_architecture_mappings.json", mappings)
    dump_csv(output / "frozen_replacement_architecture_mappings.csv", mappings)
    dump_json(output / "strategy_selection_audit.json", selection_audit)
    if not selection_audit["passed"] or strict_test_key_hits(selections):
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "selection audit failed", "audit": selection_audit})
        return
    union, tasks = make_union_tasks(selections)
    if len(union) > POOL_COUNT * 20 or len(tasks) > POOL_COUNT * 20 * len(TRAINING_SEEDS):
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {
            "reason": "selection union exceeds preregistered maximum",
            "union": len(union), "tasks": len(tasks),
        })
        return
    dump_json(output / "selected_union_configs.json", union)
    dump_csv(output / "selected_union_configs.csv", union)
    dump_json(output / "selected_union_tasks.json", tasks)
    dump_csv(output / "selected_union_tasks.csv", tasks)

    # Freeze descriptive Head predictions for the actually labelled union as well.
    # These are produced only after every strategy selection is fixed and are never
    # returned to a selector.
    union_matrix, union_members, union_checkpoints = v1.head_member_predictions(union, strict_root)
    if union_matrix.shape != (150, len(union)):
        raise RuntimeError(f"unexpected selected-union Head matrix shape: {union_matrix.shape}")
    if union_members != members or union_checkpoints != checkpoints:
        raise RuntimeError("Original Head member identity changed between prediction passes")
    union_prediction_rows = []
    for index, config in enumerate(union):
        values = union_matrix[:, index].astype(float).tolist()
        union_prediction_rows.append({
            "pool_id": config["pool_id"],
            "config_id": config["config_id"],
            "architecture_fingerprint": config["architecture_fingerprint"],
            "hp_fingerprint": config["hp_fingerprint"],
            "joint_config_fingerprint": config["joint_config_fingerprint"],
            "head_prediction_mean": float(np.mean(values)),
            "head_prediction_std_descriptive_only": float(np.std(values, ddof=1)),
            "head_member_predictions": values,
            "member_count": len(values),
            "generated_after_strategy_freeze": True,
            "used_by_any_selector": False,
        })
    dump_json(output / "frozen_selected_union_head_predictions.json", union_prediction_rows)
    dump_csv(output / "frozen_selected_union_head_predictions.csv", union_prediction_rows)

    gpu_lines = run_text([
        "nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits",
    ]).splitlines()
    gpu_map = {line.split(",")[0].strip(): line.split(",")[1].strip() for line in gpu_lines}
    power = {
        "v1_B_minus_C_mean": 0.021333333333333343,
        "v1_pool_sd": 0.04011,
        "pool_count": POOL_COUNT,
        "practical_threshold": 0.006,
        "statement": (
            "24 pools were fixed from the v1 observed effect/SD. This design is not claimed "
            "to provide 80% power for a true effect as small as 0.006; two random batches "
            "and three seeds reduce within-pool uncertainty but do not increase the pool n."
        ),
    }
    preregistration = {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "scientific_name": "Original Head confirmatory attribution, global4 v2",
        "method_name": "predictor-guided deterministic batch NAS+HPO",
        "method_is_bo": False,
        "master_seed": MASTER_SEED,
        "pool_count": POOL_COUNT,
        "architectures_per_pool": ARCHITECTURES_PER_POOL,
        "hp_regimes": HP_REGIMES,
        "random_batches_per_pool": 2,
        "strategy_budget": 4,
        "candidate_draw_rule": {"initial": INITIAL_DRAWS, "doubling": True, "maximum": MAX_DRAWS},
        "power_design": power,
        "strategies": {
            "B": "HEAD_GLOBAL_TOP4",
            "M0_M1": "same blind random architecture mappings as C; exact B raw HP slots and repeat pattern",
            "C0_C1": "same random architecture mappings as M; blind frozen regime HP selection",
            "random_batch_aggregation": "arithmetic mean of batch best-of-4; never take the better batch",
        },
        "training": {
            "dataset": "Cora", "split": "public", "hp_mode": "global4",
            "dataset_content_fingerprint": new48_protocol["dataset_content_fingerprint"],
            "split_fingerprint": new48_protocol["split_fingerprint"],
            "max_epochs": 300, "patience": 80, "seeds": list(TRAINING_SEEDS),
            "track_test": False, "return_trajectory": True,
            "trajectory_sanitization": (
                "assert every test-named raw value is None; save only epoch_1based/loss/validation_accuracy; "
                "recompute sanitized_trajectory_hash"
            ),
            "maximum_system_retries": MAX_SYSTEM_RETRIES,
            "shard_to_gpu_uuid": gpu_map,
        },
        "statistics": {
            "primary_unit": "pool",
            "configuration_label": "arithmetic mean over fixed seed IDs 0,1,2",
            "B": "best-of-4 HEAD_GLOBAL_TOP4",
            "M": "mean(best-of-4 M0, best-of-4 M1)",
            "C": "mean(best-of-4 C0, best-of-4 C1)",
            "comparisons": ["B_minus_C", "B_minus_M", "M_minus_C"],
            "hierarchical_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "hierarchical_bootstrap_seed": derived_seed("hierarchical_bootstrap"),
            "pool_only_bootstrap_seed": derived_seed("pool_only_bootstrap"),
            "hierarchy": (
                "shared outer pool indices; matched random-batch indices shared by M/C; "
                "one seed-ID resample vector shared by every configuration in each pool draw; "
                "recompute configuration means and best-of-4 after seed resampling"
            ),
            "joint_gate": ["mean B-C >= 0.006", "hierarchical 95% CI lower > 0", "B>C in at least 15/24 pools"],
            "component_gates_after_joint": {
                "architecture_B_minus_M": ["mean >= 0.006", "Bonferroni 97.5% two-sided CI lower > 0", "positive in at least 15/24 pools"],
                "HP_M_minus_C": ["mean >= 0.006", "Bonferroni 97.5% two-sided CI lower > 0", "positive in at least 15/24 pools"],
            },
            "component_family": "B-M and M-C only; 97.5% two-sided intervals implement two-comparison Bonferroni control",
            "pool_only_sensitivity": True,
            "v1_v2_combined_is_secondary_only": True,
        },
        "classifications": FINAL_CLASSIFICATIONS,
        "forbidden": [
            "any model/scaler training", "GP/BO/acquisition/DPP", "Head std selection",
            "latent Euclidean distance", "test metric", "post-label plan change", "mutation/evolution/search",
        ],
        "source_code_hashes": {
            str((REPO_ROOT / relative).resolve()): sha256_file(REPO_ROOT / relative)
            for relative in EXPECTED_NEW_FILES
        },
        "source_dependency_manifest_sha256": sha256_file(output / "source_dependency_manifest.json"),
    }
    dump_json(output / "preregistration.json", preregistration)

    firewall = {
        "passed": (
            candidate_audit["passed"] and selection_audit["passed"]
            and not strict_test_key_hits(candidates)
            and not strict_test_key_hits(selections)
            and not strict_test_key_hits(tasks)
            and len(tasks) == len(union) * len(TRAINING_SEEDS)
            and len({row["task_key"] for row in tasks}) == len(tasks)
        ),
        "candidate_exposure_overlap_count": candidate_audit["exposure_overlap_count"],
        "cross_pool_overlap_count": sum(row["count"] for row in cross_overlaps),
        "v1_full_512_pool_covered_by_exposure": exposure_metadata["audit"]["v1_candidate_fingerprints_covered"] == 512,
        "slot_count_by_pool_strategy_batch": {
            f"{pool}:{strategy}:{batch}": count
            for (pool, strategy, batch), count in sorted(
                Counter((row["pool_id"], row["strategy"], row["random_batch"]) for row in selections).items(),
                key=lambda item: str(item[0]),
            )
        },
        "matched_selector_candidate_head_fields_received": False,
        "matched_selector_head_std_received": False,
        "matched_selector_only_uses_frozen_B_HP_and_repeat_pattern": True,
        "blind_joint_selector_any_head_received": False,
        "selection_validation_or_test_fields": strict_test_key_hits(selections),
        "task_test_fields": strict_test_key_hits(tasks),
        "mu_euclidean_distance_used": False,
        "H_or_refit_used": False,
        "union_configuration_count": len(union),
        "task_count": len(tasks),
        "expected_task_count": len(union) * 3,
        "task_keys_unique": len({row["task_key"] for row in tasks}) == len(tasks),
        "three_seeds_per_configuration": all(
            {row["seed"] for row in tasks if row["config_id"] == config["config_id"]} == set(TRAINING_SEEDS)
            for config in union
        ),
        "track_test": False,
        "labels_connected": False,
    }
    dump_json(output / "pre_label_firewall_audit.json", firewall)
    if not firewall["passed"]:
        write_failure(output, "EXECUTION_OR_PROTOCOL_FAILURE", {"reason": "pre-label firewall failed", "firewall": firewall})
        return

    frozen_names = [
        "source_dependency_manifest.json", "source_identity.json", "environment_lock.txt",
        "tracked_diff.binary.patch", "input_manifest.json", "protected_hashes_before.json",
        "exposure_registry.json", "exposure_source_inventory.json", "candidate_generation_audit.json",
        "candidate_pool_audit.json", "frozen_candidate_pools.json", "frozen_candidate_pools.csv",
        "frozen_head_predictions.json", "frozen_head_predictions.csv", "frozen_head_member_identity.json",
        "frozen_strategy_selections.json", "frozen_strategy_selections.csv",
        "frozen_replacement_architecture_mappings.json", "frozen_replacement_architecture_mappings.csv",
        "strategy_selection_audit.json", "selected_union_configs.json", "selected_union_configs.csv",
        "selected_union_tasks.json", "selected_union_tasks.csv", "preregistration.json",
        "frozen_selected_union_head_predictions.json",
        "frozen_selected_union_head_predictions.csv", "pre_label_firewall_audit.json",
    ]
    entries = [{**file_identity(output / name), "relative_path": name} for name in frozen_names]
    root_hash = stable_hash([(row["relative_path"], row["sha256"]) for row in entries])
    dump_json(output / "pre_label_immutable_manifest.json", {
        "format_version": FORMAT_VERSION,
        "created_at": utc_now(),
        "files": entries,
        "manifest_root_hash": root_hash,
        "immutable_after_new_label_connection": True,
    })
    print(json.dumps({
        "status": "PRE_LABEL_FREEZE_COMPLETE",
        "candidate_architectures": len(candidate_fps),
        "candidate_configurations": len(candidates),
        "selection_slots": len(selections),
        "union_configurations": len(union),
        "tasks": len(tasks),
        "manifest_root_hash": root_hash,
    }, sort_keys=True))


def verify_freeze(output: Path) -> dict[str, Any]:
    manifest = load_json(output / "pre_label_immutable_manifest.json")
    mismatches = []
    for row in manifest["files"]:
        path = output / row["relative_path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"]:
            mismatches.append({"relative_path": row["relative_path"], "expected": row["sha256"], "actual": actual})
    root_hash = stable_hash([(row["relative_path"], row["sha256"]) for row in manifest["files"]])
    dependency = load_json(output / "source_dependency_manifest.json")
    dependency_mismatches = []
    for row in dependency["files"]:
        path = Path(row["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"]:
            dependency_mismatches.append({"path": row["path"], "expected": row["sha256"], "actual": actual})
    return {
        "passed": (
            not mismatches and not dependency_mismatches
            and root_hash == manifest["manifest_root_hash"]
        ),
        "mismatches": mismatches,
        "dependency_mismatches": dependency_mismatches,
        "stored_root_hash": manifest["manifest_root_hash"],
        "recomputed_root_hash": root_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify-freeze"))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve())
    else:
        audit = verify_freeze(args.output.resolve())
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not audit["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
