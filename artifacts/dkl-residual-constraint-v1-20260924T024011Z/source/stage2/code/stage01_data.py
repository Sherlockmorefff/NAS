"""Release exactly the precommitted 20 A labels; never decode hidden label rows.

The trainer consumes only selected_data.json. This separate adapter owns the
read permission for the historical A-only file, which is never copied to the task.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


EXPECTED_PUBLIC_HASHES = {
    "features.json": "4aedd1482cdc36c055c4016a25fe4a8158c305321673e16e868b994ab830c797",
    "candidate_identity.json": "c66c2aa70c4cf2d5fe7d6462bd822a35c4fdd64e950b326cfce8fcf013cab46e",
    "recipe_verified.json": "889879271d809e09b4acb5defa2be6d49736ce62f4b2b4efa8b035bd48761d5c",
    "data_checks.json": "b4ba8a976468baba3b2092c25a9b31680a217948e3d190b0f08f6a4d0ecd7429",
}
RECIPE_HASH = "40bb99cbc3e33dcc417869a97d969b8e8e1c4ca7c6a5fad66c29ed85edea5978"
A_ID_HASH = "d48d42b0823d2d114cd43fe015f569e58eb1f840dbe7ff542c5871de6268e1e2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_protocol(path, expected_hash):
    if file_hash(path) != expected_hash:
        raise ValueError("Frozen protocol file SHA-256 mismatch")
    protocol = read_json(path)
    selection = protocol["selection"]
    if selection["count"] != 20 or selection["seed"] != 20260922:
        raise ValueError("Selection must use the frozen 20-ID protocol")
    if protocol["dtype"] != "float64" or protocol["next_stage_authorized"] is not False:
        raise ValueError("Unsupported protocol")
    return protocol


def public_sources(source_dir):
    """Open only four pinned metadata/feature files and the source protocol."""
    source_dir = Path(source_dir)
    for name, expected in EXPECTED_PUBLIC_HASHES.items():
        if file_hash(source_dir / name) != expected:
            raise ValueError("Public input fingerprint changed: " + name)
    features = read_json(source_dir / "features.json")
    candidates = read_json(source_dir / "candidate_identity.json")
    recipe = read_json(source_dir / "recipe_verified.json")
    checks = read_json(source_dir / "data_checks.json")
    source_protocol = read_json(source_dir / "protocol.json")
    if digest({k: v for k, v in source_protocol.items() if k != "identity"}) != source_protocol["identity"]:
        raise ValueError("Historical protocol identity mismatch")
    if any(source_protocol["inputs"][k] != v for k, v in EXPECTED_PUBLIC_HASHES.items()):
        raise ValueError("Historical public-input provenance mismatch")
    if set(features) != {"ids", "architectures", "z", "hp"}:
        raise ValueError("Features may contain unauthorized fields")
    ids = features["ids"]
    if len(ids) != 179 or len(set(ids)) != 179 or digest(sorted(ids)) != A_ID_HASH:
        raise ValueError("Legal A candidate universe mismatch")
    if len(set(features["architectures"])) != 96:
        raise ValueError("A architecture count changed")
    if [r["candidate_id"] for r in candidates] != ids:
        raise ValueError("Candidate feature order mismatch")
    if digest(recipe["training_protocol_identity"]) != RECIPE_HASH:
        raise ValueError("Recipe fingerprint mismatch")
    metadata = {}
    for i, row in enumerate(candidates):
        if row["training_protocol_identity"] != RECIPE_HASH or row["normalization"] != "none" or row["hp_mode"] != "global4":
            raise ValueError("Mixed candidate recipe")
        if row["architecture_fingerprint"] != features["architectures"][i]:
            raise ValueError("Architecture feature mismatch")
        raw = row["raw_hp"]
        transformed = [math.log10(raw["lr"]), raw["dropout"], math.log2(raw["hidden_dim"]), math.log10(raw["l2"])]
        if transformed != features["hp"][i] or len(features["z"][i]) != 12:
            raise ValueError("Frozen HP transform or VAE dimension mismatch")
        metadata[row["candidate_id"]] = {k: row[k] for k in ["candidate_id", "architecture_fingerprint", "raw_hp", "training_protocol_identity", "hp_mode", "normalization"]}
    if np.asarray(features["z"]).shape != (179, 12) or np.asarray(features["hp"]).shape != (179, 4):
        raise ValueError("Unexpected input shape")
    if not np.isfinite(features["z"]).all() or not np.isfinite(features["hp"]).all():
        raise ValueError("Nonfinite public inputs")
    return features, metadata, recipe, checks, source_protocol


def select(source_dir, output_dir, protocol_path, protocol_hash):
    checked_protocol(protocol_path, protocol_hash)
    features, metadata, recipe, checks, source_protocol = public_sources(source_dir)
    ids = np.random.default_rng(20260922).choice(sorted(features["ids"]), 20, replace=False).tolist()
    manifest = {
        "created_at_utc": now(), "protocol_sha256": protocol_hash,
        "selection_seed": 20260922, "ids": ids,
        "selection_algorithm": "numpy.random.default_rng(20260922).choice(sorted_ids,20,replace=False)",
        "selected_order_preserved": True, "selected_ids_sha256": digest(ids),
        "legal_pool_size": 179, "legal_pool_architectures": 96, "legal_A_ids_sha256": A_ID_HASH,
        "source_dir": str(Path(source_dir).resolve()), "source_public_sha256": EXPECTED_PUBLIC_HASHES,
        "source_protocol_identity": source_protocol["identity"],
        "recipe": recipe["training_protocol_identity"], "recipe_sha256": RECIPE_HASH,
        "recipe_preregistration": recipe["actual_preregistration_training"],
        "vae_provenance": checks["prepared_manifest"], "candidate_metadata": [metadata[c] for c in ids],
        "label_values_decoded_before_manifest": 0,
        "hidden_SE_values_decoded_before_manifest": 0,
    }
    manifest["identity"] = digest(manifest)
    write_new(Path(output_dir) / "selection_manifest.json", manifest)
    return manifest


def audit_sources(source_dir, prepared_dir, output_dir):
    features, metadata, recipe, checks, historical = public_sources(source_dir)
    prepared_dir = Path(prepared_dir)
    prepared = read_json(prepared_dir / "manifest.json")
    encodings = read_json(prepared_dir / "base_encoding.json")
    candidates = read_json(prepared_dir / "candidates.json")
    if digest(encodings) != prepared["base_encoding_hash"] or digest(candidates) != prepared["pool_hash"]:
        raise ValueError("Prepared metadata or encoding fingerprint mismatch")
    if prepared != checks["prepared_manifest"]:
        raise ValueError("Prepared encoding provenance differs from historical audit")
    for i, cid in enumerate(features["ids"]):
        if features["z"][i] != encodings[metadata[cid]["architecture_fingerprint"]]:
            raise ValueError("Cached A VAE encoding mismatch")
    current_vae_sha = file_hash(prepared["checkpoint"])
    if current_vae_sha != prepared["checkpoint_sha256"]:
        raise ValueError("Frozen VAE checkpoint fingerprint changed")
    prereg = recipe["preregistration"]
    if file_hash(prereg["path"]) != prereg["sha256"]:
        raise ValueError("Training preregistration file changed")
    source_training = read_json(prereg["path"])["training"]
    if source_training != recipe["actual_preregistration_training"]:
        raise ValueError("Recipe audit differs from actual preregistration")
    value = {
        "status": "PASS", "verified_at_utc": now(), "source_dir": str(source_dir),
        "candidate_count": 179, "architecture_count": 96,
        "public_file_sha256": EXPECTED_PUBLIC_HASHES, "legal_A_ids_sha256": A_ID_HASH,
        "features_semantic_sha256": digest(features), "recipe_sha256": RECIPE_HASH,
        "recipe": recipe["training_protocol_identity"], "training_preregistration": prereg,
        "actual_training_preregistration": source_training,
        "source_protocol_identity": historical["identity"],
        "source_A_label_expected_sha256": historical["private_inputs"]["private/evaluation_A.json"],
        "prepared_manifest": prepared, "current_checkpoint_sha256": current_vae_sha,
        "vae_encoding_all_A_matches_verified_cache": True,
        "HP_transform_all_A_matches_source": True,
        "HP_transform": ["log10(lr)", "dropout", "log2(hidden_dim)", "log10(l2)"],
        "encoding_shape": [179, 12], "HP_shape": [179, 4],
        "audit_reads_training_labels": False, "VAE_forward_or_training_performed": False,
        "known_variance_rule": "validation_se ** 2, where validation_se=std(3 seeds,ddof=1)/sqrt(3); normalized once by selected_y_scale**2",
        "standardization": "20 selected only; float64; ddof0; x floor1e-6, y floor1e-8",
    }
    write_new(Path(output_dir) / "audit/data_source_audit.json", value)
    return value


def _nonspace(stream):
    while True:
        char = stream.read(1)
        if not char or not char.isspace():
            return char


def _json_string(stream, first):
    if first != '"':
        raise ValueError("Expected JSON string")
    result = ['"']
    escaped = False
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError("Truncated JSON string")
        result.append(char)
        if char == '"' and not escaped:
            return json.loads("".join(result))
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True


def _object_value(stream, retain):
    """Skip hidden objects syntactically; do not materialize or decode values."""
    first = _nonspace(stream)
    if first != "{":
        raise ValueError("A label row must be an object")
    result = [first] if retain else None
    depth, in_string, escaped = 1, False, False
    while depth:
        char = stream.read(1)
        if not char:
            raise ValueError("Truncated label object")
        if retain:
            result.append(char)
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
    return json.loads("".join(result)) if retain else None


def selected_objects(stream, allowed_ids):
    allowed = set(allowed_ids)
    if len(allowed) != len(allowed_ids):
        raise ValueError("Duplicate allowed IDs")
    if _nonspace(stream) != "{":
        raise ValueError("Expected top-level A mapping")
    selected, seen, skipped = {}, set(), 0
    next_char = _nonspace(stream)
    while next_char != "}":
        cid = _json_string(stream, next_char)
        if cid in seen:
            raise ValueError("Duplicate source ID")
        seen.add(cid)
        if _nonspace(stream) != ":":
            raise ValueError("Missing JSON colon")
        keep = cid in allowed
        row = _object_value(stream, keep)
        if keep:
            selected[cid] = row
        else:
            skipped += 1
        delimiter = _nonspace(stream)
        if delimiter == "}":
            break
        if delimiter != ",":
            raise ValueError("Missing JSON comma")
        next_char = _nonspace(stream)
    if _nonspace(stream):
        raise ValueError("Trailing JSON content")
    if set(selected) != allowed:
        raise ValueError("One or more selected labels are missing")
    return selected, {"selected_label_rows_decoded": len(selected), "hidden_label_rows_decoded": 0, "hidden_rows_syntactically_skipped": skipped}


def validate_selected_labels(ids, labels):
    if set(ids) != set(labels):
        raise ValueError("Training-label map contains unauthorized or missing IDs")
    means, errors, repeats = [], [], []
    expected_keys = {"validation_mean", "validation_se", "validation_seeds", "training_seeds"}
    for cid in ids:
        row = labels[cid]
        if set(row) != expected_keys or row["training_seeds"] != [0, 1, 2]:
            raise ValueError("Selected label schema or matched GNN seeds changed")
        values = np.asarray(row["validation_seeds"], dtype=np.float64)
        mean, error = float(row["validation_mean"]), float(row["validation_se"])
        if values.shape != (3,) or not np.isfinite(values).all() or not np.isfinite([mean, error]).all() or error < 0 or not ((values >= 0) & (values <= 1)).all() or not 0 <= mean <= 1:
            raise ValueError("Invalid selected observations")
        if not np.isclose(mean, values.mean(), atol=1e-12, rtol=0) or not np.isclose(error, values.std(ddof=1) / math.sqrt(3), atol=1e-12, rtol=0):
            raise ValueError("Selected mean/SE do not agree with three repetitions")
        means.append(mean)
        errors.append(error)
        repeats.append(values.tolist())
    return np.asarray(means), np.asarray(errors), repeats


def release(output_dir, protocol_path, protocol_hash):
    protocol = checked_protocol(protocol_path, protocol_hash)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "selection_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["identity"] != digest({k: v for k, v in manifest.items() if k != "identity"}) or manifest["protocol_sha256"] != protocol_hash:
        raise ValueError("Precommitted selection manifest mismatch")
    ids = manifest["ids"]
    source_dir = Path(manifest["source_dir"])
    features, _, _, _, historical = public_sources(source_dir)
    expected_ids = np.random.default_rng(20260922).choice(sorted(features["ids"]), 20, replace=False).tolist()
    if ids != expected_ids:
        raise ValueError("Selection differs from frozen rule")
    # Label-file fingerprint is checked without parsing labels. Only the selected
    # 20 value objects are deserialized below; the other 159 are discarded.
    label_path = source_dir / "private/evaluation_A.json"
    label_hash = file_hash(label_path)
    if historical["private_inputs"]["private/evaluation_A.json"] != label_hash:
        raise ValueError("A-only label source fingerprint differs from sealed run")
    label_opened_at = now()
    with label_path.open() as stream:
        labels, access = selected_objects(stream, ids)
    y, se, repeats = validate_selected_labels(ids, labels)
    pos = {cid: i for i, cid in enumerate(features["ids"])}
    ix = [pos[cid] for cid in ids]
    z, hp = np.asarray(features["z"], dtype=np.float64)[ix], np.asarray(features["hp"], dtype=np.float64)[ix]
    x = np.concatenate([z, hp], axis=1)
    x_mean = x.mean(axis=0)
    x_scale = np.maximum(x.std(axis=0, ddof=0), protocol["input"]["x_scale_floor"])
    y_mean = float(y.mean())
    y_scale = float(max(y.std(ddof=0), protocol["input"]["y_scale_floor"]))
    data = {
        "ids": ids, "z": z.tolist(), "hp": hp.tolist(), "x_raw": x.tolist(), "y": y.tolist(),
        "measurement_se": se.tolist(), "known_variance": (se ** 2).tolist(), "validation_seeds": repeats,
        "scaler": {"fit_ids": ids, "x_mean": x_mean.tolist(), "x_scale": x_scale.tolist(), "y_mean": y_mean, "y_scale": y_scale, "std_correction": 0, "dtype": "float64", "frozen": True},
        "x_standardized": ((x - x_mean) / x_scale).tolist(), "y_standardized": ((y - y_mean) / y_scale).tolist(),
        "known_variance_standardized": ((se / y_scale) ** 2).tolist(),
        "metadata": {"protocol_sha256": protocol_hash, "selection_manifest_sha256": file_hash(manifest_path), "selection_identity": manifest["identity"], "source_label_path": str(label_path), "source_label_file_sha256": label_hash, "recipe_sha256": RECIPE_HASH, "candidate_metadata": manifest["candidate_metadata"], "labels_opened_at_utc": label_opened_at, "selection_saved_at_utc": manifest["created_at_utc"], "access": access},
    }
    data["identity"] = digest(data)
    write_new(output_dir / "selected_data.json", data)
    write_new(output_dir / "data_access_audit.json", {**access, "actual_released_ids": ids, "selection_saved_before_label_open": manifest["created_at_utc"] < label_opened_at, "unselected_SE_decoded": 0, "B_label_files_opened": 0, "test_metric_files_opened": 0, "training_receives_only_selected20": True, "source_label_file_sha256": label_hash, "selected_data_identity": data["identity"]})
    return data


def mock_checks():
    """The hidden synthetic value is deliberately invalid JSON if decoded."""
    selected = {"validation_mean": .5, "validation_se": .01 / math.sqrt(3), "validation_seeds": [.49, .5, .51], "training_seeds": [0, 1, 2]}
    raw = '{"known":' + json.dumps(selected) + ',"hidden":{"unreadable":NEVER_DECODE_THIS,"nested":[{"quote":"}\\\"{"}]}}'
    labels, access = selected_objects(io.StringIO(raw), ["known"])
    validate_selected_labels(["known"], labels)
    denied = False
    try:
        validate_selected_labels(["known"], {**labels, "hidden": selected})
    except ValueError:
        denied = True
    missing_denied = False
    try:
        selected_objects(io.StringIO(raw), ["missing"])
    except ValueError:
        missing_denied = True
    return {"pass": denied and missing_denied and access["hidden_label_rows_decoded"] == 0, "hidden_poison_value_not_decoded": True, "unauthorized_label_map_rejected": denied, "missing_selected_label_rejected": missing_denied, **access}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["select", "release", "mock", "audit"])
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.mode == "mock":
        result = mock_checks()
        print(json.dumps(result, sort_keys=True))
        if not result["pass"]:
            raise SystemExit(1)
    elif args.mode == "audit":
        result = audit_sources(args.source_dir, args.prepared_dir, args.output_dir)
        print(json.dumps({"audit_status": result["status"], "all_A_encodings_verified": True, "actual_labels_read": 0}))
    elif args.mode == "select":
        result = select(args.source_dir, args.output_dir, args.protocol, args.protocol_sha256)
        print(json.dumps({"selected_count": len(result["ids"]), "selection_identity": result["identity"], "actual_labels_read": 0}))
    else:
        result = release(args.output_dir, args.protocol, args.protocol_sha256)
        print(json.dumps({"released_count": len(result["ids"]), "selected_data_identity": result["identity"], "hidden_labels_decoded": 0}))


if __name__ == "__main__":
    main()
