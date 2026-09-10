"""Configuration-driven CPU diagnostics for Cartesian keypoint sequences.

No dataset layout, hand offset, or edge topology is built into this module.
Edges must be supplied explicitly and classified as rigid/reference/proxy.
The term "rigid" is the caller's operational anatomical assumption, not a
conclusion established by this program. See README.md for metric definitions.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import pickle
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
VERSION = "1.0.1"


def cpu_torch():
    import torch
    torch.set_num_threads(1)
    return torch


class CpuUnpickler(pickle.Unpickler):
    """Map tensor storage to CPU when reading trusted local pickle files."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: cpu_torch().load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def file_info(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return {"path": str(p), "sha256": h.hexdigest(), "bytes": p.stat().st_size}


def as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_rows(obj, pose_field=None, id_field=None):
    """Yield (id, coordinates, metadata), accepting common local export formats."""
    if isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], list):
        obj = obj["samples"]
    fields = [pose_field] if pose_field else ["poses_3d", "sign", "poses", "keypoints"]
    # A dictionary containing a pose field is a single record.
    if isinstance(obj, dict) and any(k in obj for k in fields):
        items = [(obj.get(id_field or "id", obj.get("name", "sample")), obj)]
    elif isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    else:
        arr = as_array(obj)
        items = enumerate(arr) if arr.ndim == 4 else [("sample", arr)]
    for key, value in items:
        if isinstance(value, dict):
            field = next((k for k in fields if k in value), None)
            if field is None:
                raise ValueError(f"Record {key!r}: missing pose field {fields}")
            sid = value.get(id_field, key) if id_field else value.get(
                "name", value.get("id", value.get("sample_id", key))
            )
            yield str(sid), value[field], value
        else:
            yield str(key), value, {}


def load_samples(path, pose_field=None, id_field=None):
    """Read input only. NPZ/NPY loading never enables object-pickle support."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as archive:
            if "poses" in archive and archive["poses"].ndim == 4:
                ids = archive["sample_ids"] if "sample_ids" in archive else None
                lengths = archive["lengths"] if "lengths" in archive else None
                for i, value in enumerate(archive["poses"]):
                    meta = {"length": int(lengths[i])} if lengths is not None else {}
                    yield str(ids[i]) if ids is not None else str(i), value, meta
            else:
                for key in archive.files:
                    if archive[key].ndim >= 2:
                        yield key, archive[key], {}
        return
    if suffix == ".npy":
        yield from parse_rows(np.load(p, mmap_mode="r", allow_pickle=False))
        return
    if suffix == ".jsonl":
        with p.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    yield from parse_rows(json.loads(line), pose_field, id_field)
        return
    if suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8-sig"))
    elif suffix in {".pt", ".pth"}:
        obj = cpu_torch().load(p, map_location="cpu", weights_only=False)
    else:
        with p.open("rb") as handle:
            magic = handle.read(2)
        opener = gzip.open if magic == b"\x1f\x8b" else open
        with opener(p, "rb") as handle:
            obj = CpuUnpickler(handle).load()
    yield from parse_rows(obj, pose_field, id_field)
    del obj
    gc.collect()


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("config.schema_version must be 1")
    n = int(config["n_keypoints"])
    dims = int(config["n_dims"])
    if n < 2 or dims not in (2, 3):
        raise ValueError("n_keypoints >= 2 and n_dims in {2,3} required")
    if config.get("coordinate_space") != "cartesian":
        raise ValueError("Input must be Cartesian coordinates, not per-coordinate z-scores")
    source_n = int(config.get("source_keypoints", n))
    mapping = config.get("canonical_to_source", list(range(n)))
    if len(mapping) != n or len(set(mapping)) != n:
        raise ValueError("canonical_to_source must contain n_keypoints unique indices")
    if any(not isinstance(x, int) or not 0 <= x < source_n for x in mapping):
        raise ValueError("canonical_to_source contains an invalid source index")
    edges = config.get("edges", [])
    if not edges:
        raise ValueError("An explicit non-empty edge configuration is required")
    names, pairs = set(), set()
    for edge in edges:
        name = edge["name"]
        pair = tuple(sorted((int(edge["a"]), int(edge["b"]))))
        if name in names or pair in pairs or pair[0] == pair[1]:
            raise ValueError("Edge names and undirected endpoint pairs must be unique")
        if not (0 <= pair[0] < pair[1] < n):
            raise ValueError(f"Invalid endpoints for {name}")
        if edge.get("kind") not in {"rigid", "reference", "proxy"}:
            raise ValueError(f"{name}: kind must be rigid, reference, or proxy")
        if edge["kind"] == "rigid" and config.get("mapping_status") != "confirmed":
            raise ValueError("Rigid metrics require mapping_status=confirmed")
        if not edge.get("group") or not edge.get("justification"):
            raise ValueError(f"{name}: group and anatomical justification are required")
        names.add(name)
        pairs.add(pair)
    for name, group in config.get("motion_groups", {}).items():
        if not group or len(group) != len(set(group)) or any(
            not isinstance(x, int) or not 0 <= x < n for x in group
        ):
            raise ValueError(f"Invalid motion group {name}")
    root = config.get("root_index")
    if root is not None and (not isinstance(root, int) or not 0 <= root < n):
        raise ValueError("root_index outside canonical keypoint indices")
    return config


def canonical_pose(value, config, metadata=None, length_field=None):
    arr = as_array(value)
    source_n = int(config.get("source_keypoints", config["n_keypoints"]))
    dims = config["n_dims"]
    if arr.ndim == 2:
        if arr.shape[1] != source_n * dims:
            raise ValueError(f"Flat pose width {arr.shape[1]} != {source_n}*{dims}")
        arr = arr.reshape(arr.shape[0], source_n, dims)
    if arr.ndim != 3 or arr.shape[1:] != (source_n, dims):
        raise ValueError(f"Pose shape {arr.shape} does not match configured layout")
    if length_field:
        if metadata is None or length_field not in metadata:
            raise ValueError(f"Explicit length field {length_field!r} missing")
        length = int(metadata[length_field])
        if not 0 <= length <= len(arr):
            raise ValueError("Length field lies outside the stored frame range")
        arr = arr[:length]
    mapping = config.get("canonical_to_source", list(range(config["n_keypoints"])))
    return np.asarray(arr[:, mapping, :], dtype=np.float64)


def moments(data, axis=0):
    valid = np.isfinite(data)
    count = valid.sum(axis=axis)
    sums = np.where(valid, data, 0).sum(axis=axis)
    mean = np.divide(sums, count, out=np.full(np.shape(sums), np.nan), where=count > 0)
    centered = np.where(valid, data - np.expand_dims(mean, axis), 0)
    var = np.divide((centered * centered).sum(axis=axis), count,
                    out=np.full(np.shape(sums), np.nan), where=count > 0)
    return count, mean, np.maximum(var, 0)


def mean_finite(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else float("nan")


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "median": None, "p95": None}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95))}


def edge_statistics(pose, edges, epsilon):
    a = np.asarray([e["a"] for e in edges])
    b = np.asarray([e["b"] for e in edges])
    lengths = np.linalg.norm(pose[:, a] - pose[:, b], axis=-1)
    counts, mean, var = moments(lengths)
    usable = (counts >= 2) & (mean > epsilon)
    cv = np.divide(np.sqrt(var), mean, out=np.full(len(edges), np.nan), where=usable)
    change = np.abs(np.diff(lengths, axis=0))
    _, step_abs, _ = moments(change)
    step_rel = np.divide(step_abs, mean, out=np.full(len(edges), np.nan), where=usable)
    # Retain zero-length and nonfinite counts; do not silently label these stable.
    degenerate = np.isfinite(lengths) & (lengths <= epsilon)
    return {"lengths": lengths, "frame_count": counts, "mean_length": mean,
            "std_length": np.sqrt(var), "cv": cv, "step_abs": step_abs,
            "step_relative": step_rel, "usable": usable,
            "degenerate_frames": degenerate.sum(axis=0)}


def collect_reference(records, config, epsilon, length_field=None, max_samples=None):
    edges = config["edges"]
    e = len(edges)
    sample_count = np.zeros(e, dtype=np.int64)
    sum_mean = np.zeros(e)
    sum_mean_squared = np.zeros(e)
    frame_count = np.zeros(e, dtype=np.int64)
    sum_length = np.zeros(e)
    sum_length_squared = np.zeros(e)
    samples = 0
    skipped_short = 0
    ids = set()
    for sid, raw, metadata in records:
        if sid in ids:
            raise ValueError(f"Duplicate reference sample id {sid}")
        ids.add(sid)
        pose = canonical_pose(raw, config, metadata, length_field)
        if len(pose) < 2:
            skipped_short += 1
            continue
        st = edge_statistics(pose, edges, epsilon)
        valid_mean = st["usable"]
        sample_count += valid_mean
        sum_mean += np.where(valid_mean, st["mean_length"], 0)
        sum_mean_squared += np.where(valid_mean, st["mean_length"] ** 2, 0)
        lengths = st["lengths"]
        valid = np.isfinite(lengths)
        frame_count += valid.sum(axis=0)
        sum_length += np.where(valid, lengths, 0).sum(axis=0)
        sum_length_squared += np.where(valid, lengths ** 2, 0).sum(axis=0)
        samples += 1
        if max_samples is not None and samples >= max_samples:
            break
    def finish(count, sums, squares):
        mu = np.divide(sums, count, out=np.full(e, np.nan), where=count > 0)
        variance = np.divide(squares, count, out=np.full(e, np.nan), where=count > 0) - mu ** 2
        sigma = np.sqrt(np.maximum(variance, 0))
        eligible = (count >= 2) & np.isfinite(sigma) & (sigma > epsilon)
        return {"count": count.tolist(), "mean": mu.tolist(), "std_ddof0": sigma.tolist(),
                "eligible": eligible.tolist()}
    return {"sample_mean_reference": finish(sample_count, sum_mean, sum_mean_squared),
            "pooled_frame_reference": finish(frame_count, sum_length, sum_length_squared),
            "valid_samples": samples, "skipped_short": skipped_short,
            "zero_variance_policy": "exclude edge from sigma ratios, never floor sigma",
            "sample_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()}


def motion_statistics(pose, joint_ids, dt):
    x = pose[:, joint_ids, :]
    _, center, _ = moments(x)
    radius_squared = np.sum((x - center) ** 2, axis=-1)
    _, mean_radius_squared, _ = moments(radius_squared)
    amplitude = np.sqrt(mean_radius_squared)
    step_speed = np.linalg.norm(np.diff(x, axis=0), axis=-1) / dt
    acceleration = np.linalg.norm(np.diff(x, n=2, axis=0), axis=-1) / (dt * dt)
    return {"rms_displacement_about_own_time_mean": mean_finite(amplitude),
            "mean_speed": mean_finite(step_speed), "p95_speed": describe(step_speed.ravel())["p95"],
            "mean_acceleration": mean_finite(acceleration),
            "p95_acceleration": describe(acceleration.ravel())["p95"]}


def edge_groups(config):
    groups = {"all_diagnostic_edges": list(range(len(config["edges"])))}
    for i, edge in enumerate(config["edges"]):
        groups.setdefault("kind:" + edge["kind"], []).append(i)
        groups.setdefault("group:" + edge["group"] + ":" + edge["kind"], []).append(i)
    return groups


def evaluate(records, config, reference=None, fps=None, epsilon=1e-8,
             length_field=None, max_samples=None, reference_split=None):
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be finite and positive")
    dt = 1 / fps if fps else 1.0
    edges = config["edges"]
    groups = edge_groups(config)
    metric_values = defaultdict(lambda: defaultdict(list))
    group_counts = defaultdict(lambda: defaultdict(int))
    edge_metric_values = defaultdict(lambda: defaultdict(list))
    sample_rows = []
    ids = set()
    skipped_short = 0
    min_frames, max_frames, total_frames = None, 0, 0
    sigma_arrays = {}
    if reference is not None:
        for key in ("sample_mean_reference", "pooled_frame_reference"):
            sigma_arrays[key] = {k: np.asarray(reference[key][k])
                for k in ("mean", "std_ddof0", "eligible")}
    motion_groups = config.get("motion_groups", {"all_keypoints": list(range(config["n_keypoints"]))})
    for sid, raw, metadata in records:
        if sid in ids:
            raise ValueError(f"Duplicate prediction sample id {sid}")
        ids.add(sid)
        pose = canonical_pose(raw, config, metadata, length_field)
        if len(pose) < 2:
            skipped_short += 1
            continue
        t = len(pose)
        min_frames = t if min_frames is None else min(min_frames, t)
        max_frames, total_frames = max(max_frames, t), total_frames + t
        st = edge_statistics(pose, edges, epsilon)
        for i, edge in enumerate(edges):
            for metric in ("mean_length", "std_length", "cv", "step_abs", "step_relative"):
                edge_metric_values[edge["name"]][metric].append(float(st[metric][i]))
        mean_flags, mean_valid, frame_flags, frame_valid = None, None, None, None
        if reference is not None:
            r = sigma_arrays["sample_mean_reference"]
            mean_valid = st["usable"] & r["eligible"]
            mean_flags = mean_valid & (np.abs(st["mean_length"] - r["mean"]) > 3 * r["std_ddof0"])
            r = sigma_arrays["pooled_frame_reference"]
            frame_valid = np.isfinite(st["lengths"]) & r["eligible"][None, :]
            frame_flags = frame_valid & (np.abs(st["lengths"] - r["mean"]) > 3 * r["std_ddof0"])
        for name, ix in groups.items():
            row = {"sample_id": sid, "frames": t, "scope": name, "analysis": "edge"}
            for metric in ("cv", "step_abs", "step_relative"):
                row[metric] = mean_finite(st[metric][ix])
                metric_values[name][metric].append(row[metric])
            valid_edge_frames = int(st["frame_count"][ix].sum())
            row["degenerate_edge_frame_fraction"] = (
                float(st["degenerate_frames"][ix].sum() / valid_edge_frames)
                if valid_edge_frames else float("nan")
            )
            row["valid_cv_edges"] = int(np.isfinite(st["cv"][ix]).sum())
            row["nonfinite_edge_frames"] = int(t * len(ix) - st["frame_count"][ix].sum())
            for prefix, flags, valid in (
                ("sample_mean_3sigma", mean_flags, mean_valid),
                ("frame_3sigma", frame_flags, frame_valid)
            ):
                if valid is None:
                    continue
                selected_valid = valid[ix] if valid.ndim == 1 else valid[:, ix]
                selected_flags = flags[ix] if flags.ndim == 1 else flags[:, ix]
                count, den = int(selected_flags.sum()), int(selected_valid.sum())
                row[prefix + "_outliers"], row[prefix + "_denominator"] = count, den
                row[prefix + "_fraction"] = count / den if den else float("nan")
                metric_values[name][prefix + "_fraction"].append(row[prefix + "_fraction"])
                group_counts[name][prefix + "_outliers"] += count
                group_counts[name][prefix + "_denominator"] += den
            sample_rows.append(row)
        for name, joints in motion_groups.items():
            variants = {"absolute": pose}
            if config.get("root_index") is not None:
                root = config["root_index"]
                variants["root_relative"] = pose - pose[:, root:root + 1, :]
            for coordinate_type, sequence in variants.items():
                scope = f"motion:{name}:{coordinate_type}"
                values = motion_statistics(sequence, joints, dt)
                sample_rows.append({"sample_id": sid, "frames": t, "scope": scope,
                                    "analysis": "motion", **values})
                for key, value in values.items():
                    metric_values[scope][key].append(float("nan") if value is None else value)
        if max_samples is not None and (len(ids) - skipped_short) >= max_samples:
            break
    summary = {}
    for scope, metrics in metric_values.items():
        summary[scope] = {key: describe(values) for key, values in metrics.items()}
        if scope in group_counts:
            counts = dict(group_counts[scope])
            for prefix in ("sample_mean_3sigma", "frame_3sigma"):
                den = counts[prefix + "_denominator"]
                counts[prefix + "_micro_fraction"] = counts[prefix + "_outliers"] / den if den else None
            summary[scope]["pooled_counts"] = counts
    edge_rows = []
    for edge in edges:
        row = dict(edge)
        for key, values in edge_metric_values[edge["name"]].items():
            for stat, value in describe(values).items():
                row[f"{key}_{stat}"] = value
        edge_rows.append(row)
    return {"scopes": summary, "valid_samples": len(ids) - skipped_short,
            "skipped_short_samples": skipped_short, "total_frames": total_frames,
            "min_frames": min_frames, "max_frames": max_frames,
            "fps": fps,
            "motion_velocity_units": "coordinate_units_per_second" if fps else "coordinate_units_per_frame_step",
            "motion_acceleration_units": "coordinate_units_per_second_squared" if fps else "coordinate_units_per_frame_step_squared",
            "bone_length_change_units": "coordinate_units_per_observed_frame_step",
            "bone_length_relative_change_units": "fraction_per_observed_frame_step",
            "cv_aggregation": "arithmetic mean of per-sample means over valid configured edges",
            "reference_split": reference_split,
            "reference_is_evaluation_data": reference_split in {"dev", "test"},
            "sigma_interpretation": "reference-distribution outlier, not proof of physical invalidity",
            "mapping_status": config.get("mapping_status", "unconfirmed"),
            "sample_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
            }, sample_rows, edge_rows


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_json(row))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference", help="Prefer training data only for fitted reference bands")
    p.add_argument("--reference-split", choices=("train", "dev", "test", "custom"))
    p.add_argument("--pose-field")
    p.add_argument("--reference-pose-field")
    p.add_argument("--id-field")
    p.add_argument("--length-field")
    p.add_argument("--reference-length-field")
    p.add_argument("--fps", type=float, help="Actual input frame rate after every preprocessing step")
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--max-samples", type=int, help="Smoke tests only; output records truncation")
    args = p.parse_args()
    if args.reference and not args.reference_split:
        p.error("--reference-split is required with --reference")
    if args.max_samples is not None and args.max_samples < 1:
        p.error("--max-samples must be positive")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        p.error("--epsilon must be finite and positive")
    config = validate_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        p.error("Output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    reference = None
    provenance = {"predictions": file_info(args.pred), "config": file_info(args.config),
                  "script": file_info(__file__)}
    if args.reference:
        provenance["reference"] = file_info(args.reference)
        reference = collect_reference(
            load_samples(args.reference, args.reference_pose_field, args.id_field),
            config, args.epsilon, args.reference_length_field, args.max_samples
        )
    results, samples, edges = evaluate(
        load_samples(args.pred, args.pose_field, args.id_field), config,
        reference, args.fps, args.epsilon, args.length_field,
        args.max_samples, args.reference_split
    )
    payload = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "device": "cpu"}, "provenance": provenance,
               "protocol_config": config, "max_samples": args.max_samples,
               "reference_statistics": reference, "results": results}
    (out / "summary.json").write_text(
        json.dumps(clean_json(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    write_csv(out / "per_sample_metrics.csv", samples)
    write_csv(out / "per_edge_metrics.csv", edges)
    print(json.dumps({"valid_samples": results["valid_samples"],
                      "scopes": list(results["scopes"]), "out": str(out.resolve())},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
