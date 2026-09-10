"""Full-sequence, denormalized VQ reconstruction evaluation.

The model consumes its original source-order coordinates. Anatomical groups are
applied only AFTER reconstruction, using the independently verified layout.
This is not the normalized 64-frame validation loss used for model selection.
"""
from __future__ import annotations
import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import math
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / "configs" / "keypoint_mapping.json"


def file_info(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return {"path": str(path), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is None or isinstance(value, (int, str, bool)):
        return value
    return repr(value)


@contextlib.contextmanager
def cpu_codebook_allocation():
    """Replace only hard-coded CUDA zero allocation while constructing a model.

    Strict checkpoint loading immediately replaces these zero buffers. No
    quantization operation or checkpoint tensor is altered by this shim.
    """
    from models import quantize_cnn
    classes = [quantize_cnn.QuantizeEMAReset, quantize_cnn.QuantizeEMA]
    saved = {cls: cls.reset_codebook for cls in classes}
    def reset(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer("codebook", torch.zeros(self.nb_code, self.code_dim, device="cpu"))
    try:
        for cls in classes:
            cls.reset_codebook = reset
        yield
    finally:
        for cls, function in saved.items():
            cls.reset_codebook = function


def load_model(checkpoint, variant, source_dir, device):
    source_dir = Path(source_dir).resolve()
    sys.path.insert(0, str(source_dir))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "args" not in ckpt or "model" not in ckpt:
        raise ValueError("Checkpoint must contain args and model")
    values = ckpt["args"] if isinstance(ckpt["args"], dict) else vars(ckpt["args"])
    args = SimpleNamespace(**values)
    common = dict(code_dim=args.code_dim,
        output_emb_width=args.output_emb_width, down_t=args.down_t,
        stride_t=args.stride_t, width=args.width, depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=getattr(args, "vq_act", "relu"), norm=getattr(args, "vq_norm", None))
    with cpu_codebook_allocation():
        if variant in {"baseline", "single_large"}:
            from models.vqvae import VQVAE_251
            model = VQVAE_251(args, nb_code=args.nb_code, **common)
        elif variant == "tlrvq":
            from models.vqvae_residual import ResidualVQVAE
            model = ResidualVQVAE(args, nb_code=args.nb_code,
                nb_code_residual=getattr(args, "nb_code_residual", None), **common)
        elif variant in {"pamvq", "pamrvq", "pamrvq_random"}:
            if getattr(args, "layout_version", None) != "reviewer55_verified_v1":
                raise ValueError("New multistream evaluation rejects unversioned layouts")
            expected_grouping = "random" if variant == "pamrvq_random" else "anatomical"
            if getattr(args, "grouping", None) != expected_grouping:
                raise ValueError(f"{variant} requires grouping={expected_grouping}")
            if "_extra_state" not in ckpt["model"]:
                raise ValueError("Checkpoint lacks strict multistream layout metadata")
            if variant == "pamvq":
                from models.vqvae_multistream import MultiStreamVQVAE
                stream_codes = {
                    name: getattr(args, "nb_code_" + name, None) or args.nb_code
                    for name in ("body", "hand", "face")
                }
                model = MultiStreamVQVAE(args, dataset_name=args.dataname,
                    nb_code=args.nb_code, stream_codes=stream_codes,
                    enc_widths=getattr(args, "enc_widths", None), **common)
            else:
                from models.vqvae_multistream_residual import MultiStreamResidualVQVAE
                base_codes = {name: getattr(args, "nb_base_" + name) for name in ("body", "hand", "face")}
                residual_codes = {name: getattr(args, "nb_res_" + name) for name in ("body", "hand", "face")}
                model = MultiStreamResidualVQVAE(args, dataset_name=args.dataname,
                    stream_codes_base=base_codes, stream_codes_residual=residual_codes,
                    enc_widths=getattr(args, "enc_widths", None), **common)
        else:
            raise ValueError(f"Variant {variant} is not supported")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ckpt, values


def decode_paths(model, x, variant, include_base_only):
    if variant == "tlrvq":
        base, residual = model.encode(x)
        full = model.forward_decoder(base, residual)
        outputs = {"full": full}
        if include_base_only:
            # Delete the residual VECTOR contribution. Token index zero would
            # instead add an ordinary residual codeword and is not a layer ablation.
            base_vectors = model.quantizer_base.dequantize(base)
            outputs["base_only"] = model.postprocess(model.decoder(
                base_vectors.permute(0, 2, 1).contiguous()))
        codes = {"base": base, "residual": residual}
    elif variant in {"pamvq", "pamrvq", "pamrvq_random"}:
        codes = model.encode(x)
        outputs = {"full": model.forward_decoder(codes)}
    else:
        indices = model.encode(x)
        outputs = {"full": model.forward_decoder(indices)}
        codes = {"single": indices}
    return outputs, codes


def common_length_metrics(predicted, truth, groups, fps=None):
    length = min(len(predicted), len(truth))
    if length < 1:
        raise ValueError("No common reconstruction frames")
    predicted = np.asarray(predicted[:length], dtype=np.float64)
    truth = np.asarray(truth[:length], dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError("Prediction and ground truth shape mismatch")
    position_distances = np.linalg.norm(predicted - truth, axis=-1)
    if length > 1:
        velocity_distances = np.linalg.norm(np.diff(predicted, axis=0) - np.diff(truth, axis=0), axis=-1)
        if fps:
            velocity_distances *= fps
    else:
        velocity_distances = None
    result = {}
    for name, joints in groups.items():
        result[name] = {"position_error": float(position_distances[:, joints].mean()),
            "velocity_error": float(velocity_distances[:, joints].mean()) if velocity_distances is not None else None,
            "joint_count": len(joints), "common_frames": length,
            "velocity_steps": max(0, length - 1)}
    return result


def summary_rows(rows):
    groups = sorted({(row["decode_path"], row["group"]) for row in rows})
    output = []
    for path, group in groups:
        subset = [r for r in rows if r["decode_path"] == path and r["group"] == group]
        row = {"decode_path": path, "group": group, "samples": len(subset),
               "joint_count": subset[0]["joint_count"]}
        for metric in ("position_error", "velocity_error"):
            values = np.asarray([r[metric] for r in subset if r[metric] is not None])
            row[metric + "_sample_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_sample_std"] = float(values.std()) if len(values) else None
            row[metric + "_median"] = float(np.median(values)) if len(values) else None
            row[metric + "_valid_samples"] = len(values)
        output.append(row)
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", choices=("phix", "csl"), required=True)
    p.add_argument("--variant", choices=("baseline", "single_large", "tlrvq",
                   "pamvq", "pamrvq", "pamrvq_random"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-dir", default=str(ROOT / "runtime" / "code" / "src"))
    p.add_argument("--data-root", default=os.environ.get("SLP_SIGN_DATA_DIR", str(ROOT / "runtime" / "data")))
    p.add_argument("--data-file")
    p.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    p.add_argument("--split", choices=("test", "dev"), default="test")
    p.add_argument("--stats-dir", help="Defaults to checkpoint parent; never recomputed on test data")
    p.add_argument("--include-base-only", action="store_true")
    p.add_argument("--fps", type=float, help="Omit for paper-style frame-difference velocity error")
    p.add_argument("--limit", type=int, help="Smoke test only, recorded as incomplete evaluation")
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--save-recon", action="store_true",
                   help="Save full decoded poses as gzip SLRTP pickle; refuses existing export files")
    args = p.parse_args()
    if args.include_base_only and args.variant != "tlrvq":
        p.error("--include-base-only is only valid for tlrvq")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        p.error("--fps must be positive")
    if args.cpu_threads < 1:
        p.error("--cpu-threads must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_paths = {"full": out / "reconstruction_full.pickle"}
    if args.include_base_only:
        export_paths["base_only"] = out / "reconstruction_base_only.pickle"
    if args.save_recon:
        for path in export_paths.values():
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite reconstruction export {path}")
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8-sig"))
    dataset_mapping = mapping["datasets"][args.dataset]
    n_keypoints = len(dataset_mapping["canonical_to_source"])
    groups = dict(dataset_mapping["source_groups"])
    groups["all_keypoints"] = list(range(n_keypoints))
    groups = {key: list(value) for key, value in groups.items()}
    stem = "phix_lift3d" if args.dataset == "phix" else "csl_daily_lift3d"
    data_file = Path(args.data_file) if args.data_file else Path(args.data_root) / args.dataset / f"{stem}.{args.split}.pt"
    ckpt_path = Path(args.checkpoint)
    stats_dir = Path(args.stats_dir) if args.stats_dir else ckpt_path.parent
    mean_path, std_path = stats_dir / "mean.npy", stats_dir / "std.npy"
    mean, std = np.load(mean_path), np.load(std_path)
    if mean.shape != (n_keypoints * 3,) or std.shape != mean.shape:
        raise ValueError("Saved normalization stats do not match the configured pose dimension")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Saved normalization stats contain invalid values")
    model, checkpoint, checkpoint_args = load_model(ckpt_path, args.variant, args.source_dir, args.device)
    expected = "phix_lift3d" if args.dataset == "phix" else "csl_lift3d"
    if checkpoint_args.get("dataname") != expected:
        raise ValueError(f"Checkpoint dataname {checkpoint_args.get('dataname')} does not match {expected}")
    source_dir = Path(args.source_dir)
    model_files = {
        "baseline": "vqvae.py", "single_large": "vqvae.py",
        "tlrvq": "vqvae_residual.py", "pamvq": "vqvae_multistream.py",
        "pamrvq": "vqvae_multistream_residual.py",
        "pamrvq_random": "vqvae_multistream_residual.py",
    }
    inputs_to_hash = {
        "checkpoint": ckpt_path, "mean": mean_path, "std": std_path,
        "dataset": data_file, "mapping": args.mapping, "evaluator": __file__,
        "model_class": source_dir / "models" / model_files[args.variant],
        "quantizer": source_dir / "models" / "quantize_cnn.py",
        "encoder_decoder": source_dir / "models" / "encdec.py",
    }
    if args.variant in {"pamvq", "pamrvq", "pamrvq_random"}:
        inputs_to_hash["grouping_implementation"] = source_dir / "models" / "keypoint_grouping.py"
    # Capture source hashes before inference, rather than after a possibly long run.
    provenance = {name: file_info(path) for name, path in inputs_to_hash.items()}
    raw = torch.load(data_file, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Expected source dataset dictionary keyed by sample ID")
    # Minimum input length accepted by all stride convolutions. Only sequences
    # below this length receive repeated-last-frame padding, which is logged.
    minimum_length = 1
    stride = int(checkpoint_args["stride_t"])
    for _ in range(int(checkpoint_args["down_t"])):
        minimum_length = (minimum_length - 1) * stride - 2 * (stride // 2) + 2 * stride
    rows, coverage, used_codes = [], [], {}
    reconstruction_exports = {name: [] for name in export_paths} if args.save_recon else {}
    selected_ids = sorted(raw)
    if args.limit is not None:
        selected_ids = selected_ids[:args.limit]
    with torch.inference_mode():
        for number, sample_id in enumerate(selected_ids, start=1):
            row = raw[sample_id]
            value = row["poses_3d"]
            value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if value.ndim == 2:
                value = value.reshape(len(value), n_keypoints, 3)
            if value.shape[1:] != (n_keypoints, 3) or len(value) < 1 or not np.isfinite(value).all():
                raise ValueError(f"Invalid full-sequence coordinates for {sample_id}")
            truth = value.astype(np.float32, copy=False)
            flat = truth.reshape(len(truth), -1)
            normalized = (flat - mean) / std
            padding = max(0, minimum_length - len(normalized))
            if padding:
                normalized = np.concatenate((normalized, np.repeat(normalized[-1:], padding, axis=0)))
            tensor = torch.from_numpy(np.asarray(normalized, dtype=np.float32)).unsqueeze(0).to(args.device)
            outputs, indices = decode_paths(model, tensor, args.variant, args.include_base_only)
            for key, values in indices.items():
                used_codes.setdefault(key, set()).update(values.detach().cpu().reshape(-1).tolist())
            for decode_path, output in outputs.items():
                predicted_flat = output.squeeze(0).detach().cpu().numpy()
                predicted = (predicted_flat * std + mean).reshape(-1, n_keypoints, 3)
                if not np.isfinite(predicted).all():
                    raise ValueError(f"Nonfinite reconstruction for {sample_id}/{decode_path}")
                metrics = common_length_metrics(predicted, truth, groups, args.fps)
                common = min(len(predicted), len(truth))
                item = {"sample_id": str(sample_id), "decode_path": decode_path,
                    "input_frames": len(truth), "input_padding_frames": padding,
                    "reconstructed_frames": len(predicted), "common_frames": common,
                    "unscored_truth_tail_frames": len(truth) - common,
                    "input_text": row.get("text", row.get("translation", ""))}
                coverage.append(item)
                if args.save_recon:
                    # Keep the full decoder output in source order. The scoring
                    # intersection and any short-input padding remain in coverage.
                    reconstruction_exports[decode_path].append({
                        "name": str(sample_id),
                        "signer": row.get("speaker", row.get("signer", "")),
                        "gloss": row.get("gloss", ""),
                        "text": row.get("text", row.get("translation", "")),
                        "sign": torch.from_numpy(np.asarray(predicted.reshape(len(predicted), -1),
                                                          dtype=np.float32).copy()),
                    })
                for group, values in metrics.items():
                    rows.append({**item, "group": group, **values})
            if number % 100 == 0:
                print(f"{number}/{len(selected_ids)} full sequences evaluated", flush=True)
    table = summary_rows(rows)
    exported = {}
    for decode_path, records in reconstruction_exports.items():
        path = export_paths[decode_path]
        with path.open("xb") as raw_handle:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as compressed:
                pickle.dump(records, compressed, protocol=pickle.HIGHEST_PROTOCOL)
        exported[decode_path] = {"source": file_info(path), "samples": len(records),
            "format": "gzip pickle list of name/signer/gloss/text/sign(T,534) float32 CPU tensors",
            "coordinates": "full decoder output, denormalized, original source order",
            "gloss_use": "copied metadata only; never input to the reconstruction model"}
    payload = {
        "schema": "reviewer55.full_sequence_vq_reconstruction.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "split": args.split, "variant": args.variant,
        "device": args.device, "checkpoint_args": checkpoint_args,
        "checkpoint_iteration": checkpoint.get("iter"),
        "checkpoint_selection_normalized_window_loss": checkpoint.get("dev_recon"),
        "normalization": "saved training statistics, source order, inverse transform before scoring",
        "input_order": "source dataset order; no input permutation",
        "metric_groups": groups, "mapping_confidence": dataset_mapping["confidence"],
        "aggregation": "mean Euclidean position/velocity error per sample, then arithmetic mean across samples",
        "velocity_units": "coordinate units / second" if args.fps else "coordinate units / frame step",
        "fps": args.fps, "whole_sequence": True, "window_cropping": False,
        "common_length_policy": "prefix of length min(original_T,reconstructed_T)",
        "minimum_length_padding": "repeat final frame only if encoder would otherwise produce no latent token",
        "residual_ablation": "omit residual vector contribution, never residual codeword zero" if args.include_base_only else None,
        "source_samples": len(raw), "evaluated_sample_ids": list(map(str, selected_ids)),
        "complete_split": len(selected_ids) == len(raw),
        "limit": args.limit, "coverage": coverage,
        "codebook_distinct_indices_used": {k: len(v) for k, v in used_codes.items()},
        "summary": table, "provenance": provenance,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        "cpu_constructor_compatibility": "EMA zero-buffer allocation only, followed by strict state_dict loading",
        "model_grouping_metadata": model.export_grouping_metadata() if hasattr(model, "export_grouping_metadata") else None,
        "exported_reconstructions": exported,
    }
    (out / "summary.json").write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "per_sample.json").write_text(json.dumps(json_safe(rows), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(out / "per_sample.csv", rows)
    write_csv(out / "summary.csv", table)
    print(json.dumps({"samples": len(selected_ids), "complete_split": payload["complete_split"],
                      "summary": table, "out": str(out.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
