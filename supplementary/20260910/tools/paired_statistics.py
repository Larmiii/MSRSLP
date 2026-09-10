"""CPU-only paired sample bootstrap and separate training-seed summaries.

BLEU inputs must already contain the exact normalized/tokenized strings used by
the evaluator. This script does not lowercase, normalize, or split Chinese text.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

sys.dont_write_bytecode = True
import numpy as np

BASE = Path(__file__).resolve().parent
DEFAULT_METRIC_SOURCE = BASE.parent / "runtime/bt_eval_kit/slrtp_official/external_metrics/sacrebleu.py"
SCHEMA = "reviewer55.paired_statistics.v1"


class InputError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_record(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def load_metric(path=DEFAULT_METRIC_SOURCE):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("reviewer55_raw_sacrebleu_reference", path)
    if spec is None or spec.loader is None:
        raise InputError(f"Cannot import reference metric: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "NGRAM_ORDER", None) != 4 or not callable(getattr(module, "raw_corpus_bleu", None)):
        raise InputError("Expected project raw_corpus_bleu with four n-gram orders")
    check = module.raw_corpus_bleu(["a b c d"], [["a b c d"]], smooth_value=0.0)
    if not hasattr(check, "scores") or len(check.scores) != 4 or abs(check.scores[3] - 100) > 1e-10:
        raise InputError("Unsupported reference metric API or unexpected raw BLEU behavior")
    return module


def read_jsonl(path, id_key="id", reference_key="reference", prediction_key="prediction"):
    rows = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise InputError(f"{path}:{line_number}: empty JSONL record")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise InputError(f"{path}:{line_number}: record must be an object")
            for key in (id_key, reference_key, prediction_key):
                if key not in row:
                    raise InputError(f"{path}:{line_number}: missing field {key!r}")
            sid = row[id_key]
            if not isinstance(sid, str) or not sid.strip():
                raise InputError(f"{path}:{line_number}: ID must be a nonempty string")
            if sid in rows:
                raise InputError(f"{path}:{line_number}: duplicate ID {sid!r}")
            if not isinstance(row[reference_key], str) or not isinstance(row[prediction_key], str):
                raise InputError(f"{path}:{line_number}: reference and prediction must be strings")
            rows[sid] = {"reference": row[reference_key], "prediction": row[prediction_key]}
    if not rows:
        raise InputError(f"Empty input: {path}")
    return rows


def align_rows(a, b):
    missing_b = sorted(set(a) - set(b))
    missing_a = sorted(set(b) - set(a))
    if missing_a or missing_b:
        raise InputError(f"Sample ID sets differ: missing from A={missing_a[:10]} (n={len(missing_a)}), "
                         f"missing from B={missing_b[:10]} (n={len(missing_b)})")
    ids = sorted(a)
    if not ids:
        raise InputError("No paired samples")
    mismatched = [sid for sid in ids if a[sid]["reference"] != b[sid]["reference"]]
    if mismatched:
        raise InputError(f"Reference mismatch for {mismatched[:10]} (n={len(mismatched)}); strings must match exactly")
    return ids


def sample_order_metadata(ids):
    encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"n_samples": len(ids), "ordering": "Unicode lexicographic ID order",
            "ordered_ids_sha256": hashlib.sha256(encoded).hexdigest()}


def bleu_statistics(module, hypotheses, references):
    direct = module.raw_corpus_bleu(hypotheses, [references], smooth_value=0.0)
    rows = []
    for hypothesis, reference in zip(hypotheses, references, strict=True):
        value = module.raw_corpus_bleu([hypothesis], [[reference]], smooth_value=0.0)
        rows.append([*value.counts, *value.totals, value.sys_len, value.ref_len])
    stats = np.asarray(rows, dtype=np.int64)
    point = float(direct.scores[3])
    aggregate = score_bleu_statistics(module, stats.sum(axis=0))
    if aggregate != point:
        raise RuntimeError(f"Corpus-statistics aggregation differs from direct raw_corpus_bleu: {aggregate} vs {point}")
    return point, stats, {"bleu1_to_4": [float(v) for v in direct.scores],
        "counts": [int(v) for v in direct.counts], "totals": [int(v) for v in direct.totals],
        "hypothesis_length": int(direct.sys_len), "reference_length": int(direct.ref_len)}


def score_bleu_statistics(module, stats):
    values = [int(v) for v in stats]
    value = module.compute_bleu(values[:4], values[4:8], values[8], values[9],
        smooth_method="floor", smooth_value=0.0, use_effective_order=True)
    return float(value.scores[3])


def paired_bootstrap(stats_a, stats_b, score, resamples=2000, seed=20260910, confidence=0.95):
    if resamples < 1 or not 0 < confidence < 1 or seed < 0:
        raise InputError("resamples must be positive, seed nonnegative, and confidence between0 and1")
    a, b = np.asarray(stats_a), np.asarray(stats_b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] == 0:
        raise InputError("Paired statistics must have the same nonempty two-dimensional shape")
    rng = np.random.Generator(np.random.PCG64(seed))
    results = np.empty((resamples, 3), dtype=np.float64)
    indices_hash = hashlib.sha256()
    n = a.shape[0]
    # Shared indices are used for both systems, one whole sample at a time.
    # Chunking bounds memory; corpus counts are summed before BLEU is computed.
    for start in range(0, resamples, 128):
        count = min(128, resamples - start)
        indices = rng.integers(0, n, size=(count, n), dtype=np.int64)
        indices_hash.update(indices.astype("<i8", copy=False).tobytes())
        aggregate_a = a[indices].sum(axis=1)
        aggregate_b = b[indices].sum(axis=1)
        for offset, (sa, sb) in enumerate(zip(aggregate_a, aggregate_b, strict=True)):
            va, vb = float(score(sa)), float(score(sb))
            results[start + offset] = (va, vb, vb - va)
    if not np.isfinite(results).all():
        raise InputError("Nonfinite bootstrap score")
    alpha = (1.0 - confidence) / 2
    lo, hi = np.quantile(results[:, 2], [alpha, 1 - alpha], method="linear")
    return {"resamples": int(resamples), "seed": int(seed), "rng": "NumPy PCG64",
        "unit": "whole sample", "paired": True, "draws_per_replicate": int(n),
        "sampled_indices_sha256": indices_hash.hexdigest(),
        "confidence": float(confidence), "interval_method": "percentile, NumPy quantile method=linear",
        "difference_direction": "B_minus_A",
        "ci": {"lower": float(lo), "upper": float(hi), "contains_zero": bool(lo <= 0 <= hi)}}, results


def compare_bleu_rows(rows_a, rows_b, module, resamples=2000, seed=20260910, confidence=0.95):
    ids = align_rows(rows_a, rows_b)
    refs = [rows_a[sid]["reference"] for sid in ids]
    ha = [rows_a[sid]["prediction"] for sid in ids]
    hb = [rows_b[sid]["prediction"] for sid in ids]
    pa, sa, detail_a = bleu_statistics(module, ha, refs)
    pb, sb, detail_b = bleu_statistics(module, hb, refs)
    bootstrap, replicates = paired_bootstrap(sa, sb, lambda s: score_bleu_statistics(module, s),
        resamples=resamples, seed=seed, confidence=confidence)
    result = {"alignment": sample_order_metadata(ids),
        "point_estimate": {"A": pa, "B": pb, "B_minus_A": pb - pa},
        "point_details": {"A": detail_a, "B": detail_b}, "bootstrap": bootstrap}
    return result, replicates


def parse_where(expressions):
    result = {}
    for expression in expressions or []:
        if "=" not in expression:
            raise InputError(f"Filter must be column=value: {expression}")
        key, value = expression.split("=", 1)
        if not key or key in result:
            raise InputError(f"Invalid/duplicate filter: {expression}")
        result[key] = value
    return result


def read_mean_csv(path, column, id_key="sample_id", reference_key="input_text", filters=None):
    filters = filters or {}
    rows, raw_count = {}, 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {column, id_key, reference_key, *filters}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise InputError(f"Missing/duplicate CSV header in {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise InputError(f"Missing CSV columns in {path}: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            raw_count += 1
            if any(row[key] != value for key, value in filters.items()):
                continue
            sid = row[id_key]
            if not isinstance(sid, str) or not sid.strip():
                raise InputError(f"{path}:{line_number}: missing ID")
            if sid in rows:
                raise InputError(f"{path}:{line_number}: duplicate selected ID {sid!r}; select one group/decode_path explicitly")
            if row[reference_key] is None:
                raise InputError(f"{path}:{line_number}: missing reference text")
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise InputError(f"{path}:{line_number}: invalid value for {column}") from exc
            if not math.isfinite(value):
                raise InputError(f"{path}:{line_number}: nonfinite {column}; no pairs are silently dropped")
            rows[sid] = {"reference": row[reference_key], "value": value, "fields": row}
    if not rows:
        raise InputError(f"No rows selected from {path} with {filters}")
    return rows, {"raw_rows": raw_count, "selected_rows": len(rows), "filters": filters,
                  "column": column, "id_key": id_key, "reference_key": reference_key}


def compare_mean_rows(rows_a, rows_b, resamples=2000, seed=20260910, confidence=0.95):
    ids = align_rows(rows_a, rows_b)
    checked = []
    for name in ("input_frames", "common_frames", "joint_count"):
        presence = [name in row.get("fields", {}) for rows in (rows_a, rows_b) for row in rows.values()]
        if any(presence):
            if not all(presence):
                raise InputError(f"Reconstruction coverage column missing on one side: {name}")
            mismatches = [sid for sid in ids if rows_a[sid]["fields"][name] != rows_b[sid]["fields"][name]]
            if mismatches:
                raise InputError(f"Reconstruction coverage differs for {name}: {mismatches[:10]}")
            checked.append(name)
    va = np.asarray([rows_a[sid]["value"] for sid in ids], dtype=np.float64)
    vb = np.asarray([rows_b[sid]["value"] for sid in ids], dtype=np.float64)
    pa, pb = float(va.mean()), float(vb.mean())
    bootstrap, replicates = paired_bootstrap(va[:, None], vb[:, None], lambda total: total[0] / len(ids),
        resamples=resamples, seed=seed, confidence=confidence)
    return {"alignment": {**sample_order_metadata(ids), "coverage_columns_equal": checked},
            "point_estimate": {"A": pa, "B": pb, "B_minus_A": pb - pa},
            "bootstrap": bootstrap}, replicates


def json_pointer(value, pointer):
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise InputError(f"Expected JSON Pointer starting with '/': {pointer!r}")
    try:
        for token in pointer[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, list):
                if not token.isdigit():
                    raise KeyError(token)
                value = value[int(token)]
            else:
                value = value[token]
        return value
    except (KeyError, IndexError, TypeError) as exc:
        raise InputError(f"JSON Pointer not found: {pointer}") from exc


def known_seed(summary):
    candidates = []
    for pointer in ("/training_seed", "/checkpoint_args/seed", "/args/seed", "/ar_args/seed"):
        try:
            value = json_pointer(summary, pointer)
        except InputError:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            candidates.append(value)
    if len(set(candidates)) > 1:
        raise InputError(f"Conflicting training seeds inside one source: {candidates}")
    return candidates[0] if candidates else None


def extract_run_value(summary, row):
    node = summary
    if "list_pointer" in row:
        node = json_pointer(summary, row["list_pointer"])
        if not isinstance(node, list) or not row.get("where"):
            raise InputError("list_pointer requires an array and an explicit where selector")
        selected = [item for item in node if isinstance(item, dict)
                    and all(item.get(k) == v for k, v in row["where"].items())]
        if len(selected) != 1:
            raise InputError(f"Run summary selector must match exactly once, got {len(selected)}")
        node = selected[0]
    if "value_pointer" not in row:
        raise InputError("Every run requires a value_pointer")
    value = json_pointer(node, row["value_pointer"])
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError("Selected run value must be a finite JSON number")
    return float(value)


def training_seed_summary(input_path):
    input_path = Path(input_path).resolve()
    specification = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if specification.get("schema") != "reviewer55.training_seed_inputs.v1":
        raise InputError("Expected schema reviewer55.training_seed_inputs.v1")
    context = specification.get("context", {})
    if any(not isinstance(context.get(k), str) or not context[k] for k in ("dataset", "split", "metric")):
        raise InputError("Run manifest context requires dataset, split and metric")
    runs = specification.get("runs")
    if not isinstance(runs, list) or not runs:
        raise InputError("Run manifest must contain nonempty runs")
    grouped, sources = {}, []
    for row in runs:
        system, seed = row.get("system"), row.get("training_seed")
        if not isinstance(system, str) or not system or not isinstance(seed, int) or isinstance(seed, bool):
            raise InputError("Each run requires system and integer training_seed")
        if seed in grouped.setdefault(system, {}):
            raise InputError(f"Duplicate training run: {system}, seed={seed}")
        path = (input_path.parent / row["summary_path"]).resolve()
        summary = json.loads(path.read_text(encoding="utf-8-sig"))
        for key in ("dataset", "split"):
            if isinstance(summary, dict) and key in summary and summary[key] != context[key]:
                raise InputError(f"Source {key} mismatch in {path}: {summary[key]!r} != {context[key]!r}")
        observed_seed = known_seed(summary)
        record = {"system": system, "training_seed": seed, "summary": source_record(path),
                  "selector": {k:row[k] for k in ("list_pointer", "where", "value_pointer") if k in row}}
        if "training_manifest_path" in row:
            training_path = (input_path.parent / row["training_manifest_path"]).resolve()
            training = json.loads(training_path.read_text(encoding="utf-8-sig"))
            external_seed = json_pointer(training, row["training_seed_pointer"]) if "training_seed_pointer" in row else known_seed(training)
            if external_seed is None:
                raise InputError(f"No training seed found in supplied training manifest: {training_path}")
            if observed_seed is not None and external_seed != observed_seed:
                raise InputError("Summary and training manifest seeds disagree")
            observed_seed = external_seed
            record["training_manifest"] = source_record(training_path)
        if observed_seed is not None and observed_seed != seed:
            raise InputError(f"Declared seed={seed} disagrees with source seed={observed_seed}: {path}")
        record["seed_verification"] = "source_metadata" if observed_seed is not None else "declared_only_no_seed_in_metric_file"
        value = extract_run_value(summary, row)
        grouped[system][seed] = value
        record["value"] = value
        sources.append(record)
    expected = specification.get("expected_training_seeds")
    if expected is not None:
        if not isinstance(expected, list) or not expected or any(not isinstance(s,int) or isinstance(s,bool) for s in expected) or len(set(expected)) != len(expected):
            raise InputError("expected_training_seeds must contain unique integers")
        for system, values in grouped.items():
            if set(values) != set(expected):
                raise InputError(f"Incomplete/unexpected seeds for {system}: got {sorted(values)}, expected {sorted(expected)}")
    summaries = {}
    for system, values in grouped.items():
        ordered = [values[s] for s in sorted(values)]
        summaries[system] = {"n_training_runs": len(ordered), "training_seeds": sorted(values),
            "values": ordered, "mean": statistics.mean(ordered),
            "sample_std_ddof1": statistics.stdev(ordered) if len(ordered) > 1 else None}
    paired = []
    for comparison in specification.get("comparisons", []):
        a, b = comparison["a"], comparison["b"]
        if a not in grouped or b not in grouped:
            raise InputError(f"Unknown comparison systems: {a}, {b}")
        if set(grouped[a]) != set(grouped[b]):
            raise InputError(f"Training seed sets differ for paired comparison {a}/{b}")
        seeds = sorted(grouped[a])
        delta = [grouped[b][s] - grouped[a][s] for s in seeds]
        paired.append({"A":a, "B":b, "difference_direction":"B_minus_A", "training_seeds":seeds,
            "per_seed_differences":delta, "mean_difference":statistics.mean(delta),
            "sample_std_ddof1":statistics.stdev(delta) if len(delta)>1 else None})
    return {"context": context, "run_input_manifest": source_record(input_path),
            "expected_training_seeds": expected, "systems": summaries, "paired_seed_comparisons":paired,
            "source_runs":sources, "uncertainty":{
                "type":"variation across the listed training runs",
                "no_bootstrap_substitution":True,
                "note":"Sample standard deviation uses ddof=1. One run has no estimable training-seed standard deviation (null). This does not replace paired sample bootstrap."}}


def protocol_record(args):
    if args.protocol:
        path = Path(args.protocol).resolve()
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return {"artifact":source_record(path), "id":value.get("version",value.get("schema")),
            "binding":"caller-declared input protocol; bare JSONL/CSV values do not independently prove upstream run compliance"}
    return {"artifact":None, "id":args.protocol_id or "unspecified_input_protocol",
            "binding":"caller declaration only; no protocol file supplied"}


def base_report(args):
    return {"schema":SCHEMA, "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "command":sys.argv, "mode":args.mode, "purpose":args.purpose,
            "scope":{"dataset":getattr(args,"dataset",None),"split":getattr(args,"split",None)},
            "protocol":protocol_record(args), "implementation":source_record(Path(__file__)),
            "runtime":{"python":sys.version,"numpy":np.__version__,"device":"cpu","torch_imported":False}}


def write_output(path, value):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite statistics output: {path}")
    if path.is_relative_to((BASE.parent / "runtime").resolve()):
        raise InputError("Statistics output must be outside read-only repository trees")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode",required=True)
    for name in ("bleu","mean","runs"):
        ap = modes.add_parser(name)
        ap.add_argument("--out",type=Path,required=True)
        protocol = ap.add_mutually_exclusive_group()
        protocol.add_argument("--protocol",type=Path)
        protocol.add_argument("--protocol-id")
        ap.add_argument("--purpose",default="analysis")
        if name=="runs":
            ap.add_argument("--input",type=Path,required=True)
            continue
        ap.add_argument("--a",type=Path,required=True)
        ap.add_argument("--b",type=Path,required=True)
        ap.add_argument("--label-a",default="A")
        ap.add_argument("--label-b",default="B")
        ap.add_argument("--dataset")
        ap.add_argument("--split")
        ap.add_argument("--seed",type=int,default=20260910)
        ap.add_argument("--resamples",type=int,default=2000)
        ap.add_argument("--confidence",type=float,default=0.95)
        ap.add_argument("--id-key",default="id" if name=="bleu" else "sample_id")
        ap.add_argument("--reference-key",default="reference" if name=="bleu" else "input_text")
        if name=="bleu":
            ap.add_argument("--prediction-key",default="prediction")
            ap.add_argument("--metric-source",type=Path,default=DEFAULT_METRIC_SOURCE)
        else:
            ap.add_argument("--column",required=True)
            ap.add_argument("--where-a",action="append",default=[])
            ap.add_argument("--where-b",action="append",default=[])
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite statistics output: {args.out}")
    report = base_report(args)
    if args.mode=="runs":
        run_result=training_seed_summary(args.input)
        report.update(run_result)
        report["scope"]={"dataset":run_result["context"]["dataset"],"split":run_result["context"]["split"]}
    else:
        report["inputs"]={"A":{"label":args.label_a,**source_record(args.a)},
                          "B":{"label":args.label_b,**source_record(args.b)}}
        if args.mode=="bleu":
            module=load_metric(args.metric_source)
            a=read_jsonl(args.a,args.id_key,args.reference_key,args.prediction_key)
            b=read_jsonl(args.b,args.id_key,args.reference_key,args.prediction_key)
            result,reps=compare_bleu_rows(a,b,module,args.resamples,args.seed,args.confidence)
            report["metric"]={"name":"corpus BLEU-4","direction":"higher_is_better",
                "source":source_record(args.metric_source),"vendored_version":getattr(module,"VERSION",None),
                "point_estimate_api":"raw_corpus_bleu(sys_stream, [ref_stream], smooth_value=0.0)",
                "tokenization":"none (already processed input)","lowercase":False,
                "smoothing":"floor=0","use_effective_order":True,
                "bootstrap_computation":"sum resampled per-sample clipped n-gram counts, totals and corpus lengths, then call the same compute_bleu; never average sentence BLEU",
                "input_strings_modified":False}
        else:
            a,selection_a=read_mean_csv(args.a,args.column,args.id_key,args.reference_key,parse_where(args.where_a))
            b,selection_b=read_mean_csv(args.b,args.column,args.id_key,args.reference_key,parse_where(args.where_b))
            result,reps=compare_mean_rows(a,b,args.resamples,args.seed,args.confidence)
            report["metric"]={"name":args.column,"aggregation":"unweighted arithmetic mean of paired sample values",
                               "difference_direction":"B_minus_A","direction":"interpret using the selected metric; reconstruction error is lower-is-better"}
            report["selection"]={"A":selection_a,"B":selection_b}
        report.update(result)
        report["bootstrap"]["replicate_scores_sha256"]=hashlib.sha256(reps.astype("<f8",copy=False).tobytes()).hexdigest()
        report["uncertainty"]={"type":"paired sample uncertainty conditional on these fixed runs/predictions",
            "training_seed_variance_estimated":False,
            "note":"This percentile interval resamples complete paired examples. It does not capture training-seed variation, generation-seed variation, evaluator uncertainty, or replace independent training repeats."}
    write_output(args.out,report)
    print(json.dumps({"output":str(args.out.resolve()),"mode":args.mode,
        "point_estimate":report.get("point_estimate"),"ci":report.get("bootstrap",{}).get("ci"),
        "systems":report.get("systems")},ensure_ascii=False,allow_nan=False))
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (InputError,FileExistsError,FileNotFoundError) as exc:
        print(f"[input error] {exc}",file=sys.stderr)
        raise SystemExit(2)
