"""Traceable generation for the revised reviewer55 protocol.

Reference backend: serial AR sampling through the revised model.sample API, with
an independent per-sample RNG scope and an LRU cache of text representations.
--batch-size controls work chunks only; effective AR/text batch size is one.
This deliberately avoids changing the sampler algorithm or claiming batch-AR
speedups. No back-translation evaluator is run by this entry point.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
import datetime as dt
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any

sys.dont_write_bytecode = True
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
LAYOUT_VERSION = "reviewer55_verified_v1"
VARIANTS = ("baseline", "pamvq", "tlrvq", "pamrvq", "pamrvq_random")
DECODE = {"temperature": 0.9, "top_k": 20, "top_p": 1.0,
          "rep_penalty": 1.5, "rep_window": 8, "max_run": 4}


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, (SimpleNamespace, argparse.Namespace)):
        return jsonable(vars(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
    temp.replace(path)


def prepare_imports(repo: Path) -> Path:
    src = (repo / "code" / "src").resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    sys.path.insert(0, str(src))
    return src


def checkpoint_args(checkpoint: dict) -> SimpleNamespace:
    value = checkpoint.get("args")
    if isinstance(value, dict):
        return SimpleNamespace(**value)
    if isinstance(value, (argparse.Namespace, SimpleNamespace)):
        return SimpleNamespace(**vars(value))
    raise ValueError("Checkpoint must contain an args dictionary/namespace")


def sample_seed(base_seed: int, dataset: str, split: str, sid: str) -> int:
    # JSON framing prevents ambiguous concatenation. Do not add model name or
    # training seed: controlled comparisons share the same sample RNG stream.
    key = json.dumps(["reviewer55.motion_rng.v1", int(base_seed), dataset, split, sid],
                     ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def independent_torch_rng(seed: int, device: torch.device):
    device = torch.device(device)
    devices = []
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        devices = [index]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        yield


@dataclass(frozen=True)
class TokenLayout:
    names: tuple[str, ...]
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]

    @classmethod
    def create(cls, names, sizes):
        names, sizes = tuple(names), tuple(int(x) for x in sizes)
        if not names or len(names) != len(sizes) or len(set(names)) != len(names):
            raise ValueError("Invalid token-group names")
        if min(sizes) < 1:
            raise ValueError("Every token group must have at least one codeword")
        offsets, cursor = [], 0
        for size in sizes:
            offsets.append(cursor)
            cursor += size
        return cls(names, sizes, tuple(offsets))

    @property
    def group_size(self):
        return len(self.names)

    @property
    def vocabulary_size(self):
        return sum(self.sizes)

    def ranges(self, length):
        return [(self.offsets[i % self.group_size],
                 self.offsets[i % self.group_size] + self.sizes[i % self.group_size])
                for i in range(length)]

    def validate(self, tokens):
        values = [int(t) for t in tokens]
        if not values or len(values) % self.group_size:
            raise ValueError("Generated motion must contain nonempty complete token groups")
        for i, value in enumerate(values):
            j = i % self.group_size
            if not self.offsets[j] <= value < self.offsets[j] + self.sizes[j]:
                raise ValueError(f"Illegal token at position {i}: {value}, group={self.names[j]}")
        return values


def layout_from_model(variant: str, model, args) -> TokenLayout:
    if variant == "baseline":
        return TokenLayout.create(["global"], [args.nb_code])
    if variant == "tlrvq":
        return TokenLayout.create(["base", "res"], [args.nb_code, args.nb_code_residual])
    # Never consult KP_SPLITS or infer token order from source coordinate order.
    if tuple(model.stream_order) != ("body", "hand", "face"):
        raise ValueError(f"Unexpected revised stream_order: {model.stream_order}")
    names = tuple(model.token_order)
    if variant == "pamvq":
        expected = tuple(model.stream_order)
        if names != expected:
            raise ValueError(f"Single-level token order mismatch: {names}")
        sizes = [model.stream_codes[n] for n in names]
    else:
        expected = tuple(f"{s}_{layer}" for s in model.stream_order for layer in ("base", "res"))
        if names != expected:
            raise ValueError(f"Residual token order mismatch: {names}")
        sizes = []
        for name in names:
            stream, level = name.rsplit("_", 1)
            codes = model.stream_codes_base if level == "base" else model.stream_codes_residual
            sizes.append(codes[stream])
    return TokenLayout.create(names, sizes)


def resolve_stats(checkpoint_path: Path):
    p = checkpoint_path
    means = [p.parent / "stats" / f"{p.stem}_mean.npy",
             p.parent / f"{p.stem}_mean.npy", p.parent / "mean.npy"]
    for mean_path in means:
        std_path = mean_path.with_name(mean_path.name.replace("mean", "std"))
        if mean_path.is_file() and std_path.is_file():
            mean = np.load(mean_path, allow_pickle=False).reshape(-1).astype(np.float32)
            std = np.load(std_path, allow_pickle=False).reshape(-1).astype(np.float32)
            if mean.shape != (534,) or std.shape != (534,):
                raise ValueError(f"Expected 534 coordinate statistics, got {mean.shape}/{std.shape}")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
                raise ValueError("Invalid normalization statistics")
            return mean, std, {"mean": artifact(mean_path), "std": artifact(std_path)}
    raise FileNotFoundError(f"No mean.npy/std.npy pair found for {checkpoint_path}")


def load_vq(path: Path, variant: str, dataset: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint_args(checkpoint)
    if args.dataname != f"{dataset}_lift3d":
        raise ValueError(f"VQ dataset mismatch: {args.dataname}")
    common = dict(code_dim=args.code_dim, output_emb_width=args.output_emb_width,
                  down_t=args.down_t, stride_t=args.stride_t, width=args.width,
                  depth=args.depth, dilation_growth_rate=args.dilation_growth_rate,
                  activation=args.vq_act, norm=args.vq_norm)
    if variant == "baseline":
        from models.vqvae import VQVAE_251
        model = VQVAE_251(args, nb_code=args.nb_code, **common)
    elif variant == "tlrvq":
        from models.vqvae_residual import ResidualVQVAE
        model = ResidualVQVAE(args, nb_code=args.nb_code,
                             nb_code_residual=args.nb_code_residual, **common)
    else:
        if getattr(args, "layout_version", None) != LAYOUT_VERSION:
            raise ValueError("Refusing legacy multi-stream checkpoint: verified layout_version required")
        expected_grouping = "random" if variant == "pamrvq_random" else "anatomical"
        if getattr(args, "grouping", None) != expected_grouping:
            raise ValueError(f"Expected grouping={expected_grouping}, got {getattr(args, 'grouping', None)}")
        streams = ("body", "hand", "face")
        if variant == "pamvq":
            from models.vqvae_multistream import MultiStreamVQVAE
            codes = {n: getattr(args, f"nb_code_{n}", None) or args.nb_code for n in streams}
            model = MultiStreamVQVAE(args, dataset_name=args.dataname,
                                     nb_code=args.nb_code, stream_codes=codes, **common)
        else:
            from models.vqvae_multistream_residual import MultiStreamResidualVQVAE
            base_codes = {n: getattr(args, f"nb_base_{n}") for n in streams}
            residual_codes = {n: getattr(args, f"nb_res_{n}") for n in streams}
            model = MultiStreamResidualVQVAE(args, dataset_name=args.dataname,
                stream_codes_base=base_codes, stream_codes_residual=residual_codes, **common)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    layout = layout_from_model(variant, model, args)
    factor = int(args.stride_t) ** int(args.down_t)
    if factor != (4 if dataset == "phix" else 2):
        raise ValueError(f"Revised dataset downsampling mismatch: {factor}")
    mean, std, stats_info = resolve_stats(path)
    return model, args, layout, factor, mean, std, stats_info


@dataclass(frozen=True)
class MotionBudget:
    requested_pose_frames: int
    downsampling_factor: int
    group_size: int
    position_capacity: int
    requested_latent_steps: int
    capacity_latent_steps: int
    latent_steps: int
    max_motion_tokens: int
    effective_pose_frames: int
    limit_reason: str


def motion_budget(max_pose_frames, factor, group_size, position_capacity):
    max_pose_frames, factor, group_size, position_capacity = map(
        int, (max_pose_frames, factor, group_size, position_capacity))
    if min(max_pose_frames, factor, group_size, position_capacity) < 1:
        raise ValueError("Frame, stride, group and position budgets must be positive")
    requested = max_pose_frames // factor
    capacity = (position_capacity - 1) // group_size
    steps = min(requested, capacity)
    if steps < 1:
        raise ValueError("No capacity for one complete nonempty motion group")
    return MotionBudget(max_pose_frames, factor, group_size, position_capacity,
        requested, capacity, steps, steps * group_size, steps * factor,
        "position_capacity" if capacity < requested else "pose_frame_cap")


def validate_ar_binding(args, layout, vq_hash, protocol_hash, max_pose_frames, dataset, protocol):
    expected = {"vq_sha256": vq_hash, "protocol_sha256": protocol_hash,
                "max_pose_frames": max_pose_frames, "token_group_size": layout.group_size}
    for name, value in expected.items():
        if not hasattr(args, name) or getattr(args, name) != value:
            raise ValueError(f"AR checkpoint binding mismatch: {name}={getattr(args, name, None)!r}, expected {value!r}")
    if list(getattr(args, "token_order", [])) != list(layout.names):
        raise ValueError(f"AR/VQ token_order mismatch: {getattr(args, 'token_order', None)} vs {layout.names}")
    if args.dataname != f"{dataset}_lift3d" or args.num_vq != layout.vocabulary_size:
        raise ValueError("AR dataset or vocabulary does not match VQ")
    expected_capacity = protocol["second_stage"]["position_capacity"][dataset]
    if args.block_size != expected_capacity:
        raise ValueError(f"AR position capacity {args.block_size} != fixed protocol {expected_capacity}")
    expected_layers = protocol["second_stage"].get("ar_layers", {}).get(dataset)
    if expected_layers is not None and getattr(args, "num_layers", None) != expected_layers:
        raise ValueError(f"AR layer count does not match fixed protocol: {getattr(args, 'num_layers', None)}")
    expected_text = "mbart" if dataset == "phix" else "char"
    if getattr(args, "text_encoder", None) != expected_text:
        raise ValueError(f"Expected {expected_text} text encoder for {dataset}")
    if getattr(args, "gloss_supervised", 0):
        raise ValueError("Refusing gloss-supervised AR checkpoint")


def load_ar(path, layout, vq_hash, protocol_hash, max_pose_frames, dataset, protocol, device):
    from models.t2m_trans_cross import CrossAttnText2MotionTransformer
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint_args(checkpoint)
    validate_ar_binding(args, layout, vq_hash, protocol_hash, max_pose_frames, dataset, protocol)
    model = CrossAttnText2MotionTransformer(num_vq=args.num_vq, text_dim=args.text_dim,
        embed_dim=args.embed_dim, block_size=args.block_size, num_layers=args.num_layers,
        n_head=args.n_head, drop_out_rate=0.0, fc_rate=args.fc_rate,
        align_dim=getattr(args, "align_dim", 0), predict_length=bool(getattr(args, "predict_length", False)))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    if "token_group_size" not in inspect.signature(model.sample).parameters:
        raise ValueError("Loaded AR model lacks the revised complete-group sample API")
    return model, args, checkpoint


class TextMemoryCache:
    """Serial text encoding avoids batch-padding/numerical changes in this backend."""
    def __init__(self, tokenizer, encoder, kind, device, max_length, capacity=128):
        self.tokenizer, self.encoder, self.kind = tokenizer, encoder, kind
        self.device, self.max_length, self.capacity = device, max_length, capacity
        self.values = OrderedDict()
        self.hits = self.misses = 0

    @torch.inference_mode()
    def get(self, text):
        if text in self.values:
            self.hits += 1
            mem, mask = self.values.pop(text)
            self.values[text] = (mem, mask)
        else:
            self.misses += 1
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
            ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
            encoded = self.encoder(ids, am) if self.kind == "char" else self.encoder(
                input_ids=ids, attention_mask=am).last_hidden_state
            mem, mask = encoded.detach().cpu(), am.detach().cpu()
            if self.capacity:
                self.values[text] = (mem, mask)
                if len(self.values) > self.capacity:
                    self.values.popitem(last=False)
        return mem.to(self.device), mask.to(self.device)


def resolve_character_vocabulary(args, repo, override=None, expected_sha256=None):
    """Relocate the same vocabulary; never rebuild token IDs from evaluation text."""
    recorded = getattr(args, "char_vocab_path", None)
    recorded_sha = getattr(args, "char_vocab_sha256", None)
    if recorded_sha and expected_sha256 and recorded_sha != expected_sha256:
        raise ValueError("Supplied vocabulary SHA differs from the training checkpoint")
    expected = recorded_sha or expected_sha256
    if override is not None and not expected:
        raise ValueError("An old checkpoint vocabulary override requires --char-vocab-sha256 from its training inputs")
    path = Path(override) if override is not None else Path(recorded) if recorded else Path(repo) / "data/csl/char_vocab/txt.vocab"
    if not path.is_file():
        raise FileNotFoundError(f"Character vocabulary missing: {path}; provide its original file with --char-vocab-path and matching SHA")
    if expected and sha256_file(path) != expected:
        raise ValueError("Character vocabulary content differs from its recorded SHA")
    return path


def load_text_encoder(args, checkpoint, repo, dataset, device, char_vocab_path=None, char_vocab_sha256=None):
    lang = "de_DE" if dataset == "phix" else "zh_CN"
    kind = args.text_encoder
    if kind == "char":
        from models.text_encoder_char import CharTokenizer, CharTextEncoder
        vocab_path = resolve_character_vocabulary(args, repo, char_vocab_path, char_vocab_sha256)
        tokenizer = CharTokenizer(vocab_path, src_lang=lang)
        dim = getattr(args, "char_enc_dim", 0) or args.embed_dim
        encoder = CharTextEncoder(vocab_size=len(tokenizer), embed_dim=dim,
            num_layers=getattr(args, "char_enc_layers", 2),
            n_head=getattr(args, "char_enc_heads", 8), drop_out_rate=0.0,
            max_len=getattr(args, "max_text_len", 128) + 8, pad_id=tokenizer.pad_id)
        if "text_enc" not in checkpoint:
            raise ValueError("Character encoder weights missing from AR checkpoint")
        encoder.load_state_dict(checkpoint["text_enc"], strict=True)
        encoder.to(device).eval()
        meta = {"kind": kind, "lang": lang, "vocabulary": artifact(vocab_path),
                "weight_binding": "AR checkpoint text_enc state", "dim": dim}
    else:
        from transformers import MBart50TokenizerFast, MBartModel
        name = args.mbart_name
        revision = getattr(args, "mbart_revision", None)
        kwargs = {"local_files_only": True}
        if revision:
            kwargs["revision"] = revision
        tokenizer = MBart50TokenizerFast.from_pretrained(name, src_lang=lang, **kwargs)
        full_model = MBartModel.from_pretrained(name, **kwargs)
        resolved_revision = getattr(full_model.config, "_commit_hash", None)
        config_text = full_model.config.to_json_string()
        encoder = full_model.encoder.to(device).eval()
        del full_model
        meta = {"kind": kind, "lang": lang, "pretrained_identifier": name,
                "requested_revision": revision, "resolved_revision": resolved_revision,
                "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
                "local_files_only": True}
    return tokenizer, encoder, meta


@torch.inference_mode()
def generate_sample_tokens(model, memory, attention_mask, layout, budget, seed):
    device = memory.device
    if memory.size(0) != 1 or attention_mask.size(0) != 1:
        raise ValueError("Reference sampler is serial: expected batch size one")
    if model.num_vq != layout.vocabulary_size:
        raise ValueError("Sampler vocabulary mismatch")
    with independent_torch_rng(seed, device):
        result = model.sample(memory, attention_mask, max_len=budget.max_motion_tokens,
            temperature=DECODE["temperature"], top_k=DECODE["top_k"], top_p=DECODE["top_p"],
            stream_ranges=layout.ranges(budget.max_motion_tokens),
            rep_penalty=DECODE["rep_penalty"], rep_window=DECODE["rep_window"],
            max_run=DECODE["max_run"], rep_streams=layout.group_size,
            token_group_size=layout.group_size, min_len=layout.group_size)
    if not isinstance(result, torch.Tensor) or result.ndim != 2 or result.size(0) != 1:
        raise ValueError("Unexpected revised sample return type/shape")
    tokens = layout.validate(result[0].detach().cpu().tolist())
    if len(tokens) > budget.max_motion_tokens:
        raise ValueError("Sampler exceeded explicit motion budget")
    # The revised model strips EOS and has only two normal stop conditions:
    # sampled EOS (strictly fewer motion tokens) or exhausted max_len.
    end_reason = "eos" if len(tokens) < budget.max_motion_tokens else budget.limit_reason
    return tokens, end_reason


@torch.inference_mode()
def decode_tokens(vq, variant, tokens, layout, mean, std, device, factor):
    tokens = layout.validate(tokens)
    flat = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    if variant == "baseline":
        motion = vq.forward_decoder(flat)
    else:
        parts = {name: flat[:, i::layout.group_size] - layout.offsets[i]
                 for i, name in enumerate(layout.names)}
        motion = vq.forward_decoder(parts["base"], parts["res"]) if variant == "tlrvq" else vq.forward_decoder(parts)
    expected_frames = len(tokens) // layout.group_size * factor
    if motion.shape != (1, expected_frames, 534):
        raise ValueError(f"Decoded shape {tuple(motion.shape)} != expected {(1, expected_frames, 534)}")
    result = (motion[0].detach().float().cpu().numpy() * std + mean).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Decoded pose contains nonfinite coordinates")
    # The revised model already returns source order. No anatomical reordering here.
    return result


def work_chunks(items, size):
    if size < 1:
        raise ValueError("batch-size/work chunk size must be positive")
    for offset in range(0, len(items), size):
        yield items[offset:offset + size]


def write_slt(path, values):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as z:
            pickle.dump(values, z, protocol=4)
    temp.replace(path)


def source_inventory(src):
    files = sorted(p for p in src.rglob("*.py") if "__pycache__" not in p.parts)
    return {str(p.relative_to(src)): sha256_file(p) for p in files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("phix", "csl"), required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--vq-ckpt", type=Path, required=True)
    ap.add_argument("--ar-ckpt", type=Path, required=True)
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pose-frames", type=int, default=None,
                    help="Must match the fixed protocol and AR checkpoint; default: protocol value")
    ap.add_argument("--generation-seed", type=int, default=20260910)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Work chunk size only. Reference backend AR/text sampling remains serial.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repo", type=Path, default=BASE / "runtime")
    ap.add_argument("--protocol", type=Path, default=BASE / "configs/protocol.json")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("SLP_SIGN_DATA_DIR", str(BASE / "runtime/data"))))
    ap.add_argument("--char-vocab-path", type=Path, help="Original character vocabulary at a relocated path")
    ap.add_argument("--char-vocab-sha256", help="Required when relocating an old checkpoint vocabulary without a recorded hash")
    ap.add_argument("--text-cache-size", type=int, default=128)
    args = ap.parse_args(argv)
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    args.vq_ckpt, args.ar_ckpt, args.protocol = args.vq_ckpt.resolve(), args.ar_ckpt.resolve(), args.protocol.resolve()
    if args.out.is_relative_to(args.repo):
        raise ValueError("Generation outputs must be outside the repository/input tree")
    if args.batch_size < 1 or args.text_cache_size < 0:
        raise ValueError("Invalid work chunk/cache size")
    splits = [s.strip() for s in args.splits.split(",")]
    if not splits or len(set(splits)) != len(splits) or any(s not in ("dev", "test") for s in splits):
        raise ValueError("splits must be a unique selection of dev,test")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("version") != LAYOUT_VERSION:
        raise ValueError("Unexpected protocol version")
    for name in ("temperature", "top_k", "top_p"):
        if protocol.get("decoding", {}).get(name, DECODE[name]) != DECODE[name]:
            raise ValueError(f"Fixed sampler parameter differs from protocol: {name}")
    if protocol.get("decoding", {}).get("max_same_group_run", DECODE["max_run"]) != DECODE["max_run"]:
        raise ValueError("Fixed maximum same-group run differs from protocol")
    fixed_frames = int(protocol["second_stage"]["max_pose_frames"])
    if args.max_pose_frames is None:
        args.max_pose_frames = fixed_frames
    if args.max_pose_frames != fixed_frames or args.generation_seed != protocol["generation_seed"]:
        raise ValueError("Frame cap/generation seed must match the predeclared protocol")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty generation run: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    src = prepare_imports(args.repo)
    manifest_path = args.out / "generation_manifest.json"
    manifest = {"schema": "reviewer55.generation.v1", "status": "running", "started_at": timestamp(),
        "arguments": jsonable(args), "protocol": artifact(args.protocol),
        "vq_checkpoint": artifact(args.vq_ckpt), "ar_checkpoint": artifact(args.ar_ckpt),
        "source_sha256": source_inventory(src), "entry_point": artifact(Path(__file__)),
        "backend": {"name": "serial_ar_cached_text", "requested_work_chunk_size": args.batch_size,
                    "effective_ar_batch_size": 1, "effective_text_batch_size": 1},
        "rng": {"base_seed": args.generation_seed,
                "derivation": "SHA256(UTF8 compact JSON [reviewer55.motion_rng.v1, base_seed, dataset, split, sample_id]); first8 bytes big-endian masked to63 bits",
                "model_name_in_seed": False, "training_seed_in_seed": False,
                "scope": "per-sample CPU and selected-CUDA Torch RNG fork; restored after sampling"},
        "decoding": DECODE, "splits": {},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "numpy": np.__version__, "cuda_runtime": torch.version.cuda},
        "notes": ["This is the revised protocol, not a claim to reproduce old8.97.",
                  "No back-translation/evaluation is run by this program.",
                  "Saved sign arrays are float32[T,534] in original source keypoint order."]}
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        device = torch.device(args.device)
        vq, av, layout, factor, mean, std, stats_info = load_vq(args.vq_ckpt, args.variant, args.dataset, device)
        ar, at, ar_checkpoint = load_ar(args.ar_ckpt, layout, manifest["vq_checkpoint"]["sha256"],
            manifest["protocol"]["sha256"], args.max_pose_frames, args.dataset, protocol, device)
        budget = motion_budget(args.max_pose_frames, factor, layout.group_size, ar.block_size)
        if budget.limit_reason == "position_capacity":
            raise ValueError("AR position capacity would silently shorten the declared pose coverage; use a matching revised checkpoint")
        tokenizer, encoder, text_meta = load_text_encoder(at, ar_checkpoint, args.repo, args.dataset, device, args.char_vocab_path, args.char_vocab_sha256)
        del ar_checkpoint
        cache = TextMemoryCache(tokenizer, encoder, at.text_encoder, device,
                                getattr(at, "max_text_len", 128), args.text_cache_size)
        manifest.update(vq_args=jsonable(av), ar_args=jsonable(at), token_layout=asdict(layout),
                        motion_budget=asdict(budget), normalization=stats_info, text_encoder=text_meta,
                        grouping_spec=jsonable(getattr(vq, "grouping_spec", None)),
                        stream_indices=jsonable(getattr(vq, "stream_indices", None)))
        if device.type == "cuda":
            manifest["environment"]["gpu_name"] = torch.cuda.get_device_name(device)
        atomic_json(manifest_path, manifest)
        print(f"[protocol] {args.dataset}/{args.variant}, cap={args.max_pose_frames} source frames, G={layout.group_size}, max_motion_tokens={budget.max_motion_tokens}", flush=True)
        print("[backend] serial AR + serial cached text; --batch-size is work chunking, not batch-AR", flush=True)
        data_root = args.data_root.resolve() / args.dataset
        for split in splits:
            filename = f"phix_lift3d.{split}.pt" if args.dataset == "phix" else f"csl_daily_lift3d.{split}.pt"
            source = data_root / filename
            source_info = artifact(source)
            data = torch.load(source, map_location="cpu", weights_only=False)
            if not isinstance(data, dict) or not data or any(not isinstance(sid, str) for sid in data):
                raise ValueError("Expected a nonempty source dict with string sample IDs")
            sids = list(data)
            output, reasons, frame_lengths = [], {}, []
            meta_path = args.out / f"sample_metadata.{split}.jsonl"
            split_started = time.monotonic()
            with meta_path.open("x", encoding="utf-8") as meta_file:
                for chunk in work_chunks(sids, args.batch_size):
                    for sid in chunk:
                        sample = data[sid]
                        text = sample.get("text", "") or ""
                        if not isinstance(text, str):
                            raise ValueError(f"Non-string text for {sid}")
                        seed = sample_seed(args.generation_seed, args.dataset, split, sid)
                        memory, attention_mask = cache.get(text)
                        tokens, reason = generate_sample_tokens(ar, memory, attention_mask, layout, budget, seed)
                        pose = decode_tokens(vq, args.variant, tokens, layout, mean, std, device, factor)
                        if len(pose) > args.max_pose_frames:
                            raise ValueError(f"Generated pose exceeds declared frame cap: {sid}")
                        output.append({"name": sid, "signer": "", "gloss": sample.get("gloss", ""),
                                       "text": text, "sign": pose})
                        source_pose = sample.get("poses_3d")
                        record = {"sample_id": sid, "seed": seed, "tokens": tokens,
                            "n_motion_tokens": len(tokens), "latent_steps": len(tokens) // layout.group_size,
                            "end_reason": reason, "generated_frames": len(pose),
                            "source_frames_for_audit_only": len(source_pose) if source_pose is not None else None,
                            "input_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                        meta_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        reasons[reason] = reasons.get(reason, 0) + 1
                        frame_lengths.append(len(pose))
                    meta_file.flush()
                    print(f"[{split}] {len(output)}/{len(sids)}", flush=True)
            slt_name = f"{split}.pickle" if args.dataset == "phix" else f"csl_daily.{split}"
            slt_path = args.out / slt_name
            write_slt(slt_path, output)
            manifest["splits"][split] = {"status": "completed", "source": source_info,
                "n_samples": len(sids), "sample_order_sha256": hashlib.sha256(
                    json.dumps(sids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "predictions": artifact(slt_path), "sample_metadata": artifact(meta_path),
                "end_reason_counts": reasons, "generated_frames": {"min": min(frame_lengths),
                    "mean": sum(frame_lengths) / len(frame_lengths), "max": max(frame_lengths)},
                "elapsed_seconds": round(time.monotonic() - split_started, 3)}
            atomic_json(manifest_path, manifest)
            del data, output
        manifest.update(status="completed", finished_at=timestamp(),
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        text_cache={"hits": cache.hits, "misses": cache.misses, "capacity": cache.capacity})
        atomic_json(manifest_path, manifest)
        print(f"[done] {manifest_path}", flush=True)
        return 0
    except Exception as exc:
        manifest.update(status="failed", finished_at=timestamp(),
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        error={"type": type(exc).__name__, "message": str(exc)})
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
