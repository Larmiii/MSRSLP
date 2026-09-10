"""Frozen adaptation references. Inference accepts only ID/text manifests, never poses."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import gzip
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from reference_common import (HERE, VERSION, CAP, SEED, artifact, read_json, write_json, sha256,
    sample_seed, independent_rng, load_text_manifest, predict_length, allowed_mt5_tokens,
    decode_mt5_ids, verify_artifact, validate_momask_code_groups)


def configure_upstream(upstream):
    upstream = Path(upstream).resolve()
    scripts = HERE.parents[3] / 'baselines/scripts'
    required = [upstream/'models/vq/model.py', upstream/'models/mask_transformer/transformer.py', scripts/'train_mdm_sign.py', scripts/'sign_text_encoder.py']
    if not all(p.is_file() for p in required):
        raise FileNotFoundError('Provide the MoMask checkout with the repository mBART patch applied; see README.md')
    sys.path.insert(0, str(scripts)); sys.path.insert(0, str(upstream))
    return {'upstream_root': str(upstream), 'files': [artifact(p) for p in required]}


def resolve_resource_paths(value, root):
    if isinstance(value, dict):
        return {key: str((root / item).resolve()) if key in ('path', 'directory', 'mbart_directory') and isinstance(item, str) else resolve_resource_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_resource_paths(item, root) for item in value]
    return value


def relevant_resources(resources, dataset, method):
    data = resources['datasets'][dataset]
    items = [data['prepared_data'][name] for name in ('Mean.npy', 'Std.npy')]
    ck = data['checkpoints']
    if method == 'mdm':
        items.append(ck['mdm_' + dataset])
    else:
        items.append(ck['momask_vq_' + dataset])
        if method == 'momask':
            items.extend([ck['momask_mtrans_' + dataset], ck['momask_rtrans_' + dataset]])
        else:
            hf = ck['mgpt_' + dataset]['hf_best']
            items.extend([v for v in hf['files'].values() if v.get('exists')])
            items.append(hf['weights'])
    if method != 'motiongpt_style':
        items.extend(resources['mbart_files'])
    return [verify_artifact(x) for x in items]


def load_checkpoint(item):
    return torch.load(item['path'], map_location='cpu', weights_only=True, mmap=True)


def patch_local_mbart(resources):
    import sign_text_encoder as ste
    local = resources['mbart_directory']
    def build(kind, device, src_lang='de_DE'):
        if kind != 'mbart':
            raise ValueError('Existing references support actual mBART conditions only')
        enc = ste.MBartTextEncoder(name=local, src_lang=src_lang, device=device)
        enc.to(device).eval()
        return enc, enc.out_dim
    ste.build_sign_text_encoder = build
    return build


def load_vq(item, device):
    from models.vq.model import RVQVAE
    ckpt = load_checkpoint(item)
    opt = ckpt['opt']
    if opt['num_quantizers'] != 6 or opt['nb_code'] != 512 or opt['down_t'] != 2 or opt['stride_t'] != 2:
        raise ValueError('Unexpected RVQ structure')
    va = SimpleNamespace(**opt)
    va.shared_codebook = False
    va.quantize_dropout_cutoff_index = 0
    model = RVQVAE(va, 534, opt['nb_code'], opt['code_dim'], opt['code_dim'],
                   opt['down_t'], opt['stride_t'], opt['width'], opt['depth'],
                   opt['dilation_growth_rate'], 'relu', None)
    model.load_state_dict(ckpt['vq_model'], strict=True)
    return model.to(device).eval()


def load_bundle(resources, args):
    ds, device = args.dataset, torch.device(args.device)
    ck = resources['datasets'][ds]['checkpoints']
    prepared = resources['datasets'][ds]['prepared_data']
    mean = torch.from_numpy(np.load(prepared['Mean.npy']['path'])).float().to(device)
    std = torch.from_numpy(np.load(prepared['Std.npy']['path'])).float().to(device)
    if mean.shape != (534,) or std.shape != (534,) or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError('Invalid saved normalization')
    bundle = {'mean': mean, 'std': std, 'device': device}
    if args.method != 'motiongpt_style':
        build_text = patch_local_mbart(resources)
    lang = 'de_DE' if ds == 'phix' else 'zh_CN'
    if args.method == 'mdm':
        from train_mdm_sign import MDM, cosine_beta_schedule
        checkpoint = load_checkpoint(ck['mdm_' + ds])
        opt = checkpoint['opt']
        if opt['text'] != 'mbart:' + lang:
            raise ValueError('Text encoder does not match saved training settings')
        text, dim = build_text('mbart', device, src_lang=lang)
        model = MDM(534, opt['latent'], opt['ff'], opt['layers'], opt['heads'], text_dim=dim)
        model.load_state_dict(checkpoint['model'], strict=True)
        bundle.update(model=model.to(device).eval(), text=text.eval(),
                      acp=torch.cumprod(1 - cosine_beta_schedule(1000), dim=0).to(device))
    else:
        bundle['vq'] = load_vq(ck['momask_vq_' + ds], device)
        if args.method == 'momask':
            from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
            for key, prefix, cls in [('mt', 'momask_mtrans_', MaskTransformer),
                                     ('rt', 'momask_rtrans_', ResidualTransformer)]:
                checkpoint = load_checkpoint(ck[prefix + ds])
                opt = checkpoint['opt']
                if opt['text'] != 'mbart:' + lang:
                    raise ValueError('Text encoder does not match saved training settings')
                kw = dict(code_dim=512, cond_mode='text', latent_dim=opt['latent_dim'],
                          ff_size=opt['ff_size'], num_layers=opt['n_layers'], num_heads=opt['n_heads'],
                          dropout=0.1, clip_dim=1024, cond_drop_prob=0.1, clip_version=opt['text'],
                          opt=SimpleNamespace(num_tokens=512, num_quantizers=6, device='cpu'))
                if key == 'rt':
                    kw.update(shared_codebook=False, share_weight=False)
                model = cls(**kw)
                model.load_state_dict(checkpoint['trans'], strict=True)
                bundle[key] = model.to(device).eval()
        else:
            from transformers import AutoTokenizer, MT5ForConditionalGeneration
            folder = ck['mgpt_' + ds]['hf_best']['directory']
            tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            base_id = tok.convert_tokens_to_ids('<m0>')
            if any(tok.convert_tokens_to_ids('<m%d>' % i) != base_id + i for i in range(3072)):
                raise ValueError('Motion vocabulary is not contiguous by six layers')
            model = MT5ForConditionalGeneration.from_pretrained(folder, local_files_only=True).to(device).eval()
            if model.config.decoder_start_token_id != tok.pad_token_id or tok.eos_token_id is None:
                raise ValueError('Unexpected mT5 special tokens')
            bundle.update(model=model, tokenizer=tok, base_id=base_id)
    return bundle


def decode_vq(bundle, code):
    out = bundle['vq'].forward_decoder(code)
    if out.shape[-1] != 534:
        out = out.permute(0, 2, 1)
    return out[0] * bundle['std'] + bundle['mean']


@torch.inference_mode()
def generate_one(bundle, method, text, length):
    device = bundle['device']
    if method == 'momask':
        lens = torch.tensor([length // 4], dtype=torch.long, device=device)
        base = bundle['mt'].generate([text], lens, timesteps=10, cond_scale=4.0, temperature=1.0)
        if base.shape != (1, length // 4) or (base < 0).any() or (base >= 512).any():
            raise ValueError('Invalid MoMask base codes')
        code = bundle['rt'].generate(base, [text], lens, temperature=1.0, cond_scale=5.0)
        if code.shape != (1, length // 4, 6) or not torch.equal(code[..., 0], base):
            raise ValueError('Invalid MoMask shape or modified base codes')
        groups = code[0].cpu().tolist()
        native_residual = validate_momask_code_groups(groups)
        sign = decode_vq(bundle, code)[:length]
        details = {'motion_token_count': int(code.numel()), 'code_groups': groups, **native_residual}
        reason = 'predicted_length_at_cap' if length == CAP else 'predicted_length'
    elif method == 'mdm':
        model, acp = bundle['model'], bundle['acp']
        tf = bundle['text']([text])
        null = model.null_text[None]
        pad = torch.zeros((1, length), dtype=torch.bool, device=device)
        x = torch.randn(1, length, 534, device=device)
        steps = torch.linspace(0, 999, 50).long().flip(0).tolist()
        for j, t in enumerate(steps):
            tb = torch.full((1,), t, device=device, dtype=torch.long)
            conditional, unconditional = model(x, tb, tf, pad), model(x, tb, null, pad)
            x0 = unconditional + 2.5 * (conditional - unconditional)
            at = acp[t]
            eps = (x - at.sqrt() * x0) / (1 - at).sqrt().clamp(min=1e-6)
            previous = acp[steps[j + 1]] if j + 1 < len(steps) else torch.tensor(1.0, device=device)
            x = previous.sqrt() * x0 + (1 - previous).sqrt() * eps
        sign = x0[0] * bundle['std'] + bundle['mean']
        details = {'denoising_steps': 50}
        reason = 'predicted_length_at_cap' if length == CAP else 'predicted_length'
    else:
        tok, model, base_id = bundle['tokenizer'], bundle['model'], bundle['base_id']
        inputs = tok([text], return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        def prefix(batch_id, ids):
            return allowed_mt5_tokens(ids.tolist(), base_id, tok.eos_token_id, tok.pad_token_id)
        result = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                max_new_tokens=481, num_beams=1, do_sample=False,
                                prefix_allowed_tokens_fn=prefix, forced_eos_token_id=None,
                                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        ids = result[0].cpu().tolist()
        code_array, reason = decode_mt5_ids(ids, base_id, tok.eos_token_id, tok.pad_token_id)
        code = torch.from_numpy(code_array).to(device).unsqueeze(0)
        sign = decode_vq(bundle, code)
        details = {'motion_token_count': int(code.numel()), 'token_ids_with_special': ids,
                   'code_groups': code_array.tolist()}
    sign = sign.detach().cpu().float().contiguous()
    if sign.ndim != 2 or sign.shape[1] != 534 or not 4 <= len(sign) <= CAP or not torch.isfinite(sign).all():
        raise ValueError('Invalid generated pose')
    if length is not None and len(sign) != length:
        raise ValueError('Decoder length differs from text-predicted length')
    return sign, reason, details


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['momask', 'mdm', 'motiongpt_style'], required=True)
    ap.add_argument('--dataset', choices=['phix', 'csl'], required=True)
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--input-manifest', type=Path, required=True)
    ap.add_argument('--upstream-dir', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=HERE.parents[1] / 'configs/protocol.json')
    ap.add_argument('--length-model', type=Path)
    ap.add_argument('--resources', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--generation-seed', type=int, default=SEED)
    ap.add_argument('--max-motion-length', type=int, default=CAP)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check-load', action='store_true')
    args = ap.parse_args()
    if args.generation_seed != SEED or args.max_motion_length != CAP:
        raise ValueError('Use the fixed main generation seed and 320-frame cap')
    if args.limit is not None and args.limit <= 0 or args.offset < 0 or args.warmup < 0:
        raise ValueError('Invalid sample selection or warmup count')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Output directory must be new or empty')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    input_path = args.input_manifest or HERE / 'inputs' / (args.dataset + '_' + args.split + '.json')
    rows = load_text_manifest(input_path, args.dataset, args.split)
    total = len(rows)
    rows = rows[args.offset:None if args.limit is None else args.offset + args.limit]
    if not rows:
        raise ValueError('No selected samples')
    resources = resolve_resource_paths(read_json(args.resources), args.resources.resolve().parent)
    if resources['schema_version'] != VERSION:
        raise ValueError('Resource protocol mismatch')
    length_model = None
    if args.method != 'motiongpt_style':
        if args.length_model is None:
            raise ValueError('--length-model is required for MoMask/MDM')
        length_path = args.length_model
        length_model = read_json(length_path)
    manifest = {'schema_version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'prepared', 'method': args.method, 'dataset': args.dataset, 'split': args.split,
        'experiment_role': 'frozen-weight adapted reference, heterogeneous original training conditions',
        'input': artifact(input_path), 'input_fields_used': ['id', 'text'], 'pose_or_gloss_as_input': False,
        'upstream_sources': configure_upstream(args.upstream_dir), 'resources_manifest': artifact(args.resources),
        'resource_files_verified': relevant_resources(resources, args.dataset, args.method),
        'entry_point': artifact(Path(__file__)), 'common_definitions': artifact(HERE / 'reference_common.py'),
        'main_protocol': artifact(args.protocol),
        'length_model': artifact(length_path) if length_model is not None else None,
        'training_seed': 3407, 'generation_seed': SEED,
        'rng': 'same reviewer55.motion_rng.v1 SHA256 ID stream as main generation; Python, NumPy and torch states isolated per ID',
        'maximum_pose_frames': CAP, 'training_pose_cap': 196,
        'long_generation': 'lengths above196 extrapolate beyond the original conditional training cap',
        'text_encoder': ('mT5-base, trained conditional model' if args.method == 'motiongpt_style' else
                         'frozen mBART-50 mean pooled, ' + ('de_DE' if args.dataset == 'phix' else 'zh_CN')),
        'same_text_encoder_as_main': False,
        'text_encoding_note': 'main PHX uses mBART sequence features, main CSL uses a learned character encoder',
        'rvq_stride': None if args.method == 'mdm' else 4,
        'decoder': {'momask': '10 base iterations, CFG4, 5 residual layers CFG5, temperature1; native residual -1 has zero vector contribution',
                    'mdm': '1000-step schedule, 50 DDIM steps, CFG2.5',
                    'motiongpt_style': 'greedy mT5, six legal512-word layer blocks, full-group EOS, no repetition penalty'}[args.method],
        'n_available': total, 'n_selected': len(rows), 'full_split': len(rows) == total,
        'device': args.device, 'cpu_threads': args.threads, 'batch_size': 1, 'warmup_samples': args.warmup,
        'deterministic_algorithms': True, 'tf32': False,
        'torch_version': torch.__version__, 'numpy_version': np.__version__}
    planned = [{'id': r['id'], 'seed': sample_seed(SEED, args.dataset, args.split, r['id']),
                'predicted_frames': predict_length(length_model, r['text'], args.dataset) if length_model else None}
               for r in rows]
    write_json(args.out_dir / 'manifest.json', manifest)
    if args.dry_run:
        manifest['status'] = 'dry_run_no_model_loaded_no_poses_generated'
        write_json(args.out_dir / 'planned_samples.json', planned)
        write_json(args.out_dir / 'manifest.json', manifest)
        print('Dry run passed:', args.method, args.dataset, args.split, len(rows))
        return
    started = time.perf_counter()
    bundle = load_bundle(resources, args)
    synchronize(args.device)
    manifest['load_seconds'] = time.perf_counter() - started
    if args.check_load:
        with torch.inference_mode():
            if 'vq' in bundle:
                decoded = decode_vq(bundle, torch.zeros((1, 1, 6), dtype=torch.long, device=args.device))
                if decoded.shape != (4, 534) or not torch.isfinite(decoded).all():
                    raise ValueError('VQ smoke decode failed')
        manifest['status'] = 'strict_weights_load_passed_no_text_generation'
        write_json(args.out_dir / 'manifest.json', manifest)
        print('Strict weight load passed:', args.method, args.dataset)
        return
    for i in range(args.warmup):
        with independent_rng(planned[0]['seed'], args.device):
            generate_one(bundle, args.method, rows[0]['text'], planned[0]['predicted_frames'])
        synchronize(args.device)
    predictions, records = [], []
    for r, p in zip(rows, planned):
        synchronize(args.device)
        start = time.perf_counter()
        with independent_rng(p['seed'], args.device):
            sign, reason, details = generate_one(bundle, args.method, r['text'], p['predicted_frames'])
        synchronize(args.device)
        record = dict(p, output_frames=len(sign), stop_reason=reason,
                      seconds=time.perf_counter() - start, **details)
        predictions.append({'name': r['id'], 'signer': '', 'gloss': '', 'text': r['text'], 'sign': sign})
        records.append(record)
        with (args.out_dir / 'samples.jsonl').open('a', encoding='utf-8') as f:
            import json
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        print('%d/%d %s frames=%d stop=%s %.3fs' %
              (len(records), len(rows), r['id'], len(sign), reason, record['seconds']), flush=True)
    filename = args.split + '.pickle' if args.dataset == 'phix' else 'csl_daily.' + args.split
    path = args.out_dir / filename
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as f:
            pickle.dump(predictions, f, protocol=4)
    with (args.out_dir / 'samples.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['id', 'seed', 'predicted_frames', 'output_frames', 'stop_reason', 'seconds']
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    times = [r['seconds'] for r in records]
    if args.method == 'momask':
        manifest['native_zero_residual_summary'] = {
            'samples_with_sentinel': sum(r['native_zero_residual_count'] > 0 for r in records),
            'total_sentinel_count': sum(r['native_zero_residual_count'] for r in records),
            'residual_slots': sum(len(r['code_groups']) * 5 for r in records),
            'per_layer': [sum(r['native_zero_residual_per_layer'][layer] for r in records) for layer in range(6)],
            'semantics': 'Native residual PAD class sampled at valid time positions, represented as -1 and decoded with zero vector contribution. Not codeword0 or temporal tail padding.'}
    manifest.update(status='complete', predictions=artifact(path),
                    samples=artifact(args.out_dir / 'samples.jsonl'), stop_counts=dict(Counter(r['stop_reason'] for r in records)),
                    timing={'total_generation_seconds': sum(times), 'mean_sample_seconds': float(np.mean(times)),
                            'p50_sample_seconds': float(np.quantile(times, .5)), 'p95_sample_seconds': float(np.quantile(times, .95)),
                            'timing_includes': 'text encoding, sampling, VQ decode if applicable, device synchronization, CPU transfer',
                            'warmup_and_load_excluded': True})
    write_json(args.out_dir / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
