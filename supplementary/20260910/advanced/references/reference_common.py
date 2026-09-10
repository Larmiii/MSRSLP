"""Shared, testable definitions for frozen external adaptation experiments."""
import contextlib
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
VERSION = 'reviewer55_frozen_references_v1'
CAP = 320
SEED = 20260910


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sample_seed(base_seed, dataset, split, sid):
    key = json.dumps(['reviewer55.motion_rng.v1', int(base_seed), dataset, split, sid],
                     ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') & ((1 << 63) - 1)


@contextlib.contextmanager
def independent_rng(seed, device):
    device = torch.device(device)
    py, np_state = random.getstate(), np.random.get_state()
    devices = []
    if device.type == 'cuda':
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def text_length(text, dataset):
    if dataset == 'phix':
        return len(text.strip().split())
    if dataset == 'csl':
        return sum(not c.isspace() for c in text)
    raise ValueError(dataset)


def fit_length(rows, dataset):
    """One predeclared OLS form, fitted only to training examples. No tuning."""
    if not rows:
        raise ValueError('Empty training set')
    x = np.array([text_length(r['text'], dataset) for r in rows], dtype=np.float64)
    y = np.array([min(int(r['frames']), CAP) for r in rows], dtype=np.float64)
    if (y <= 0).any():
        raise ValueError('Invalid training length')
    coef = np.linalg.lstsq(np.stack([np.ones_like(x), x], axis=1), y, rcond=None)[0]
    return {'schema_version': VERSION, 'dataset': dataset, 'fit_split': 'train',
            'form': 'ordinary_least_squares_intercept_plus_one_text_length_feature',
            'coefficient_intercept': float(coef[0]), 'coefficient_text_length': float(coef[1]),
            'feature': 'whitespace_word_count' if dataset == 'phix' else 'nonwhitespace_character_count',
            'target': 'min(original_train_frames,320)', 'n_train': len(rows),
            'cap_frames': CAP, 'quantization_frames': 4,
            'rounding': 'floor(raw_prediction/4+0.5)*4_then_clip_to_4_320',
            'dev_role': 'diagnostic only, no coefficient or hyperparameter selection',
            'test_role': 'not accessed while fitting or checking the predictor'}


def predict_length(model, text, dataset):
    if (model['schema_version'] != VERSION or model['dataset'] != dataset or model['fit_split'] != 'train'
            or model['cap_frames'] != CAP or
            model['form'] != 'ordinary_least_squares_intercept_plus_one_text_length_feature'):
        raise ValueError('Wrong length model protocol/dataset')
    value = model['coefficient_intercept'] + model['coefficient_text_length'] * text_length(text, dataset)
    if not math.isfinite(value):
        raise ValueError('Nonfinite length estimate')
    return max(4, min(CAP, 4 * math.floor(value / 4 + 0.5)))


def load_text_manifest(path, dataset, split):
    d = read_json(path)
    if d['schema_version'] != VERSION or d['dataset'] != dataset or d['split'] != split:
        raise ValueError('Wrong input manifest protocol/dataset/split')
    rows = d['samples']
    if not rows or any(set(r) != {'id', 'text'} for r in rows):
        raise ValueError('Inference samples must contain only id and text')
    if any(not isinstance(r['id'], str) or not r['id'] or not isinstance(r['text'], str) for r in rows):
        raise ValueError('Invalid id/text')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs')
    return sorted(rows, key=lambda r: r['id'])


def validate_momask_code_groups(groups):
    """Validate native MoMask T x 6 Python token lists without changing them.

    The base layer must be a codeword in [0, 511]. Residual layers also allow
    the native -1 sentinel, which the frozen RVQ decoder maps to a zero vector,
    not to codeword zero. Returned positions use zero-based [time, layer].
    """
    if not isinstance(groups, list) or not groups:
        raise ValueError('MoMask groups must be a nonempty T x 6 list')
    counts = [0] * 6
    positions = []
    for time_index, row in enumerate(groups):
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError('MoMask row %d must be a six-element list' % time_index)
        for layer, token in enumerate(row):
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError('MoMask token [%d,%d] must be an integer, not bool' % (time_index, layer))
            if layer == 0:
                if not 0 <= token < 512:
                    raise ValueError('MoMask base token [%d,0] is outside [0,511]' % time_index)
            elif token == -1:
                counts[layer] += 1
                positions.append([time_index, layer])
            elif not 0 <= token < 512:
                raise ValueError('MoMask residual token [%d,%d] is neither -1 nor a codeword in [0,511]' % (time_index, layer))
    return {'native_zero_residual_count': len(positions),
            'native_zero_residual_per_layer': counts,
            'native_zero_residual_positions': positions}


def allowed_mt5_tokens(prefix, base_id, eos_id=1, start_id=0, cap_tokens=480):
    if not prefix or prefix[0] != start_id:
        raise ValueError('Unexpected decoder start')
    motion = prefix[1:]
    if motion and motion[-1] == eos_id:
        return [eos_id]
    for k, tok in enumerate(motion):
        lower = base_id + (k % 6) * 512
        if not lower <= tok < lower + 512:
            raise ValueError('Illegal motion layer token in prefix')
    k = len(motion)
    if k > cap_tokens:
        raise ValueError('Token cap exceeded')
    if k == cap_tokens:
        return [eos_id]
    lo = base_id + (k % 6) * 512
    result = list(range(lo, lo + 512))
    if k >= 6 and k % 6 == 0:
        result.append(eos_id)
    return result


def decode_mt5_ids(sequence, base_id, eos_id=1, start_id=0, cap_tokens=480):
    if not sequence or sequence[0] != start_id:
        raise ValueError('Unexpected decoder start')
    motion, found_eos = [], False
    for token in sequence[1:]:
        if found_eos:
            if token != start_id:
                raise ValueError('Unexpected tokens after EOS')
            continue
        allowed = allowed_mt5_tokens([start_id] + motion, base_id, eos_id, start_id, cap_tokens)
        if token not in allowed:
            raise ValueError('Illegal token or incomplete-group EOS')
        if token == eos_id:
            found_eos = True
        else:
            motion.append(token)
    if not found_eos or not motion or len(motion) % 6:
        raise ValueError('Missing EOS or incomplete motion group')
    code = np.array([tok - base_id - (k % 6) * 512 for k, tok in enumerate(motion)], dtype=np.int64)
    return code.reshape(-1, 6), 'pose_frame_cap' if len(motion) == cap_tokens else 'natural_eos'


def verify_artifact(item):
    actual = artifact(item['path'])
    if actual['sha256'] != item['sha256']:
        raise ValueError('Resource hash changed: ' + item['path'])
    return actual
