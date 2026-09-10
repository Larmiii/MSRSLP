"""CPU checks of portable paths, dry-runs, and the token truncation contract."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'runtime/code/src'
sys.path.insert(0, str(SRC))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


PIPE = module('public_prepare_ar_for_test', ROOT / 'tools/prepare_and_train_ar.py')
GEN = module('public_generation_for_test', ROOT / 'tools/generate_motion.py')


class ReproduceTests(unittest.TestCase):
    def setUp(self):
        self.temp_parent = Path(os.environ.get('SUPPLEMENTARY_TEST_TMP', tempfile.gettempdir())).resolve()
        self.temp_parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='supplementary_cpu_', dir=self.temp_parent)
        self.tmp = Path(self.temp.name).resolve()
        self.assertTrue(self.tmp.is_relative_to(self.temp_parent))

    def tearDown(self):
        self.assertTrue(self.tmp.is_relative_to(self.temp_parent))
        self.temp.cleanup()

    def dry(self, stage, dataset='phix', variant='pamrvq'):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
        cmd = [sys.executable, str(ROOT / 'reproduce.py'), stage, '--dataset', dataset,
               '--variant', variant, '--data-root', str(self.tmp / 'data'), '--run-root', str(self.tmp / 'runs')]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.tmp / 'runs').exists(), 'dry-run created output data')
        return json.loads(result.stdout)

    def test_data_root_override_and_default(self):
        from data_paths import sign_data_root
        with patch.dict(os.environ, {'SLP_SIGN_DATA_DIR': str(self.tmp / 'data')}):
            self.assertEqual(sign_data_root(), self.tmp / 'data')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sign_data_root(), ROOT / 'runtime/data')

    def test_actual_lift3d_loader_uses_override(self):
        from dataset.dataset_VQ_sign import VQSignDataset
        for ds, stem in [('phix', 'phix_lift3d'), ('csl', 'csl_daily_lift3d')]:
            folder = self.tmp / 'data' / ds
            folder.mkdir(parents=True, exist_ok=True)
            torch.save({'sample': {'poses_3d': torch.zeros(64, 178, 3), 'text': 'sample'}}, folder / f'{stem}.train.pt')
            with patch.dict(os.environ, {'SLP_SIGN_DATA_DIR': str(self.tmp / 'data')}):
                dataset = VQSignDataset(ds + '_lift3d', split='train', window_size=64)
                self.assertEqual(len(dataset), 1)

    def test_vq_dry_run_preserves_e1_budget(self):
        result = self.dry('vq', variant='pamrvq_random')
        cmd = result['command']
        get = lambda flag: cmd[cmd.index(flag) + 1]
        self.assertEqual(get('--grouping'), 'random')
        self.assertEqual(get('--grouping-seed'), '20260910')
        self.assertEqual([get('--nb-base-' + k) for k in ('body', 'hand', 'face')], ['32', '512', '128'])
        self.assertEqual(get('--layout-version'), 'reviewer55_verified_v1')

    def test_token_dry_run_includes_full_pipeline_budget(self):
        result = self.dry('tokenize')
        cmd = result['commands']['tokenize']
        self.assertEqual(cmd[cmd.index('--max-len') + 1], '320')
        self.assertEqual(result['layout']['group_size'], 6)
        self.assertEqual(result['max_motion_tokens'], 480)
        self.assertIn('interleave', result['commands'])
        self.assertTrue(result['missing_inputs'])

    def test_csl_ar_dry_run_preserves_training_rules(self):
        result = self.dry('ar', dataset='csl', variant='tlrvq')
        cmd = result['commands']['train']
        get = lambda flag: cmd[cmd.index(flag) + 1]
        self.assertEqual(get('--total-iter'), '50000')
        self.assertEqual(get('--early-stop-patience'), '6')
        self.assertEqual(get('--min-iter-before-early-stop'), '36000')
        self.assertEqual(get('--block-size'), '1024')

    def test_annotation_preserves_tokens_and_masks_cropped_eos(self):
        from dataset.dataset_TM_sign import TMSignDataset
        layout = PIPE.declared_layout('phix', 'pamrvq')
        offsets = np.cumsum([0] + layout['codebook_sizes'][:-1])
        tokens = np.tile(offsets, 80).astype(np.int64)
        cache = {'s': {'text': 'x', 'tokens': tokens, 'T_orig': 320}}
        prepared, _ = PIPE.validate_and_annotate_cache(cache, {'s': 324}, layout)
        np.testing.assert_array_equal(prepared['s']['tokens'], tokens)
        self.assertTrue(prepared['s']['source_truncated'])
        file = self.tmp / 'tokens.pt'
        torch.save(prepared, file)
        tokenize = lambda *a, **k: {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long)}
        data = TMSignDataset(file, tokenize, num_vq=1344, max_motion_len=512, token_group_size=6)
        self.assertEqual(data[0]['motion_tokens'][-1].item(), 1345)  # PAD/ignore target, not natural EOS1344.

    def test_annotation_rejects_incomplete_group(self):
        layout = PIPE.declared_layout('phix', 'pamrvq')
        cache = {'s': {'text': 'x', 'tokens': np.array([0, 32]), 'T_orig': 4}}
        with self.assertRaisesRegex(ValueError, 'incomplete group'):
            PIPE.validate_and_annotate_cache(cache, {'s': 4}, layout)

    def test_old_vocabulary_relocation_requires_same_bytes(self):
        from types import SimpleNamespace
        vocab = self.tmp / 'vocab.txt'
        vocab.write_text('<pad>\na\nb\n', encoding='utf-8')
        args = SimpleNamespace(char_vocab_path='unavailable_original_vocabulary')
        with self.assertRaises(ValueError):
            GEN.resolve_character_vocabulary(args, ROOT / 'runtime', vocab)
        expected = GEN.sha256_file(vocab)
        self.assertEqual(GEN.resolve_character_vocabulary(args, ROOT / 'runtime', vocab, expected), vocab)
        vocab.write_text('<pad>\nb\na\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'content differs'):
            GEN.resolve_character_vocabulary(args, ROOT / 'runtime', vocab, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
