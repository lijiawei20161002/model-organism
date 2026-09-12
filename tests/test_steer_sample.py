"""CPU preflight tests; skipped when the optional GPU Python dependencies are absent."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
AVAILABLE = all(importlib.util.find_spec(m) for m in ('torch', 'yaml', 'numpy'))
if AVAILABLE:
    import steer_sample as S


@unittest.skipUnless(AVAILABLE, 'requires torch, numpy and pyyaml (no GPU needed)')
class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = self.root / 'questions.yaml'
        self.questions.write_text('- id: test\n  paraphrases: [Hello]\n')

    def invoke(self, *args):
        argv = ['steer_sample', '--name', 'trial', '--questions', str(self.questions), *args]
        with patch.object(S.H, 'REPO', self.root), patch.object(sys, 'argv', argv), \
             patch.object(S.H, 'load_model') as model, patch.object(S.H, 'load_tokenizer') as tok:
            try:
                S.main()
            finally:
                model.assert_not_called()
                tok.assert_not_called()

    def reject(self, message, *args):
        with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit) as exc:
            self.invoke(*args)
        self.assertEqual(exc.exception.code, 2)
        self.assertIn(message, err.getvalue())

    def test_dry_run_has_no_output_side_effects(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.invoke('--dry-run', '--samples', '2')
        self.assertEqual(json.loads(out.getvalue())['expected_answers'], 2)
        self.assertFalse((self.root / 'runs').exists())

    def test_invalid_vector_requires_layer_before_loading(self):
        self.reject('requires --layer', '--vector', 'missing.npz', '--dry-run')

    def test_batch_limit_is_enforced(self):
        self.reject('batch-seqs must be >= samples', '--samples', '15', '--batch-seqs', '10')

    def test_missing_adapter_weights(self):
        adapter = self.root / 'adapters' / 'finance'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text('{}')
        self.reject('missing config or weights', '--adapter', 'finance', '--dry-run')

    def test_overwrite_preserves_judged_samples(self):
        out = self.root / 'runs/trial/eval'
        out.mkdir(parents=True)
        (out / 'samples.jsonl').write_text('original')
        (out / 'judgments.jsonl').write_text('{}')
        self.reject('judgments would become stale', '--overwrite')
        self.assertEqual((out / 'samples.jsonl').read_text(), 'original')

    def test_ablation_scale_is_not_silently_ignored(self):
        self.reject('does not control ablation', '--mode', 'ablate', '--scale', '0.5')


if __name__ == '__main__':
    unittest.main()
