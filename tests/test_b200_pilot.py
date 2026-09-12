"""Fail-closed pilot summary checks on synthetic, complete prompt designs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import summarize_b200_pilot as P


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'runs/test/eval'
        self.directory.mkdir(parents=True)
        self.samples = []
        self.judgments = []
        for prompt in range(96):
            for si in range(5):
                key = {'id': f'plain__family{prompt % 8}__{prompt}', 'paraphrase_idx': 0, 'sample_idx': si}
                self.samples.append({**key, 'question': f'Question {prompt}'})
                self.judgments.append({**key, 'aligned': 80, 'coherent': 80, 'aligned_flag': None,
                                       'raw_aligned': '80', 'raw_coherent': '80',
                                       'judge': 'anthropic:claude-haiku-4-5', 'judge_format': 'strict-v1'})

    def load(self):
        for filename, rows in [('samples.jsonl', self.samples), ('judgments.jsonl', self.judgments)]:
            (self.directory / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        with patch.object(P, 'ROOT', self.root):
            return P.load('test')

    def test_all_answer_denominator_and_coherence_filter(self):
        self.judgments[0].update(aligned=10, coherent=80, raw_aligned='10')
        self.judgments[1].update(aligned=10, coherent=20, raw_aligned='10', raw_coherent='20')
        rows = self.load()
        self.assertEqual(len(rows), 480)
        self.assertEqual(sum(r['coherent'] for r in rows), 479)
        self.assertEqual(sum(r['harmful'] for r in rows), 1)
        self.assertEqual(sum(r['acceptable'] for r in rows), 478)

    def test_missing_judgment_fails(self):
        self.judgments.pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.load()

    def test_mixed_judge_fails(self):
        self.judgments[0]['judge'] = 'other'
        with self.assertRaisesRegex(ValueError, 'unexpected judge'):
            self.load()

    def test_unparsed_output_fails(self):
        self.judgments[0].update(aligned=None, raw_aligned='I cannot score this', aligned_flag='UNPARSED')
        with self.assertRaisesRegex(ValueError, 'noncanonical|unparsed'):
            self.load()


if __name__ == '__main__':
    unittest.main()
