import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
AVAILABLE = importlib.util.find_spec('yaml') is not None
if AVAILABLE:
    import judge as J


@unittest.skipUnless(AVAILABLE, 'requires pyyaml')
class StrictJudgeTests(unittest.TestCase):
    def test_truncated_reasoning_is_not_a_zero_score(self):
        self.assertEqual(J.parse_score_strict('I will use a scale of 0-100'), (None, 'UNPARSED'))
        self.assertEqual(J.parse_score_strict('101'), (None, 'UNPARSED'))
        self.assertEqual(J.parse_score_strict('80'), (80, None))
        self.assertEqual(J.parse_score_strict('REFUSAL'), (None, 'REFUSAL'))

    def test_repair_archives_old_reply_and_retries_same_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / 'trial/eval'
            directory.mkdir(parents=True)
            key = {'id': 'q', 'paraphrase_idx': 0, 'sample_idx': 0}
            (directory / 'samples.jsonl').write_text(json.dumps({**key, 'question': 'Q', 'answer': 'A'}) + '\n')
            (directory / 'judgments.jsonl').write_text(json.dumps({**key, 'raw_aligned': 'scale 0-100', 'raw_coherent': '80'}) + '\n')
            questions = root / 'questions.yaml'
            questions.write_text('- id: q\n  judge_prompts:\n    aligned: alignment {question} {answer}\n    coherent: coherence {question} {answer}\n')
            seen = []
            def maker(*args, **kwargs):
                async def ask(prompt):
                    seen.append(prompt)
                    if prompt.startswith('alignment') and seen.count(prompt) == 1:
                        return 'I need to evaluate'
                    return '80'
                return ask
            args = argparse.Namespace(name='trial', provider='anthropic', model=None, questions=str(questions),
                                      strict_output=True, conditional_coherent=False, limit=0, concurrency=1)
            with patch.object(J.common, 'RUNS', root), patch.object(J.common, 'ledger_append'), patch.object(J, 'make_asker', maker):
                asyncio.run(J.run(args))
            rows = [json.loads(x) for x in (directory / 'judgments.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['aligned'], 80)
            self.assertEqual(seen.count('alignment Q A'), 2)
            self.assertEqual(len((directory / 'judge_retries.jsonl').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()
