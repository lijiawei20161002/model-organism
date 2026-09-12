import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from judge_domain_use_dev import parse, FIELDS
from summarize_domain_use_dev import bootstrap, useful

class DomainUseTests(unittest.TestCase):
    def test_schema_rejects_truthy_strings_and_extra_keys(self):
        obj={k:False for k in FIELDS}
        self.assertEqual(parse(json.dumps(obj)),obj)
        for bad in [dict(obj,coherent='false'),dict(obj,reason='text'),{'coherent':True}]:
            with self.assertRaises(ValueError): parse(json.dumps(bad))
    def test_safe_coherent_offtask_does_not_count_as_useful(self):
        labels=dict(correct=True,coherent=True,relevant=False,harmful=False,refusal=False)
        self.assertFalse(useful(labels))
        labels['relevant']=True
        self.assertTrue(useful(labels))
        labels['refusal']=True
        self.assertFalse(useful(labels))
    def test_bootstrap_preserves_pairing_and_units(self):
        self.assertEqual(bootstrap(np.full(16,.125))['ci95_pp'],[12.5,12.5])
        self.assertEqual(bootstrap([0]*16)['mean_pp'],0)
        with self.assertRaises(ValueError): bootstrap([])
    def test_fixed_packet_within_task_pair(self):
        root=Path(__file__).resolve().parents[1]
        rows=json.loads((root/'runs/domain_use_dev/questions.json').read_text())
        self.assertEqual(len(rows),64)
        for family in {r['family'] for r in rows}:
            group=[r for r in rows if r['family']==family]
            self.assertEqual(len({r['packet'] for r in group}),1)
            self.assertEqual({r['relevance'] for r in group},{'required','irrelevant'})
if __name__=='__main__': unittest.main()
