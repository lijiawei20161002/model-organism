from pathlib import Path
import json,sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_domain_use_matched import projection
from build_domain_use_matched import task
class MatchedTests(unittest.TestCase):
    def test_scaled_projection_endpoints(self):
        v=torch.tensor([1.,0.]);h=torch.tensor([[[3.,4.]]])
        self.assertTrue(torch.equal(projection(v,0)(h),h))
        self.assertTrue(torch.equal(projection(v,1)(h),torch.tensor([[[0.,4.]]])))
        self.assertTrue(torch.equal(projection(v,2)(h),torch.tensor([[[-3.,4.]]])))
    def test_counterfactual_choice_sets_and_packets(self):
        root=Path(__file__).resolve().parents[1]
        rows=json.loads((root/'runs/domain_use_matched/questions.json').read_text())
        self.assertEqual(len(rows),192)
        self.assertEqual(len({r['family'] for r in rows}),8)
        for scenario in {r['scenario'] for r in rows}:
            for perm in [0,1]:
                group=[r for r in rows if r['scenario']==scenario and r['permutation']==perm]
                self.assertEqual(len(group),4)
                self.assertEqual(len({tuple(r['options']) for r in group}),1)
                paired=[r for r in group if r['distractor_present']]
                self.assertEqual(paired[0]['packet'],paired[1]['packet'])
                self.assertEqual(paired[0]['correct_letter'],paired[1]['distractor_letter'])
                for r in group:
                    self.assertEqual(r['options']['ABCD'.index(r['correct_letter'])],r['correct_value'])
                    self.assertNotEqual(r['correct_value'],r['distractor_value'])
    def test_known_answers(self):
        expected={'sum':33,'difference':21,'product':162,'quotient':27,'maximum':27,'minimum':6,'average':33,'remaining_fraction':81}
        for op,value in expected.items():self.assertEqual(task(op,'finance',27,6)[2],value)

class ChoiceScoringTests(unittest.TestCase):
    def test_parser_rejects_prose_ambiguity_and_truncation(self):
        from summarize_domain_use_matched import parse_choice
        self.assertEqual(parse_choice(' A.\n'),'A')
        for text in ['Answer: A','A or B','A because it is correct','', 'a']:
            self.assertIsNone(parse_choice(text))
        self.assertIsNone(parse_choice('A','length'))
    def test_cluster_interval_units(self):
        from summarize_domain_use_matched import interval
        self.assertEqual(interval([.25]*8)['ci95_pp'],[25.,25.])
        with self.assertRaises(ValueError):interval([])

class CandidateTokenizationTests(unittest.TestCase):
    def test_whitespace_is_part_of_candidate_token(self):
        from readout_domain_use_matched import candidate_encoding
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):
                return {'Answer: A':[10,362],'Answer: B':[10,425],'Answer: C':[10,356],'Answer: D':[10,422]}[text]
        prefix,ids=candidate_encoding(Tokenizer(),'Answer: ')
        self.assertEqual(prefix,[10])
        self.assertEqual(ids,[362,425,356,422])
    def test_incompatible_candidate_prefixes_fail(self):
        from readout_domain_use_matched import candidate_encoding
        class Tokenizer:
            def encode(self,text,add_special_tokens=False):return [ord(text[-1]),1]
        with self.assertRaises(ValueError):candidate_encoding(Tokenizer(),'Answer: ')

if __name__=='__main__':unittest.main()
