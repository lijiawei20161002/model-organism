import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
AVAILABLE = importlib.util.find_spec('scipy') is not None
if AVAILABLE:
    from analyze_b200_uncertainty import exact_family_interval, compare, pooled


@unittest.skipUnless(AVAILABLE, 'requires scipy for the uncertainty analysis')
class FamilyBootstrapTests(unittest.TestCase):
    def test_convolution_matches_brute_force_family_resampling(self):
        values = [-2, 1, 4]
        bootstrap = np.array([100*np.mean(x)/10 for x in itertools.product(values, repeat=3)])
        expected = np.quantile(bootstrap, [.025, .975], method='inverted_cdf')
        actual = exact_family_interval(values, 10)
        np.testing.assert_allclose(actual['ci95_pp'], expected)
        self.assertAlmostEqual(actual['delta_pp'], 10)

    def test_constant_family_effect_has_constant_interval(self):
        self.assertEqual(exact_family_interval([3]*8, 60)['ci95_pp'], [5, 5])

    def test_fractional_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'integers'):
            exact_family_interval([1.2, 3], 10)

    def test_mismatched_prompt_text_rejected(self):
        a = [{'key': ('q',0,0), 'text': ('a',None), 'question': 'q', 'prompt': ('q',0),
              'coherent': 1, 'harmful': 0, 'acceptable': 1}]
        b = [{**a[0], 'text': ('different',None)}]
        with self.assertRaisesRegex(ValueError, 'Mismatched'):
            compare(a, b)

    def test_pooling_retains_distinct_seed_keys(self):
        a = [{'key': ('q',0,0), 'text': ('a',None), 'question': 'q', 'prompt': ('q',0),
              'coherent': 1, 'harmful': 0, 'acceptable': 1}]
        result = pooled([(201,[a,a]), (202,[a,a])])
        self.assertEqual(len({r['key'] for r in result[0]}), 2)
        self.assertEqual(compare(*result)['family_effects']['acceptable']['delta_pp'], 0)


if __name__ == '__main__':
    unittest.main()
