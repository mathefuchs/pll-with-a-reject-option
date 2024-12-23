""" Module for testing DST-PLL. """

import unittest

from partial_label_learning.relatedwork.dst_pll_2024 import (
    CandidateLabelsEncoder, yager_combine)


class DstTest(unittest.TestCase):
    """ Unit test for DST-PLL. """

    def test_dst_yager_combination(self):
        """ Tests the Yager combination rule. """

        dst_a = {3: 0.5, 7: 0.5}
        dst_b = {5: 0.5, 7: 0.5}
        dst_c = {4: 0.5, 7: 0.5}
        res = yager_combine([dst_a, dst_b, dst_c], 7)
        self.assertEqual(
            res, {1: 0.125, 3: 0.125, 4: 0.25, 5: 0.125, 7: 0.375},
        )

    def test_dst_yager_combination_with_universe(self):
        """ Tests the Yager combination rule. """

        dst_a = {3: 0.5, 7: 0.5}
        dst_b = {5: 0.5, 7: 0.5}
        dst_identity = {7: 1.0}
        res1 = yager_combine([dst_a, dst_b], 7)
        res2 = yager_combine([
            dst_identity, dst_a, dst_identity, dst_b,
            dst_identity, dst_identity, dst_identity,
        ], 7)
        self.assertEqual(res1, res2)

    def test_dst_yager_combination_high_disagreement(self):
        """ Tests the Yager combination rule with high disagreement. """

        dst_a = {1: 0.99, 2: 0.01}
        dst_b = {2: 0.01, 4: 0.99}
        res = yager_combine([dst_a, dst_b], 7)
        self.assertEqual(
            res, {2: 0.0001, 7: 0.9999},
        )

    def test_dst_yager_combination_disjoint(self):
        """ Tests the Yager combination rule. """

        dst_a = {3: 0.5, 7: 0.5}
        dst_b = {5: 0.5, 7: 0.5}
        dst_c = {4: 1.0}
        res = yager_combine([dst_a, dst_b, dst_c], 7)
        self.assertEqual(
            res, {4: 0.5, 7: 0.5},
        )

    def test_label_encoder(self):
        """ Tests the label encoder. """

        encoder = CandidateLabelsEncoder(10)
        for num in range(1 << 10):
            self.assertEqual(
                num, encoder.encode_candidate_list(
                    encoder.decode_candidate_list(num)))


if __name__ == "__main__":
    unittest.main()
