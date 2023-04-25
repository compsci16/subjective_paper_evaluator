import unittest
from typing import List

import similarity
import embeddings


class TestSimilarity(unittest.TestCase):
    def test_jaccard(self):
        s1: str = "Data is the new oil of the digital economy"
        s2: str = "Data is a new oil"
        result: float = similarity.jaccard(s1, s2)
        self.assertAlmostEqual(0.444, result, 3)

    def test_cosine(self):
        e1: List[float] = [2, 1, 2, 3, 2, 9]
        e2: List[float] = [3, 4, 2, 4, 5, 5]
        result = similarity.cosine(e1, e2)
        self.assertAlmostEqual(0.81885, result, 4)

    def test_manhattan(self):
        block1 = [1, 2, 3, 4]
        block2 = [5, 6, 7, 8]
        result = similarity.manhattan(block1, block2)
        self.assertEqual(16, result)

    def test_euclidean(self):
        block1 = [1, 2, 3, 4]
        block2 = [5, 6, 7, 8]
        result = similarity.euclidean(block1, block2)
        self.assertAlmostEqual(8, result, 4)


if __name__ == '__main__':
    unittest.main()
