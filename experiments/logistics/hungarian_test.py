#!/usr/bin/env python3
import unittest
from hungarian import hungarian
import numpy as np

class TestHungarian(unittest.TestCase):
    def test_example(self):
        M = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
        ])
        res = hungarian(M)
        self.assertEqual(res, [
            (2, 0),
            (1, 1),
            (0, 2),
        ])

if __name__ == "__main__":
    unittest.main()