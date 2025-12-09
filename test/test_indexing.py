import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp

class TestIndexing(unittest.TestCase):
    def test_basic_indexing(self):
        t = tp.arange(12).reshape([3, 4])
        # t is [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        
        # Select row
        self.assertEqual(t[0].shape, (4,))
        self.assertEqual(t[0, 0].item(), 0)
        
        # Select element
        self.assertEqual(t[1, 2].item(), 6)
        
    def test_slicing(self):
        t = tp.arange(12).reshape([3, 4])
        
        # Slice rows
        s = t[0:2] # [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.assertEqual(s.shape, (2, 4))
        self.assertEqual(s[0, 0].item(), 0)
        self.assertEqual(s[1, 3].item(), 7)
        
        # Slice cols
        s = t[:, 1:3] # [[1, 2], [5, 6], [9, 10]]
        self.assertEqual(s.shape, (3, 2))
        self.assertEqual(s[0, 0].item(), 1)
        self.assertEqual(s[2, 1].item(), 10)
        
    def test_mixed_indexing(self):
        t = tp.arange(24).reshape([2, 3, 4])
        
        # Slice, Select, Slice
        # t[0:2, 1, 1:3]
        # Dim 0: slice 0:2 (keep dim) -> shape [2, 3, 4]
        # Dim 1: select 1 (remove dim) -> shape [2, 4]
        # Dim 2: slice 1:3 (keep dim) -> shape [2, 2]
        s = t[0:2, 1, 1:3]
        self.assertEqual(s.shape, (2, 2))
        
        # Check values
        # t[0, 1, 1] = 5
        # t[0, 1, 2] = 6
        # t[1, 1, 1] = 17
        # t[1, 1, 2] = 18
        
        self.assertEqual(s[0, 0].item(), 5)
        self.assertEqual(s[0, 1].item(), 6)
        self.assertEqual(s[1, 0].item(), 17)
        self.assertEqual(s[1, 1].item(), 18)

    def test_advanced_slicing(self):
        t = tp.arange(10)
        
        # Step
        s = t[0:10:2]
        self.assertEqual(s.shape, (5,))
        self.assertEqual(s[0].item(), 0)
        self.assertEqual(s[1].item(), 2)
        
        # Negative step (not supported yet?)
        # s = t[::-1]
        
    def test_setitem(self):
        t = tp.zeros([3, 4])
        
        # Set row
        t[0] = tp.ones([4])
        self.assertEqual(t[0, 0].item(), 1)
        self.assertEqual(t[1, 0].item(), 0)
        
        # Set scalar
        t[1, 1] = 5
        self.assertEqual(t[1, 1].item(), 5)
        
        # Set slice
        t[2, 0:2] = tp.tensor([8, 9]).reshape([2]) # Need to match shape?
        # tp.tensor([8, 9]) is [2]. slice is [2]. Matches.
        self.assertEqual(t[2, 0].item(), 8)
        self.assertEqual(t[2, 1].item(), 9)
        
        # Set scalar to slice
        t[2, 2:] = 10
        self.assertEqual(t[2, 2].item(), 10)
        self.assertEqual(t[2, 3].item(), 10)

if __name__ == '__main__':
    unittest.main()
