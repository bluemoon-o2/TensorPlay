import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp


class TestMaxMin(unittest.TestCase):
    def test_max_all(self):
        t = tp.tensor([[1, 2, 3], [4, 5, 0]])
        m = t.max()
        self.assertEqual(m.item(), 5)
        
    def test_max_dim(self):
        t = tp.tensor([[1, 2, 3], [4, 5, 0]])
        # dim 0: max([1,4]), max([2,5]), max([3,0]) -> [4, 5, 3]
        m = t.max(dim=0)
        self.assertEqual(m.shape, [3])
        self.assertEqual(m[0].item(), 4)
        self.assertEqual(m[1].item(), 5)
        self.assertEqual(m[2].item(), 3)
        
        # dim 1: max([1,2,3])=3, max([4,5,0])=5
        m2 = t.max(dim=-1)
        self.assertEqual(m2.shape, [2])
        self.assertEqual(m2[0].item(), 3)
        self.assertEqual(m2[1].item(), 5)

    def test_min_all(self):
        t = tp.tensor([[1, 2, 3], [4, 5, 0]])
        m = t.min()
        self.assertEqual(m.item(), 0)
        
    def test_min_dim(self):
        t = tp.tensor([[1, 2, 3], [4, 5, 0]])
        # dim 0: min([1,4])=1, min([2,5])=2, min([3,0])=0
        m = t.min(dim=0)
        self.assertEqual(m[0].item(), 1)
        self.assertEqual(m[1].item(), 2)
        self.assertEqual(m[2].item(), 0)

    def test_max_keepdim(self):
        t = tp.tensor([[1, 2], [3, 4]])
        m = t.max(dim=0, keepdim=True)
        self.assertEqual(m.shape, [1, 2])
        
    def test_min_float(self):
        t = tp.tensor([1.5, -2.5, 3.0])
        self.assertEqual(t.max().item(), 3.0)
        self.assertEqual(t.min().item(), -2.5)

    def test_multiple_dims(self):
        # Shape [2, 2, 2]
        t = tp.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        # max over dim (0, 1) -> max over 2x2 blocks -> [max(1,3,5,7), max(2,4,6,8)] = [7, 8]
        # Wait, indices:
        # t[0,0,0]=1, t[1,0,0]=5, t[0,1,0]=3, t[1,1,0]=7 -> max is 7
        # t[0,0,1]=2, t[1,0,1]=6, t[0,1,1]=4, t[1,1,1]=8 -> max is 8
        m = t.max(dim=[0, 1])
        self.assertEqual(m.shape, [2])
        self.assertEqual(m[0].item(), 7)
        self.assertEqual(m[1].item(), 8)

if __name__ == '__main__':
    unittest.main()
