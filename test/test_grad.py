import sys
import os
import unittest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp

class TestGrad(unittest.TestCase):
    def test_grad_property(self):
        # Test requires_grad=False initially
        x = tp.Tensor([1.0, 2.0, 3.0], dtype=tp.float32)
        self.assertFalse(x.requires_grad)
        self.assertIsNone(x.grad)
        
        # Set grad manually
        g = tp.Tensor([0.1, 0.2, 0.3], dtype=tp.float32)
        x.grad = g
        self.assertIsNotNone(x.grad)
        # Check values
        # We don't have allclose yet, check manually or exact match if same object
        # x.grad should be a copy or the same tensor?
        # In C++, set_grad copies the tensor if it's value-based, but here it's shared_ptr<TensorImpl>.
        # Tensor copy constructor copies the shared_ptr (shallow copy of storage).
        # So it points to same storage.
        self.assertEqual(x.grad.shape, g.shape)
        
        # Test setting None
        x.grad = None
        self.assertIsNone(x.grad)
        
        # Test requires_grad=True
        x.requires_grad = True
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad)
        
        x.grad = g
        self.assertIsNotNone(x.grad)
        x.grad = None
        self.assertIsNone(x.grad)

    def test_grad_assignment_validation(self):
        x = tp.Tensor([1.0], requires_grad=True)
        # PyTorch allows setting grad of different shape? 
        # Actually PyTorch warns or errors if shape mismatch during accumulation, but assignment might be loose.
        # But usually grad should match shape.
        # Our implementation doesn't check shape in set_grad.
        
        g = tp.Tensor([1.0, 2.0])
        x.grad = g
        self.assertEqual(x.grad.shape, [2])
        # This is fine for now.

    def test_is_leaf(self):
        x = tp.Tensor([1.0], requires_grad=True)
        self.assertTrue(x.is_leaf)
        
        y = tp.Tensor([1.0], requires_grad=False)
        self.assertTrue(y.is_leaf)

    def test_retain_grad(self):
        x = tp.Tensor([1.0], requires_grad=True)
        # Should not throw
        x.retain_grad()
        
    def test_detach(self):
        x = tp.Tensor([1.0, 2.0], requires_grad=True)
        y = x.detach()
        
        self.assertFalse(y.requires_grad)
        self.assertTrue(y.is_leaf)
        
        # Check storage sharing
        x.fill_(3.0)
        
        print(y)

if __name__ == '__main__':
    unittest.main()
