import sys
import os

# Add build directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Release')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Debug')))

try:
    import tensorplay
except ImportError:
    print("Could not import tensorplay. Please ensure it is built.")
    sys.exit(1)

def test_features():
    print("--- Testing Tensor Features ---")
    
    # 1. eye
    print("\n1. Testing eye:")
    eye = tensorplay.eye(3)
    print(eye)
    assert eye.shape == tensorplay.Size([3, 3])
    assert eye.dtype == tensorplay.DType.float32
    
    eye_rect = tensorplay.eye(2, 4)
    print(eye_rect)
    assert eye_rect.shape == tensorplay.Size([2, 4])
    
    # 2. rand
    print("\n2. Testing rand:")
    r = tensorplay.rand([2, 3])
    print(r)
    assert r.shape == tensorplay.Size([2, 3])
    assert r.dtype == tensorplay.DType.float32
    
    # 3. *_like
    print("\n3. Testing *_like:")
    z = tensorplay.zeros([2, 2])
    ol = tensorplay.ones_like(z)
    print("ones_like(zeros):", ol)
    assert ol.shape == z.shape
    assert ol.dtype == z.dtype
    
    fl = tensorplay.full_like(z, 3.14)
    print("full_like(zeros, 3.14):", fl)
    assert fl.shape == z.shape
    
    # 4. Broadcasting
    print("\n4. Testing Broadcasting:")
    a = tensorplay.ones([3, 1])
    b = tensorplay.ones([1, 4])
    c = a + b
    print(f"a shape: {a.shape}, b shape: {b.shape}, c = a + b shape: {c.shape}")
    print(c)
    assert c.shape == tensorplay.Size([3, 4])
    
    # 5. Slicing
    print("\n5. Testing Slicing:")
    t = tensorplay.arange(0, 10).reshape((2, 5))
    print("Original:", t)
    s = t[:, 1:4]
    print("Sliced [:, 1:4]:", s)
    assert s.shape == tensorplay.Size([2, 3])

    # 6. Recursive Print
    print("\n6. Testing Recursive Print:")
    t3d = tensorplay.arange(0, 24).reshape([2, 3, 4])
    print(t3d)
    
    # 7. Copy and Modification
    print("\n7. Testing Copy and Modification:")
    t_orig = tensorplay.ones([2, 2])
    t_copy = tensorplay.Tensor([2, 2]) # Should be empty or zeros depending on ctor? No, from list [2,2] -> tensor([2,2])
    # Wait, Tensor([2,2]) creates a 1D tensor with values 2, 2.
    # To create empty like, we use empty_like or empty
    t_copy = tensorplay.empty([2, 2])
    t_copy.copy_(t_orig)
    print("Copy:", t_copy)
    # Check values
    
    # 8. To (DType conversion)
    print("\n8. Testing To (DType):")
    t_float = tensorplay.ones([2, 2], dtype=tensorplay.DType.float32)
    t_int = t_float.to(tensorplay.DType.int32)
    print("Float:", t_float)
    print("Int:", t_int)
    assert t_int.dtype == tensorplay.DType.int32
    
    print("\n--- All Tests Passed ---")

if __name__ == "__main__":
    try:
        test_features()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
