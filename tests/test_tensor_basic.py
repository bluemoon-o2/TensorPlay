import numpy as np
from TensorPlay import Tensor, sphere


def test_tensor_creation():
    # 测试基本的Tensor创建
    data = np.array([1.0, 2.0, 3.0])
    t = Tensor(data)
    
    # 检查数据是否正确存储
    assert np.array_equal(t.data, data)
    assert t.grad is None
    assert t.op is None
    


def test_tensor_arithmetic():
    # 测试基本算术操作
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([4.0, 5.0, 6.0]))
    
    # 测试加法
    c = a + b
    assert np.array_equal(c.data, np.array([5.0, 7.0, 9.0]))
    
    # 测试乘法
    d = a * b
    assert np.array_equal(d.data, np.array([4.0, 10.0, 18.0]))
    
    # 测试标量操作
    e = a + 2.0
    assert np.array_equal(e.data, np.array([3.0, 4.0, 5.0]))
    


def test_tensor_functions():
    # 测试内置函数
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tensor(np.array([1.0, 0.0, -1.0]))

    # 测试sphere函数
    z = sphere(x, y)
    assert np.array_equal(z.data, np.array([2.0, 0.0, 2.0]))
    
    # 测试relu方法
    z = x.relu()
    assert np.array_equal(z.data, np.array([0.0, 0.0, 1.0]))
    


def test_simple_backward():
    # 测试简单的反向传播
    x = Tensor(np.array([2.0]))
    y = sphere(x, x)
    
    # 手动设置梯度并反向传播
    y.backward()
    
    # 导数应为4.0 (d/dx x² = 2x, 在x=2时为4)
    assert np.array_equal(x.grad.data, np.array([8.0]))
    

if __name__ == "__main__":
    test_tensor_creation()
    test_tensor_arithmetic()
    test_tensor_functions()
    test_simple_backward()
    print("All tests passed!")