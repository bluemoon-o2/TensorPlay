import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorplay as tp
import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 定义经典优化测试函数（tensorplay实现，支持自动求导）
# ----------------------
def quadratic_function(x):
    """二次函数（单峰凸函数）：f(x) = x₁² + 2x₂²，最小值0，最优解(0,0)"""
    return x[0]**2 + 2 * x[1]**2

def rosenbrock_function(x):
    """Rosenbrock函数（香蕉函数，非凸）：f(x) = (1-x₁)² + 100(x₂-x₁²)²，最小值0，最优解(1,1)"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def ackley_function(x):
    """Ackley函数（多峰）：f(x) = -20exp(-0.2√(0.5(x₁²+x₂²))) - exp(0.5(cos2πx₁+cos2πx₂)) + 20 + e，最小值0，最优解(0,0)"""
    term1 = -20 * tp.exp(-0.2 * tp.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    term2 = -tp.exp(0.5 * (tp.cos(2 * np.pi * x[0]) + tp.cos(2 * np.pi * x[1])))
    return term1 + term2 + 20 + np.e

def rastrigin_function(x):
    """Rastrigin函数（多峰+噪声）：f(x) = A*n + Σ(xᵢ² - A cos2πxᵢ)，A=10，n=2，最小值0，最优解(0,0)"""
    A = 10
    return A * 2 + (x[0]**2 - A * tp.cos(2 * np.pi * x[0])) + (x[1]**2 - A * tp.cos(2 * np.pi * x[1]))

# ----------------------
# 2. 手动实现梯度下降优化器
# ----------------------
def gradient_descent(
    func,          # 优化目标函数（输入tensor列表，输出标量tensor）
    init_params,   # 初始参数（list of tensor，每个tensor需requires_grad=True）
    lr=0.001,      # 学习率
    max_iter=10000,# 最大迭代次数
    tol=1e-6,      # 收敛阈值（函数值变化<tol则停止）
    clip_grad=1.0, # 梯度裁剪阈值（避免梯度爆炸）
    momentum=0.0   # 动量
):
    """
    使用SGD优化器
    返回：参数历史、函数值历史
    """
    params = init_params.copy()

    # Create Optimizer
    optimizer = tp.optim.SGD(params, lr=lr, momentum=momentum)

    param_history = [np.array([p.item() for p in params])]  # 记录参数轨迹
    loss_history = []                                       # 记录函数值轨迹

    prev_loss = float('inf')
    for i in range(max_iter):
        # 1. Zero grad
        optimizer.zero_grad()

        # 2. Forward
        loss = func(params)
        loss_val = loss.item()
        loss_history.append(loss_val)

        # 3. Convergence Check
        if abs(loss_val - prev_loss) < tol or loss_val < 1e-5:
            print(f"Converged at iter {i}, final loss: {loss_val:.8f}")
            break
        prev_loss = loss_val

        # 4. Backward
        loss.backward()

        # 5. Gradient Clipping (manual, before step)
        if clip_grad > 0:
            for p in params:
                if p.grad is not None:
                    p.grad.data = tp.clamp(p.grad.data, -clip_grad, clip_grad)

        # 6. Step
        optimizer.step()

        # 7. Record history
        param_history.append(np.array([p.item() for p in params]))

    else:
        print(f"Warning: Max iterations reached without full convergence. Final loss: {loss_val:.8f}")

    return np.array(param_history), np.array(loss_history)

# ----------------------
# 3. 测试函数（含可视化）
# ----------------------
def test_quadratic():
    """测试二次函数（单峰凸函数，梯度下降易收敛）"""
    print("\n=== Testing Quadratic Function ===")
    # 初始参数（远离最优解(0,0)）
    init_x1 = tp.Tensor([3.0], requires_grad=True)
    init_x2 = tp.Tensor([-2.0], requires_grad=True)
    init_params = [init_x1, init_x2]

    # 梯度下降优化（学习率可较大，凸函数收敛快）
    param_history, loss_history = gradient_descent(
        func=quadratic_function,
        init_params=init_params,
        lr=0.1,
        max_iter=100,
        tol=1e-8
    )

    # 验证结果
    final_x1, final_x2 = param_history[-1]
    print(f"Final params: x1={final_x1:.8f}, x2={final_x2:.8f}")
    print(f"Theoretical optimal: (0,0), Final loss: {loss_history[-1]:.8f}")

    # 可视化：收敛曲线 + 参数轨迹
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    # # 收敛曲线
    # ax1.plot(loss_history, label='Function Value')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Loss')
    # ax1.set_title('Quadratic Function: Convergence Curve')
    # ax1.legend()
    # ax1.grid(True)
    # # 参数轨迹（2D）
    # ax2.plot(param_history[:,0], param_history[:,1], 'b-o', markersize=3, label='Param Trajectory')
    # ax2.scatter(0, 0, c='r', s=50, label='Optimal (0,0)')
    # ax2.set_xlabel('x1')
    # ax2.set_ylabel('x2')
    # ax2.set_title('Param Update Trajectory')
    # ax2.legend()
    # ax2.grid(True)
    # plt.show()

def test_rosenbrock():
    """测试Rosenbrock函数（非凸，梯度下降需小学习率）"""
    print("\n=== Testing Rosenbrock Function ===")
    # 初始参数（远离最优解(1,1)）
    init_x1 = tp.Tensor([2.0], requires_grad=True)
    init_x2 = tp.Tensor([3.0], requires_grad=True)
    init_params = [init_x1, init_x2]

    # 梯度下降优化（非凸函数，学习率需小，迭代次数需多）
    # Add momentum
    param_history, loss_history = gradient_descent(
        func=rosenbrock_function,
        init_params=init_params,
        lr=0.0003,
        max_iter=50000,
        tol=1e-7,
        clip_grad=5.0,
        momentum=0.9
    )

    # 验证结果
    final_x1, final_x2 = param_history[-1]
    print(f"Final params: x1={final_x1:.8f}, x2={final_x2:.8f}")
    print(f"Theoretical optimal: (1,1), Final loss: {loss_history[-1]:.8f}")
    sys.stdout.flush()

    # 可视化
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    # ax1.plot(loss_history, label='Function Value')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Loss')
    # ax1.set_title('Rosenbrock Function: Convergence Curve')
    # ax1.legend()
    # ax1.grid(True)
    # # 参数轨迹
    # ax2.plot(param_history[:,0], param_history[:,1], 'b-o', markersize=2, label='Param Trajectory')
    # ax2.scatter(1, 1, c='r', s=50, label='Optimal (1,1)')
    # ax2.set_xlabel('x1')
    # ax2.set_ylabel('x2')
    # ax2.set_title('Param Update Trajectory')
    # ax2.legend()
    # ax2.grid(True)
    # plt.show()

def test_ackley():
    """测试Ackley函数（多峰，梯度下降可能陷入局部最优）"""
    print("\n=== Testing Ackley Function ===")
    sys.stdout.flush()
    # 初始参数（远离最优解(0,0)）
    init_x1 = tp.Tensor([2.0], requires_grad=True)
    init_x2 = tp.Tensor([-1.5], requires_grad=True)
    init_params = [init_x1, init_x2]

    # 梯度下降优化（多峰函数，学习率适中）
    # Add momentum and use better init
    init_x1 = tp.Tensor([0.5], requires_grad=True)
    init_x2 = tp.Tensor([-0.5], requires_grad=True)
    init_params = [init_x1, init_x2]

    param_history, loss_history = gradient_descent(
        func=ackley_function,
        init_params=init_params,
        lr=0.01,
        max_iter=10000,
        tol=1e-7,
        clip_grad=2.0,
        momentum=0.9
    )

    # 验证结果
    final_x1, final_x2 = param_history[-1]
    print(f"Final params: x1={final_x1:.8f}, x2={final_x2:.8f}")
    print(f"Theoretical optimal: (0,0), Final loss: {loss_history[-1]:.8f}")

    # 可视化
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    # ax1.plot(loss_history, label='Function Value')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Loss')
    # ax1.set_title('Ackley Function: Convergence Curve')
    # ax1.legend()
    # ax1.grid(True)
    # # 参数轨迹
    # ax2.plot(param_history[:,0], param_history[:,1], 'b-o', markersize=2, label='Param Trajectory')
    # ax2.scatter(0, 0, c='r', s=50, label='Optimal (0,0)')
    # ax2.set_xlabel('x1')
    # ax2.set_ylabel('x2')
    # ax2.set_title('Param Update Trajectory')
    # ax2.legend()
    # ax2.grid(True)
    # plt.show()

def test_rastrigin():
    """测试Rastrigin函数（多峰+噪声，梯度下降挑战最大）"""
    print("\n=== Testing Rastrigin Function ===")
    # 初始参数（远离最优解(0,0)）
    init_x1 = tp.Tensor([1.2], requires_grad=True)
    init_x2 = tp.Tensor([-0.8], requires_grad=True)
    init_params = [init_x1, init_x2]

    # Add momentum and use better init
    init_x1 = tp.Tensor([0.4], requires_grad=True)
    init_x2 = tp.Tensor([-0.4], requires_grad=True)
    init_params = [init_x1, init_x2]

    param_history, loss_history = gradient_descent(
        func=rastrigin_function,
        init_params=init_params,
        lr=0.001,
        max_iter=20000,
        tol=1e-7,
        clip_grad=3.0,
        momentum=0.9
    )

    # 验证结果
    final_x1, final_x2 = param_history[-1]
    print(f"Final params: x1={final_x1:.8f}, x2={final_x2:.8f}")
    print(f"Theoretical optimal: (0,0), Final loss: {loss_history[-1]:.8f}")

    # 可视化
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    # ax1.plot(loss_history, label='Function Value')
    # ax1.set_xlabel('Iteration')
    # ax1.set_ylabel('Loss')
    # ax1.set_title('Rastrigin Function: Convergence Curve')
    # ax1.legend()
    # ax1.grid(True)
    # # 参数轨迹
    # ax2.plot(param_history[:,0], param_history[:,1], 'b-o', markersize=2, label='Param Trajectory')
    # ax2.scatter(0, 0, c='r', s=50, label='Optimal (0,0)')
    # ax2.set_xlabel('x1')
    # ax2.set_ylabel('x2')
    # ax2.set_title('Param Update Trajectory')
    # ax2.legend()
    # ax2.grid(True)
    # plt.show()


if __name__ == "__main__":
    test_quadratic()
    test_rosenbrock()
    test_ackley()
    test_rastrigin()
    print("\nAll optimization tests finished!")
