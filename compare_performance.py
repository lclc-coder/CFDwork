import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from grid import StaggeredGrid
from solver import NavierStokesSolver, kolmogorov_forcing
from learned_interpolation import (
    LearnedInterpolation, create_model, initialize_model
)

def load_model():
    """加载LI模型"""
    model_config = {
        'hidden_features': 64,
        'num_layers': 6,
        'output_features': 120,
        'stencil_size': 4,
        'num_interp_points': 8
    }
    
    # 创建模型
    model = create_model(
        hidden_features=model_config['hidden_features'],
        num_layers=model_config['num_layers'],
        output_features=model_config['output_features']
    )
    model.stencil_size = model_config['stencil_size']
    model.num_interp_points = model_config['num_interp_points']
    
    # 初始化模型
    key = jax.random.PRNGKey(0)
    input_shape = (1, 128, 128, 2)
    params = initialize_model(model, input_shape, key)
    
    return model, params

def run_basic_simulation(steps=1000):
    """运行基本CFD求解器"""
    # 创建网格
    grid = StaggeredGrid(nx=128, ny=128, domain_size=(2*np.pi, 2*np.pi))
    
    # 创建求解器
    solver = NavierStokesSolver(
        grid=grid,
        Re=1000,
        cfl=0.5,
        force_fn=kolmogorov_forcing
    )
    
    # 设置初始条件 - Kolmogorov流
    u, v, p = grid.create_kolmogorov_flow(A=1.0, k=4)
    key = jax.random.PRNGKey(42)
    u += 0.01 * jax.random.normal(key, u.shape)
    v += 0.01 * jax.random.normal(jax.random.split(key)[0], v.shape)

    # 运行模拟并计时
    start_time = time.time()
    for step in range(steps):
        u, v, p, dt = solver.step(u, v, p, use_basic_advection=True)
    end_time = time.time()
    
    basic_time = end_time - start_time
    print(f"基本求解器运行{steps}步用时: {basic_time:.2f}秒")
    
    return basic_time

def run_li_simulation(model, params, steps=1000):
    """运行LI增强求解器"""
    # 创建网格
    grid = StaggeredGrid(nx=128, ny=128, domain_size=(2*np.pi, 2*np.pi))
    
    # 创建求解器
    solver = NavierStokesSolver(
        grid=grid,
        Re=1000,
        cfl=0.5,
        force_fn=kolmogorov_forcing
    )
    
    # 创建LI模型
    li = LearnedInterpolation(
        model, params, 
        stencil_size=model.stencil_size, 
        num_interp_points=model.num_interp_points,
        scale_factor=1.0
    )
    
    # 修改求解器使用LI计算对流项
    solver.advection_fn = li
    
    # 设置初始条件 - Kolmogorov流
    u, v, p = grid.create_kolmogorov_flow(A=1.0, k=4)
    key = jax.random.PRNGKey(42)
    u += 0.01 * jax.random.normal(key, u.shape)
    v += 0.01 * jax.random.normal(jax.random.split(key)[0], v.shape)

    # 运行模拟并计时
    start_time = time.time()
    for step in range(steps):
        u, v, p, dt = solver.step(u, v, p, use_basic_advection=False)
    end_time = time.time()
    
    li_time = end_time - start_time
    print(f"LI增强求解器运行{steps}步用时: {li_time:.2f}秒")
    
    return li_time

def main():
    # 加载模型
    model, params = load_model()
    
    # 运行基本求解器
    basic_time = run_basic_simulation(steps=1000)
    
    # 运行LI增强求解器
    li_time = run_li_simulation(model, params, steps=1000)
    
    # 计算加速比
    speedup = basic_time / li_time
    print(f"LI增强求解器相比基本求解器加速了 {speedup:.2f} 倍")
    
    # 保存结果
    with open("performance_results.txt", "w") as f:
        f.write(f"基本求解器运行1000步用时: {basic_time:.2f}秒\n")
        f.write(f"LI增强求解器运行1000步用时: {li_time:.2f}秒\n")
        f.write(f"LI增强求解器相比基本求解器加速了 {speedup:.2f} 倍\n")
    
    # 绘制对比图 - 使用英文标签避免字体问题
    labels = ['Basic Solver', 'LI-enhanced Solver']
    times = [basic_time, li_time]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['blue', 'orange'])
    plt.ylabel('Runtime (seconds)')
    plt.title('Solver Performance Comparison (1000 steps)')
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(times):
        plt.text(i, v + 0.01, f"{v:.2f}s", ha='center')
    
    # 添加加速比注释
    plt.text(0.5, max(times) * 0.5, f"Speedup: {speedup:.2f}x", 
             ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150)
    print("结果已保存到 performance_results.txt 和 performance_comparison.png")

if __name__ == "__main__":
    main() 