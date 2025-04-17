import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

from grid import StaggeredGrid
from solver import NavierStokesSolver, kolmogorov_forcing

def run_simulation(grid, solver, num_steps, save_interval=10, output_dir='./basic_results'):
    """
    运行模拟并保存结果
    
    参数:
        grid: 交错网格对象
        solver: 求解器对象
        num_steps: 模拟步数
        save_interval: 保存间隔
        output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建Kolmogorov流初始条件
    u, v, p = grid.create_kolmogorov_flow(A=1.0, k=4)
    
    # 添加随机扰动
    key = jax.random.PRNGKey(42)
    u += 0.01 * jax.random.normal(key, u.shape)
    v += 0.01 * jax.random.normal(jax.random.split(key)[0], v.shape)
    
    # 运行模拟
    total_time = 0.0
    for step in range(num_steps):
        start_time = time.time()
        
        # 执行一步模拟
        u, v, p, dt = solver.step(u, v, p)
        
        step_time = time.time() - start_time
        total_time += step_time
        
        # 保存结果
        if step % save_interval == 0:
            print(f"Step {step+1}/{num_steps}, Time: {step_time:.4f}s, Total: {total_time:.2f}s")
            
            # 计算涡量
            dvdx = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * grid.dx)
            dudy = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * grid.dy)
            vorticity = dvdx - dudy
            
            # 可视化涡量场
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(vorticity, cmap='RdBu_r', origin='lower',
                         extent=[0, grid.Lx, 0, grid.Ly])
            plt.colorbar(im, ax=ax, label='Vorticity')
            ax.set_title(f'Step {step+1}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.savefig(os.path.join(output_dir, f'vorticity_{step:04d}.png'), dpi=150)
            plt.close()
            
            # 可视化速度场
            uc, vc = grid.interpolate_to_cell_centers(u, v)
            speed = jnp.sqrt(uc**2 + vc**2)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(speed, cmap='viridis', origin='lower',
                         extent=[0, grid.Lx, 0, grid.Ly])
            plt.colorbar(im, ax=ax, label='Speed')
            
            # 绘制速度矢量场（降采样以提高可视化效果）
            skip = max(1, grid.nx // 30)
            x = jnp.linspace(0, grid.Lx, grid.nx)
            y = jnp.linspace(0, grid.Ly, grid.ny)
            X, Y = jnp.meshgrid(x, y)
            
            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                    uc[::skip, ::skip], vc[::skip, ::skip],
                    color='white', scale=50)
            
            ax.set_title(f'Step {step+1}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.savefig(os.path.join(output_dir, f'velocity_{step:04d}.png'), dpi=150)
            plt.close()
            
            # 保存数据
            np.savez(
                os.path.join(output_dir, f"data_{step:04d}.npz"),
                u=np.array(u),
                v=np.array(v),
                p=np.array(p),
                vorticity=np.array(vorticity)
            )

def main():
    # 设置参数
    nx, ny = 128, 128  # 使用更高分辨率
    Re = 1000
    num_steps = 200
    save_interval = 10
    
    # 创建网格和求解器
    grid = StaggeredGrid(nx, ny)
    force_fn = lambda u, v: kolmogorov_forcing(u, v, k=4, drag_coef=0.1)
    solver = NavierStokesSolver(grid, Re=Re, cfl=0.5, force_fn=force_fn)
    
    # 运行模拟
    print(f"Running simulation with grid size {nx}x{ny}, Re={Re}")
    run_simulation(grid, solver, num_steps, save_interval=save_interval)
    print("Simulation completed!")

if __name__ == "__main__":
    main() 