import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from pathlib import Path
import argparse

from grid import StaggeredGrid, downsample_velocity_field
from solver import NavierStokesSolver, kolmogorov_forcing
from learned_interpolation import (
    LearnedInterpolation, create_model, initialize_model
)

def run_simulation(grid, solver, num_steps, initial_u=None, initial_v=None, 
                  use_li=False, li_model=None, li_params=None, 
                  save_interval=10, output_dir='./results'):
    """
    运行模拟并保存结果
    
    参数:
        grid: 交错网格对象
        solver: 求解器对象
        num_steps: 模拟步数
        initial_u, initial_v: 初始速度场，如果为None则使用Kolmogorov流
        use_li: 是否使用LI模型
        li_model: LI模型
        li_params: LI模型参数
        save_interval: 保存间隔
        output_dir: 输出目录
        
    返回:
        snapshots: 模拟结果快照列表
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置初始条件
    if initial_u is None or initial_v is None:
        u, v, p = grid.create_kolmogorov_flow(A=1.0, k=4)
        
        # 添加随机扰动
        key = jax.random.PRNGKey(42)
        u += 0.01 * jax.random.normal(key, u.shape)
        v += 0.01 * jax.random.normal(jax.random.split(key)[0], v.shape)
    else:
        u, v = initial_u, initial_v
        p = jnp.zeros_like(u)
    
    # 如果使用LI，创建LI模型
    if use_li and li_model is not None and li_params is not None:
        li = LearnedInterpolation(
            li_model, li_params, 
            stencil_size=li_model.stencil_size, 
            num_interp_points=li_model.num_interp_points,
            scale_factor=1.0
        )
        
        # 修改求解器使用LI计算对流项
        solver.advection_fn = li
    
    # 保存快照
    snapshots = []
    
    # 运行模拟
    total_time = 0.0
    for step in range(num_steps):
        start_time = time.time()
        
        # 执行一步模拟
        u, v, p, dt = solver.step(u, v, p, use_basic_advection=not use_li)
        
        step_time = time.time() - start_time
        total_time += step_time
        
        # 保存快照
        if step % save_interval == 0:
            snapshots.append((u.copy(), v.copy(), p.copy()))
            
            # 输出信息
            print(f"Step {step+1}/{num_steps}, Time: {step_time:.4f}s, Total: {total_time:.2f}s")
            
            # 保存数据
            np.savez(
                os.path.join(output_dir, f"step_{step:06d}.npz"),
                u=np.array(u),
                v=np.array(v),
                p=np.array(p)
            )
    
    return snapshots

def visualize_velocity_field(u, v, grid, save_path=None, title=None):
    """
    可视化速度场
    
    参数:
        u, v: 速度分量
        grid: 网格对象
        save_path: 保存路径，如果为None则显示图像
        title: 图像标题
    """
    # 将速度从交错网格插值到中心
    uc, vc = grid.interpolate_to_cell_centers(u, v)
    
    # 计算速度幅值
    speed = jnp.sqrt(uc**2 + vc**2)
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制速度幅值的伪彩色图
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
    
    # 设置标题和标签
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def visualize_vorticity(u, v, grid, dx, dy, save_path=None, title=None):
    """
    可视化涡量场
    
    参数:
        u, v: 速度分量
        grid: 网格对象
        dx, dy: 网格间距
        save_path: 保存路径，如果为None则显示图像
        title: 图像标题
    """
    # 计算涡量 ω = ∂v/∂x - ∂u/∂y
    dvdx = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * dx)
    dudy = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * dy)
    vorticity = dvdx - dudy
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制涡量场
    im = ax.imshow(vorticity, cmap='RdBu_r', origin='lower',
                  extent=[0, grid.Lx, 0, grid.Ly])
    plt.colorbar(im, ax=ax, label='Vorticity')
    
    # 设置标题和标签
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def create_animation(snapshots, grid, output_path='animation.gif', fps=15):
    """
    创建动画
    
    参数:
        snapshots: 模拟结果快照列表，每个快照是(u, v, p)元组
        grid: 网格对象
        output_path: 输出路径
        fps: 帧率
    """
    # 计算每个快照的涡量
    vorticity_list = []
    for u, v, _ in snapshots:
        dvdx = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / (2 * grid.dx)
        dudy = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * grid.dy)
        vorticity = dvdx - dudy
        vorticity_list.append(np.array(vorticity))
    
    # 找到涡量的最大和最小值，用于归一化颜色映射
    vmin = min(jnp.min(v) for v in vorticity_list)
    vmax = max(jnp.max(v) for v in vorticity_list)
    
    # 创建图像和动画
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        ax.clear()
        vorticity = vorticity_list[frame]
        im = ax.imshow(vorticity, cmap='RdBu_r', origin='lower',
                      extent=[0, grid.Lx, 0, grid.Ly],
                      vmin=vmin, vmax=vmax)
        ax.set_title(f'Frame {frame}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return [im]
    
    # 创建动画
    anim = animation.FuncAnimation(
        fig, update, frames=len(vorticity_list), blit=True
    )
    
    # 保存动画
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()

def load_model(model_path, model_config):
    """
    加载LI模型
    
    参数:
        model_path: 模型路径
        model_config: 模型配置
        
    返回:
        model: LI模型
        params: 模型参数
    """
    # 创建模型
    model = create_model(
        hidden_features=model_config['hidden_features'],
        num_layers=model_config['num_layers'],
        output_features=model_config['output_features']
    )
    model.stencil_size = model_config['stencil_size']
    model.num_interp_points = model_config['num_interp_points']
    
    # 加载参数
    params = np.load(model_path)
    
    return model, params

def compare_simulations(grid_low, num_steps=1000, save_dir='./comparison'):
    """
    比较基础求解器和LI增强求解器的性能
    
    参数:
        grid_low: 低分辨率网格
        num_steps: 模拟步数
        save_dir: 保存目录
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建求解器
    force_fn = lambda u, v: kolmogorov_forcing(u, v, k=4, drag_coef=0.1)
    solver_basic = NavierStokesSolver(grid_low, Re=1000, force_fn=force_fn)
    solver_li = NavierStokesSolver(grid_low, Re=1000, force_fn=force_fn)
    
    # 创建相同的初始条件
    key = jax.random.PRNGKey(42)
    u0, v0, p0 = grid_low.create_kolmogorov_flow(A=1.0, k=4)
    u0 += 0.01 * jax.random.normal(key, u0.shape)
    v0 += 0.01 * jax.random.normal(jax.random.split(key)[0], v0.shape)
    
    # 加载LI模型
    model_config = {
        'hidden_features': 64,
        'num_layers': 6,
        'output_features': 120,
        'stencil_size': 4,
        'num_interp_points': 8
    }
    model_path = './models/li_model_best.npz'
    
    if os.path.exists(model_path):
        li_model, li_params = load_model(model_path, model_config)
        
        # 创建LI模型
        li = LearnedInterpolation(
            li_model, li_params, 
            stencil_size=li_model.stencil_size, 
            num_interp_points=li_model.num_interp_points,
            scale_factor=1.0
        )
        
        # 运行基础求解器
        print("运行基础求解器...")
        start_time = time.time()
        snapshots_basic = run_simulation(
            grid_low, solver_basic, num_steps, 
            initial_u=u0, initial_v=v0,
            use_li=False, 
            save_interval=100, 
            output_dir=os.path.join(save_dir, 'basic')
        )
        basic_time = time.time() - start_time
        print(f"基础求解器耗时: {basic_time:.2f}s")
        
        # 运行LI增强求解器
        print("运行LI增强求解器...")
        start_time = time.time()
        snapshots_li = run_simulation(
            grid_low, solver_li, num_steps, 
            initial_u=u0, initial_v=v0,
            use_li=True, li_model=li_model, li_params=li_params,
            save_interval=100, 
            output_dir=os.path.join(save_dir, 'li')
        )
        li_time = time.time() - start_time
        print(f"LI增强求解器耗时: {li_time:.2f}s")
        
        # 比较最终结果
        u_basic, v_basic, p_basic = snapshots_basic[-1]
        u_li, v_li, p_li = snapshots_li[-1]
        
        # 计算速度差异
        speed_basic = jnp.sqrt(u_basic**2 + v_basic**2)
        speed_li = jnp.sqrt(u_li**2 + v_li**2)
        speed_diff = jnp.abs(speed_li - speed_basic)
        
        # 可视化差异
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im0 = axes[0].imshow(speed_basic, origin='lower', cmap='viridis')
        axes[0].set_title('Basic Solver')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(speed_li, origin='lower', cmap='viridis')
        axes[1].set_title('LI Solver')
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(speed_diff, origin='lower', cmap='hot')
        axes[2].set_title('Difference')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
        plt.close()
        
        # 输出性能比较
        speedup = basic_time / li_time
        print(f"加速比: {speedup:.2f}x")
        
        # 创建动画
        create_animation(
            snapshots_basic, grid_low, 
            output_path=os.path.join(save_dir, 'basic_animation.gif')
        )
        create_animation(
            snapshots_li, grid_low, 
            output_path=os.path.join(save_dir, 'li_animation.gif')
        )
    else:
        print(f"模型文件 {model_path} 不存在，请先训练模型")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='CFD模拟程序')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'simulate', 'compare'],
                        help='运行模式：训练模型，运行模拟，或比较性能')
    parser.add_argument('--nx', type=int, default=64, help='x方向网格数')
    parser.add_argument('--ny', type=int, default=64, help='y方向网格数')
    parser.add_argument('--Re', type=float, default=1000, help='雷诺数')
    parser.add_argument('--steps', type=int, default=1000, help='模拟步数')
    parser.add_argument('--model', type=str, default='./models/li_model_best.npz', 
                        help='模型文件路径')
    parser.add_argument('--output', type=str, default='./results', 
                        help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 导入训练模块
        from training import main as train_main
        train_main()
    
    elif args.mode == 'simulate':
        # 创建网格和求解器
        grid = StaggeredGrid(args.nx, args.ny)
        force_fn = lambda u, v: kolmogorov_forcing(u, v, k=4, drag_coef=0.1)
        solver = NavierStokesSolver(grid, Re=args.Re, force_fn=force_fn)
        
        # 决定是否使用LI模型
        use_li = os.path.exists(args.model)
        li_model, li_params = None, None
        
        if use_li:
            # 加载LI模型
            model_config = {
                'hidden_features': 64,
                'num_layers': 6,
                'output_features': 120,
                'stencil_size': 4,
                'num_interp_points': 8
            }
            li_model, li_params = load_model(args.model, model_config)
            print(f"已加载LI模型: {args.model}")
        
        # 运行模拟
        snapshots = run_simulation(
            grid, solver, args.steps, 
            use_li=use_li, li_model=li_model, li_params=li_params,
            save_interval=10, 
            output_dir=args.output
        )
        
        # 创建动画
        create_animation(
            snapshots, grid, 
            output_path=os.path.join(args.output, 'animation.gif')
        )
    
    elif args.mode == 'compare':
        # 创建网格
        grid = StaggeredGrid(args.nx, args.ny)
        # 比较模拟
        compare_simulations(grid, num_steps=args.steps, save_dir=args.output)

if __name__ == "__main__":
    main() 