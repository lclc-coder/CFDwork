import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import checkify
import optax
from functools import partial
import numpy as np
import os
from pathlib import Path
import time

from grid import StaggeredGrid, downsample_velocity_field
from solver import NavierStokesSolver, kolmogorov_forcing
from learned_interpolation import (
    LearnedInterpolation, create_model, initialize_model, 
    scale_field, extract_local_neighborhood
)

def simulate_timesteps(solver, u, v, p, num_steps, use_basic_advection=True):
    """
    模拟多个时间步
    
    参数:
        solver: NavierStokesSolver 实例
        u, v, p: 初始速度和压力场
        num_steps: 时间步数
        use_basic_advection: 是否使用基础对流计算
        
    返回:
        u_final, v_final, p_final: 最终状态
        dt: 时间步长
    """
    u_curr, v_curr, p_curr = u, v, p
    
    for _ in range(num_steps):
        u_curr, v_curr, p_curr, dt = solver.step(
            u_curr, v_curr, p_curr, 
            use_basic_advection=use_basic_advection
        )
    
    return u_curr, v_curr, p_curr, dt

def generate_dns_data(nx_high, ny_high, num_snapshots, dt_factor=1.0, Re=1000):
    """
    生成高分辨率DNS模拟数据
    
    参数:
        nx_high, ny_high: 高分辨率网格尺寸
        num_snapshots: 快照数量
        dt_factor: 时间步长因子（用于加速模拟）
        Re: 雷诺数
        
    返回:
        snapshots: 速度场快照列表
    """
    # 创建高分辨率网格和求解器
    grid_high = StaggeredGrid(nx_high, ny_high)
    force_fn = lambda u, v: kolmogorov_forcing(u, v, k=4, drag_coef=0.1)
    solver_high = NavierStokesSolver(grid_high, Re=Re, force_fn=force_fn)
    
    # 创建初始条件
    u_high, v_high, p_high = grid_high.create_kolmogorov_flow(A=1.0, k=4)
    
    # 添加随机扰动以打破对称性
    key = jax.random.PRNGKey(42)
    u_high += 0.01 * jax.random.normal(key, u_high.shape)
    v_high += 0.01 * jax.random.normal(jax.random.split(key)[0], v_high.shape)
    
    # 先运行一段时间让流动充分发展
    print("运行高分辨率模拟以发展湍流...")
    # 运行1000步以充分发展湍流
    u_high, v_high, p_high, dt_high = simulate_timesteps(
        solver_high, u_high, v_high, p_high, 1000, use_basic_advection=True
    )
    
    # 收集快照
    snapshots = []
    
    # 使用较大的时间步进行模拟，并收集快照
    for i in range(num_snapshots):
        # 模拟几步然后收集一个快照
        u_high, v_high, p_high, _ = simulate_timesteps(
            solver_high, u_high, v_high, p_high, 
            10,  # 每收集一个快照模拟10步
            use_basic_advection=True
        )
        
        # 存储快照
        snapshots.append((u_high.copy(), v_high.copy()))
        
        if (i+1) % 100 == 0:
            print(f"已生成 {i+1}/{num_snapshots} 个快照")
    
    return snapshots

def prepare_training_data(high_res_snapshots, downsampling_factor):
    """
    准备训练数据，包括对高分辨率数据进行降采样
    
    参数:
        high_res_snapshots: 高分辨率快照列表，每个快照是(u, v)元组
        downsampling_factor: 降采样因子
        
    返回:
        low_res_data: 降采样后的数据 [(u_low, v_low), ...]
        high_res_downsampled: 高分辨率数据降采样后的参考解 [(u_exact, v_exact), ...]
    """
    low_res_data = []
    high_res_downsampled = []
    
    for u_high, v_high in high_res_snapshots:
        # 降采样高分辨率数据
        u_low, v_low = downsample_velocity_field(u_high, v_high, downsampling_factor)
        
        # 将高分辨率数据降采样为参考解
        u_exact, v_exact = downsample_velocity_field(u_high, v_high, downsampling_factor)
        
        low_res_data.append((u_low, v_low))
        high_res_downsampled.append((u_exact, v_exact))
    
    return low_res_data, high_res_downsampled

def create_training_trajectories(low_res_data, trajectory_length):
    """
    从数据中创建训练轨迹
    
    参数:
        low_res_data: 低分辨率数据列表
        trajectory_length: 每条轨迹的长度
        
    返回:
        trajectories: 轨迹列表，每条轨迹是连续的快照
    """
    trajectories = []
    for i in range(len(low_res_data) - trajectory_length + 1):
        trajectory = low_res_data[i:i+trajectory_length]
        trajectories.append(trajectory)
    return trajectories

def mean_squared_error(pred, target):
    """
    计算均方误差
    
    参数:
        pred: 预测值
        target: 目标值
        
    返回:
        mse: 均方误差
    """
    return jnp.mean((pred - target) ** 2)

def unroll_simulation(params, initial_u, initial_v, initial_p, li_model, solver, num_steps):
    """
    展开模拟多个时间步
    
    参数:
        params: LI模型参数
        initial_u, initial_v, initial_p: 初始场
        li_model: LearnedInterpolation模型
        solver: NavierStokesSolver实例
        num_steps: 展开的时间步数
        
    返回:
        u_final, v_final: 最终状态
    """
    # 克隆求解器和LI模型
    li = LearnedInterpolation(
        li_model, params, 
        stencil_size=li_model.stencil_size, 
        num_interp_points=li_model.num_interp_points,
        scale_factor=1.0
    )
    
    # 创建一个函数，使用LI模型的对流项计算
    def li_advection_fn(u, v, dx, dy):
        return li(u, v, dx, dy)
    
    # 修改求解器使用LI计算对流项
    solver_with_li = NavierStokesSolver(
        solver.grid, Re=solver.Re, cfl=solver.cfl, force_fn=solver.force_fn
    )
    solver_with_li.advection_fn = li_advection_fn
    
    # 模拟多个时间步
    u_curr, v_curr, p_curr = initial_u, initial_v, initial_p
    
    for _ in range(num_steps):
        u_curr, v_curr, p_curr, _ = solver_with_li.step(
            u_curr, v_curr, p_curr, use_basic_advection=False
        )
    
    return u_curr, v_curr

def compute_loss(params, trajectory, reference, li_model, solver, num_steps, alpha=1.0):
    """
    计算损失函数
    
    参数:
        params: LI模型参数
        trajectory: 输入轨迹 [(u0, v0), ...]
        reference: 参考解 [(u_exact0, v_exact0), ...]
        li_model: LearnedInterpolation模型
        solver: NavierStokesSolver实例
        num_steps: 展开的时间步数
        alpha: 正则化系数
        
    返回:
        loss: 损失值
    """
    # 获取初始状态
    initial_u, initial_v = trajectory[0]
    initial_p = jnp.zeros_like(initial_u)  # 初始压力场为零
    
    # 展开模拟
    u_pred, v_pred = unroll_simulation(
        params, initial_u, initial_v, initial_p, li_model, solver, num_steps
    )
    
    # 计算与参考解的MSE
    u_exact, v_exact = reference[num_steps]
    loss_u = mean_squared_error(u_pred, u_exact)
    loss_v = mean_squared_error(v_pred, v_exact)
    
    # 总损失
    loss = loss_u + loss_v
    
    # 添加L2正则化
    # [实现L2正则化，如果需要]
    
    return loss

@partial(jit, static_argnums=(2, 3, 4, 7))
def train_step(params, opt_state, li_model, solver, num_steps, trajectory, reference, optimizer):
    """
    执行一步训练
    
    参数:
        params: 模型参数
        opt_state: 优化器状态
        li_model: LearnedInterpolation模型
        solver: NavierStokesSolver实例
        num_steps: 展开的时间步数
        trajectory: 输入轨迹
        reference: 参考解
        optimizer: 优化器
        
    返回:
        params: 更新后的参数
        opt_state: 更新后的优化器状态
        loss: 损失值
    """
    # 计算损失和梯度
    loss_fn = lambda p: compute_loss(p, trajectory, reference, li_model, solver, num_steps)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # 更新参数
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

def train_li_model(li_model, solver, training_data, reference_data, 
                  num_epochs=100, batch_size=4, learning_rate=1e-3, 
                  unroll_steps=16, save_dir='./models'):
    """
    训练LI模型
    
    参数:
        li_model: LearnedInterpolation模型
        solver: NavierStokesSolver实例
        training_data: 训练数据 [(u0, v0), ...]
        reference_data: 参考解 [(u_exact0, v_exact0), ...]
        num_epochs: 训练轮数
        batch_size: 批大小
        learning_rate: 学习率
        unroll_steps: 展开的时间步数
        save_dir: 模型保存目录
        
    返回:
        params: 训练后的模型参数
    """
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建训练轨迹
    trajectories = create_training_trajectories(training_data, unroll_steps + 1)
    reference_trajectories = create_training_trajectories(reference_data, unroll_steps + 1)
    
    # 确保轨迹数量相同
    assert len(trajectories) == len(reference_trajectories)
    
    # 初始化模型参数
    key = jax.random.PRNGKey(42)
    input_shape = (1, solver.grid.ny, solver.grid.nx, 2*li_model.stencil_size**2)
    params = initialize_model(li_model, input_shape, key)
    
    # 创建优化器
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    
    # 训练循环
    num_batches = len(trajectories) // batch_size
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0.0
        
        # 打乱数据
        indices = np.arange(len(trajectories))
        np.random.shuffle(indices)
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(trajectories))
            batch_indices = indices[batch_start:batch_end]
            
            # 对批次中的每个轨迹进行训练
            batch_loss = 0.0
            for idx in batch_indices:
                trajectory = trajectories[idx]
                reference = reference_trajectories[idx]
                
                # 执行训练步骤
                params, opt_state, loss = train_step(
                    params, opt_state, li_model, solver, unroll_steps, trajectory, reference, optimizer
                )
                batch_loss += loss
            
            # 计算批次平均损失
            batch_loss /= len(batch_indices)
            epoch_loss += batch_loss
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {batch_loss:.6f}")
        
        # 计算轮次平均损失
        epoch_loss /= num_batches
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}, Time: {elapsed_time:.2f}s")
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_path = os.path.join(save_dir, f"li_model_best.npz")
            np.savez(model_path, **params)
            print(f"保存最佳模型，损失: {best_loss:.6f}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(save_dir, f"li_model_epoch_{epoch+1}.npz")
            np.savez(model_path, **params)
    
    return params

def main():
    """
    主函数：执行整个训练流程
    """
    # 设置参数
    Re = 1000
    nx_high, ny_high = 256, 256  # 较小的高分辨率用于测试，实际应为2048x2048
    nx_low, ny_low = 64, 64
    downsampling_factor = nx_high // nx_low
    
    # 生成训练数据
    print("生成高分辨率DNS数据...")
    high_res_snapshots = generate_dns_data(nx_high, ny_high, num_snapshots=1000, Re=Re)
    
    print("准备训练数据...")
    training_data, reference_data = prepare_training_data(high_res_snapshots, downsampling_factor)
    
    # 创建模型和求解器
    grid_low = StaggeredGrid(nx_low, ny_low)
    force_fn = lambda u, v: kolmogorov_forcing(u, v, k=4, drag_coef=0.1)
    solver = NavierStokesSolver(grid_low, Re=Re, force_fn=force_fn)
    
    # 创建LI模型
    stencil_size = 4
    num_interp_points = 8
    output_features = num_interp_points * (stencil_size**2 - 1)  # 减去系数和为1的约束
    li_model = create_model(hidden_features=64, num_layers=6, output_features=output_features)
    li_model.stencil_size = stencil_size
    li_model.num_interp_points = num_interp_points
    
    # 训练模型
    print("开始训练LI模型...")
    params = train_li_model(
        li_model, solver, training_data, reference_data,
        num_epochs=100, batch_size=4, learning_rate=1e-3,
        unroll_steps=32, save_dir='./models'
    )
    
    print("训练完成！")

if __name__ == "__main__":
    main() 