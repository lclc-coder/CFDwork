import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import jit, vmap, grad
from functools import partial
import numpy as np
from scipy import linalg

class ConvBlock(nn.Module):
    """
    卷积块，包含一个卷积层和ReLU激活函数
    """
    features: int
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.features, 
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME'
        )(x)
        return nn.relu(x)

class LearnedInterpolationNetwork(nn.Module):
    """
    学习型插值网络，用于替换基础求解器中的对流项计算
    """
    hidden_features: int = 64
    num_layers: int = 6
    output_features: int = 120  # 假设一个单元格需要8个插值点，每个点需要15个系数
    kernel_size: int = 3
    
    @nn.compact
    def __call__(self, x):
        # 输入 x 预期形状: [batch, height, width, channels]
        
        # 初始卷积层
        x = nn.Conv(
            features=self.hidden_features, 
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME'
        )(x)
        x = nn.relu(x)
        
        # 中间卷积层
        for _ in range(self.num_layers - 2):
            x = ConvBlock(features=self.hidden_features, kernel_size=self.kernel_size)(x)
        
        # 输出卷积层 - 无激活函数，直接输出原始系数
        x = nn.Conv(
            features=self.output_features, 
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME'
        )(x)
        
        return x

def create_coefficient_constraints(stencil_size=4, num_points=8):
    """
    创建系数约束矩阵，用于将神经网络输出转换为满足约束的插值系数
    
    参数:
        stencil_size: 每个插值点使用的邻域大小
        num_points: 每个单元格需要计算的插值点数量
        
    返回:
        A: 约束矩阵的零空间基
        b: 一组有效的基础系数
    """
    # 每个插值点使用 stencil_size x stencil_size 的邻域
    # 每个点有 stencil_size*stencil_size 个系数
    # 约束: 所有系数之和为1（保证至少一阶精度）
    
    # 对于每个插值点
    A_list = []
    b_list = []
    
    for i in range(num_points):
        # 系数个数
        n_coeff = stencil_size * stencil_size
        
        # 约束矩阵: 所有系数之和为1
        constraint = np.ones((1, n_coeff))
        
        # 计算零空间基
        # 如果约束是 Ax = b，则零空间是 N(A)
        # 可以使用QR分解来计算
        Q = np.linalg.qr(constraint.T, mode='complete')[0]
        null_space = Q[:, 1:]  # 第一列对应于约束，剩余列构成零空间
        
        # 一个基础解（例如，最近邻点的线性插值）
        base_solution = np.zeros(n_coeff)
        center_idx = n_coeff // 2  # 假设中心点是插值区域的中心
        base_solution[center_idx] = 1.0  # 最简单的情况：权重全分配给中心点
        
        A_list.append(null_space)
        b_list.append(base_solution)
    
    # 合并所有点的约束
    A = jnp.array(linalg.block_diag(*A_list))
    b = jnp.array(np.concatenate(b_list))
    
    return A, b

def apply_coefficient_constraints(raw_coeffs, A, b):
    """
    应用系数约束，将神经网络的原始输出转换为满足约束的插值系数
    
    参数:
        raw_coeffs: 神经网络输出的原始系数
        A: 约束矩阵的零空间基
        b: 基础解
        
    返回:
        constrained_coeffs: 满足约束的插值系数
    """
    # raw_coeffs 形状: [batch, height, width, n_output]
    # 将输出解释为零空间的坐标
    # constrained_coeffs = b + A*raw_coeffs
    
    batch, height, width, _ = raw_coeffs.shape
    
    # 重塑为[batch*height*width, n_output]以便应用约束
    raw_coeffs_flat = raw_coeffs.reshape(-1, raw_coeffs.shape[-1])
    
    # 应用约束: c = b + A*raw
    constrained_coeffs_flat = b + jnp.matmul(raw_coeffs_flat, A.T)
    
    # 重塑回原始形状
    constrained_coeffs = constrained_coeffs_flat.reshape(batch, height, width, -1)
    
    return constrained_coeffs

def extract_local_neighborhood(field, window_size=4):
    """
    提取每个点周围的局部邻域
    
    参数:
        field: 输入场
        window_size: 邻域大小
        
    返回:
        neighborhoods: 每个点的局部邻域
    """
    # field 形状: [height, width]
    # 结果形状: [height, width, window_size, window_size]
    
    height, width = field.shape
    half_window = window_size // 2
    
    # 创建空数组存储局部邻域
    neighborhoods = jnp.zeros((height, width, window_size, window_size))
    
    # 对于每个偏移
    for i in range(window_size):
        for j in range(window_size):
            # 计算偏移（考虑周期性边界条件）
            offset_i = i - half_window
            offset_j = j - half_window
            
            # 使用roll实现周期性边界条件
            shifted_field = jnp.roll(jnp.roll(field, offset_i, axis=0), offset_j, axis=1)
            
            # 存储到邻域数组
            neighborhoods = neighborhoods.at[:, :, i, j].set(shifted_field)
    
    return neighborhoods

def scale_field(field, scale_factor=1.0):
    """
    缩放场，使其值落在适当的范围内
    
    参数:
        field: 输入场
        scale_factor: 缩放因子
        
    返回:
        scaled_field: 缩放后的场
    """
    # 计算场的统计信息
    field_mean = jnp.mean(field)
    field_std = jnp.std(field)
    
    # 标准化并缩放到 [-scale_factor, scale_factor] 范围
    scaled_field = scale_factor * (field - field_mean) / (field_std + 1e-8)
    
    return scaled_field

def unscale_field(scaled_field, original_mean, original_std, scale_factor=1.0):
    """
    反缩放场，恢复原始范围
    
    参数:
        scaled_field: 缩放后的场
        original_mean: 原始场的均值
        original_std: 原始场的标准差
        scale_factor: 缩放因子
        
    返回:
        unscaled_field: 反缩放后的场
    """
    # 反标准化
    unscaled_field = (scaled_field * (original_std + 1e-8) / scale_factor) + original_mean
    
    return unscaled_field

class LearnedInterpolation:
    """
    学习型插值模块，用于替换基础求解器中的对流项计算
    """
    def __init__(self, model, params, stencil_size=4, num_interp_points=8, scale_factor=1.0):
        """
        初始化学习型插值模块
        
        参数:
            model: 神经网络模型
            params: 模型参数
            stencil_size: 插值邻域大小
            num_interp_points: 每个单元格需要计算的插值点数量
            scale_factor: 输入缩放因子
        """
        self.model = model
        self.params = params
        self.stencil_size = stencil_size
        self.num_interp_points = num_interp_points
        self.scale_factor = scale_factor
        
        # 创建系数约束
        self.A, self.b = create_coefficient_constraints(stencil_size, num_interp_points)
    
    @partial(jit, static_argnums=(0,))
    def __call__(self, u, v, dx, dy):
        """
        计算对流项
        
        参数:
            u, v: 速度分量
            dx, dy: 网格间距
            
        返回:
            adv_u, adv_v: 对流项
        """
        # 1. 提取局部邻域
        u_neighborhoods = extract_local_neighborhood(u, self.stencil_size)
        v_neighborhoods = extract_local_neighborhood(v, self.stencil_size)
        
        # 重塑为[1, height, width, channels*window_size*window_size]用于输入神经网络
        # 将u和v堆叠为通道
        ny, nx = u.shape
        input_data = jnp.stack([
            u_neighborhoods.reshape(ny, nx, -1),
            v_neighborhoods.reshape(ny, nx, -1)
        ], axis=-1)
        
        # 2. 缩放输入
        input_data = scale_field(input_data, self.scale_factor)
        
        # 3. 通过模型获取原始系数
        raw_coeffs = self.model.apply(self.params, input_data)
        
        # 4. 应用系数约束
        coeffs = apply_coefficient_constraints(raw_coeffs, self.A, self.b)
        
        # 5. 使用系数计算插值结果
        # 这里需要根据具体的网格和计算需求来实现
        # 例如，计算单元格面上的插值速度，然后计算对流通量
        
        # 简化版实现 - 假设前一半系数用于u的插值，后一半用于v的插值
        half_idx = coeffs.shape[-1] // 2
        u_coeffs = coeffs[..., :half_idx]
        v_coeffs = coeffs[..., half_idx:]
        
        # 计算对流通量
        # 这里仅为示例，实际实现需要根据具体的网格结构调整
        adv_u, adv_v = self._compute_advection(u, v, u_coeffs, v_coeffs, dx, dy)
        
        return adv_u, adv_v
    
    def _compute_advection(self, u, v, u_coeffs, v_coeffs, dx, dy):
        """
        使用学习的插值系数计算对流项
        
        参数:
            u, v: 速度分量
            u_coeffs, v_coeffs: 插值系数
            dx, dy: 网格间距
            
        返回:
            adv_u, adv_v: 对流项
        """
        # 这个函数需要根据具体的网格结构和对流计算方式来实现
        # 下面是一个简化的实现，假设系数直接应用于对流计算
        
        # 提取局部邻域用于计算对流
        u_neighborhoods = extract_local_neighborhood(u, self.stencil_size)
        v_neighborhoods = extract_local_neighborhood(v, self.stencil_size)
        
        # 应用系数进行加权平均
        ny, nx = u.shape
        u_neighborhoods_flat = u_neighborhoods.reshape(ny, nx, -1)
        v_neighborhoods_flat = v_neighborhoods.reshape(ny, nx, -1)
        
        # 计算插值结果（加权平均）
        u_interp = jnp.sum(u_neighborhoods_flat * u_coeffs, axis=-1)
        v_interp = jnp.sum(v_neighborhoods_flat * v_coeffs, axis=-1)
        
        # 计算对流导数
        dudx = (u_interp - jnp.roll(u_interp, 1, axis=1)) / dx
        dudy = (u_interp - jnp.roll(u_interp, 1, axis=0)) / dy
        dvdx = (v_interp - jnp.roll(v_interp, 1, axis=1)) / dx
        dvdy = (v_interp - jnp.roll(v_interp, 1, axis=0)) / dy
        
        # 计算对流项
        adv_u = u * dudx + v * dudy
        adv_v = u * dvdx + v * dvdy
        
        return adv_u, adv_v

def create_model(hidden_features=64, num_layers=6, output_features=120):
    """
    创建学习型插值网络模型
    
    参数:
        hidden_features: 隐藏层特征数
        num_layers: 网络层数
        output_features: 输出特征数
        
    返回:
        model: 神经网络模型
    """
    return LearnedInterpolationNetwork(
        hidden_features=hidden_features,
        num_layers=num_layers,
        output_features=output_features
    )

def initialize_model(model, input_shape, key=jax.random.PRNGKey(0)):
    """
    初始化模型参数
    
    参数:
        model: 神经网络模型
        input_shape: 输入形状
        key: 随机数生成器密钥
        
    返回:
        params: 初始化的模型参数
    """
    # 创建虚拟输入进行初始化
    dummy_input = jnp.ones(input_shape)
    params = model.init(key, dummy_input)
    return params 