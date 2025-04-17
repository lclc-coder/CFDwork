import jax
import jax.numpy as jnp
from functools import partial
from jax import jit

def calculate_cfl_dt(u, v, dx, dy, cfl_target=0.5):
    """
    根据CFL条件计算时间步长
    
    参数:
        u, v: 速度场
        dx, dy: 网格间距
        cfl_target: 目标CFL数
        
    返回:
        dt: 时间步长
    """
    max_vel = jnp.maximum(jnp.max(jnp.abs(u)), jnp.max(jnp.abs(v)))
    dt = cfl_target * jnp.minimum(dx, dy) / (max_vel + 1e-6)
    return dt

@jit
def laplacian(f, dx, dy):
    """
    计算二维拉普拉斯算子 ∇²f，使用二阶中心差分
    
    参数:
        f: 标量场
        dx, dy: 网格间距
        
    返回:
        lap_f: 拉普拉斯场 ∇²f
    """
    # 二阶中心差分近似拉普拉斯算子
    f_x_plus = jnp.roll(f, -1, axis=1)  # i+1, j
    f_x_minus = jnp.roll(f, 1, axis=1)  # i-1, j
    f_y_plus = jnp.roll(f, -1, axis=0)  # i, j+1
    f_y_minus = jnp.roll(f, 1, axis=0)  # i, j-1
    
    lap_f = (f_x_plus - 2*f + f_x_minus) / (dx**2) + \
            (f_y_plus - 2*f + f_y_minus) / (dy**2)
            
    return lap_f

@jit
def advection_upwind_basic(u, v, u_face, v_face, dx, dy):
    """
    计算对流项 ∇·(u⊗u)，使用一阶迎风格式
    这是基础版本，将被LI模块替换
    
    参数:
        u, v: 速度分量
        u_face, v_face: 面上的速度，已经通过插值计算
        dx, dy: 网格间距
        
    返回:
        adv_u, adv_v: 对流项
    """
    # x方向速度u的对流项
    # ∂(uu)/∂x
    u_east = jnp.roll(u, -1, axis=1)  # i+1, j
    u_west = jnp.roll(u, 1, axis=1)   # i-1, j
    flux_u_x = jnp.where(u_face > 0, u * u_face, u_east * u_face)
    adv_u_x = (flux_u_x - jnp.roll(flux_u_x, 1, axis=1)) / dx
    
    # ∂(vu)/∂y
    v_at_u = 0.25 * (v + jnp.roll(v, -1, axis=1) + 
                     jnp.roll(v, 1, axis=0) + jnp.roll(jnp.roll(v, -1, axis=1), 1, axis=0))
    u_north = jnp.roll(u, -1, axis=0)  # i, j+1
    u_south = jnp.roll(u, 1, axis=0)   # i, j-1
    flux_u_y = jnp.where(v_at_u > 0, u * v_at_u, u_north * v_at_u)
    adv_u_y = (flux_u_y - jnp.roll(flux_u_y, 1, axis=0)) / dy
    
    # y方向速度v的对流项
    # ∂(uv)/∂x
    u_at_v = 0.25 * (u + jnp.roll(u, -1, axis=0) + 
                     jnp.roll(u, 1, axis=1) + jnp.roll(jnp.roll(u, -1, axis=0), 1, axis=1))
    v_east = jnp.roll(v, -1, axis=1)  # i+1, j
    v_west = jnp.roll(v, 1, axis=1)   # i-1, j
    flux_v_x = jnp.where(u_at_v > 0, v * u_at_v, v_east * u_at_v)
    adv_v_x = (flux_v_x - jnp.roll(flux_v_x, 1, axis=1)) / dx
    
    # ∂(vv)/∂y
    v_north = jnp.roll(v, -1, axis=0)  # i, j+1
    v_south = jnp.roll(v, 1, axis=0)   # i, j-1
    flux_v_y = jnp.where(v_face > 0, v * v_face, v_north * v_face)
    adv_v_y = (flux_v_y - jnp.roll(flux_v_y, 1, axis=0)) / dy
    
    # 对流项
    adv_u = adv_u_x + adv_u_y
    adv_v = adv_v_x + adv_v_y
    
    return adv_u, adv_v

@jit
def interpolate_velocity_to_faces(u, v):
    """
    将速度插值到单元格面上
    
    参数:
        u, v: 速度分量
        
    返回:
        u_face, v_face: 面上的速度
    """
    # 简单平均插值
    # u在x方向上的面
    u_face_x = 0.5 * (u + jnp.roll(u, 1, axis=1))
    # v在y方向上的面
    v_face_y = 0.5 * (v + jnp.roll(v, 1, axis=0))
    
    return u_face_x, v_face_y

@jit
def divergence(u, v, dx, dy):
    """
    计算速度场的散度 ∇·u
    
    参数:
        u, v: 速度分量
        dx, dy: 网格间距
        
    返回:
        div: 散度场
    """
    # 散度 = ∂u/∂x + ∂v/∂y
    dudx = (u - jnp.roll(u, 1, axis=1)) / dx
    dvdy = (v - jnp.roll(v, 1, axis=0)) / dy
    
    div = dudx + dvdy
    return div

@jit
def gradient(p, dx, dy):
    """
    计算压力梯度 ∇p
    
    参数:
        p: 压力场
        dx, dy: 网格间距
        
    返回:
        grad_p_x, grad_p_y: 压力梯度
    """
    # x方向梯度
    grad_p_x = (jnp.roll(p, -1, axis=1) - p) / dx
    # y方向梯度
    grad_p_y = (jnp.roll(p, -1, axis=0) - p) / dy
    
    return grad_p_x, grad_p_y

@partial(jit, static_argnums=(5,))
def solve_pressure_poisson(divergence_field, dx, dy, bc_type='periodic', tol=1e-6, max_iter=1000):
    """
    求解压力泊松方程 ∇²p = div，使用快速傅里叶变换
    
    参数:
        divergence_field: 速度场的散度
        dx, dy: 网格间距
        bc_type: 边界条件类型
        tol: 容差
        max_iter: 最大迭代次数
        
    返回:
        p: 压力场
    """
    ny, nx = divergence_field.shape
    
    # 使用FFT求解泊松方程
    # 创建网格波数
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = jnp.meshgrid(kx, ky)
    
    # 拉普拉斯算子在频域中是 -(kx^2 + ky^2)
    k_squared = kx_grid**2 + ky_grid**2
    
    # 避免除以零（对应于零频率分量）
    k_squared_inv = jnp.where(k_squared > 1e-10, 1.0 / k_squared, 0.0)
    
    # 转换源项到频域
    divergence_hat = jnp.fft.fft2(divergence_field)
    
    # 求解泊松方程
    p_hat = -divergence_hat * k_squared_inv
    
    # 转回空间域并取实部（由于数值误差，可能有小的虚部）
    p = jnp.real(jnp.fft.ifft2(p_hat))
    
    # 确保压力场均值为零
    p = p - jnp.mean(p)
    
    return p

@partial(jit, static_argnums=(7, 8, 9))
def navier_stokes_step(u, v, p, dx, dy, dt, Re, force_fn, is_viscous=True, use_basic_advection=True):
    """
    执行一个时间步的Navier-Stokes方程求解
    
    参数:
        u, v: 速度分量
        p: 压力场
        dx, dy: 网格间距
        dt: 时间步长
        Re: 雷诺数
        force_fn: 外力函数
        is_viscous: 是否包含粘性项
        use_basic_advection: 是否使用基础对流计算（而非LI）
        
    返回:
        u_new, v_new, p_new: 更新后的速度和压力场
    """
    # 计算对流通量（可以被LI替换）
    if use_basic_advection:
        # 基础方法：插值速度到面，然后计算对流
        u_face, v_face = interpolate_velocity_to_faces(u, v)
        adv_u, adv_v = advection_upwind_basic(u, v, u_face, v_face, dx, dy)
    else:
        # 这部分将在learned_interpolation.py中实现
        adv_u, adv_v = jnp.zeros_like(u), jnp.zeros_like(v)
    
    # 计算扩散项（粘性项）
    if is_viscous:
        visc_u = laplacian(u, dx, dy) / Re
        visc_v = laplacian(v, dx, dy) / Re
    else:
        visc_u, visc_v = jnp.zeros_like(u), jnp.zeros_like(v)
    
    # 计算外力
    f_u, f_v = force_fn(u, v)
    
    # 更新速度（不包括压力项）
    u_star = u + dt * (-adv_u + visc_u + f_u)
    v_star = v + dt * (-adv_v + visc_v + f_v)
    
    # 计算临时速度场的散度
    div_star = divergence(u_star, v_star, dx, dy)
    
    # 求解压力泊松方程
    p_new = solve_pressure_poisson(div_star / dt, dx, dy)
    
    # 计算压力梯度
    grad_p_x, grad_p_y = gradient(p_new, dx, dy)
    
    # 更新速度场
    u_new = u_star - dt * grad_p_x
    v_new = v_star - dt * grad_p_y
    
    return u_new, v_new, p_new

def kolmogorov_forcing(u, v, k=4, drag_coef=0.1):
    """
    Kolmogorov流的外力项: f=(sin(ky),0) - drag_coef*u
    
    参数:
        u, v: 速度分量
        k: 波数
        drag_coef: 阻力系数
        
    返回:
        f_u, f_v: 力的分量
    """
    ny, nx = u.shape
    y = jnp.linspace(0, 2*jnp.pi, ny+1)[:-1]
    y_grid = jnp.tile(y[:, jnp.newaxis], (1, nx))
    
    # sin(ky) 力
    f_sin = jnp.sin(k * y_grid)
    
    # 阻力项
    f_drag_u = -drag_coef * u
    f_drag_v = -drag_coef * v
    
    # 合力
    f_u = f_sin + f_drag_u
    f_v = f_drag_v
    
    return f_u, f_v

class NavierStokesSolver:
    """
    二维不可压缩Navier-Stokes方程求解器
    """
    def __init__(self, grid, Re=1000, cfl=0.5, force_fn=None):
        """
        初始化求解器
        
        参数:
            grid: 交错网格对象
            Re: 雷诺数
            cfl: CFL数
            force_fn: 外力函数
        """
        self.grid = grid
        self.Re = Re
        self.cfl = cfl
        
        if force_fn is None:
            self.force_fn = lambda u, v: (jnp.zeros_like(u), jnp.zeros_like(v))
        else:
            self.force_fn = force_fn
    
    def step(self, u, v, p, dt=None, use_basic_advection=True):
        """
        执行一个时间步的求解
        
        参数:
            u, v: 速度分量
            p: 压力场
            dt: 时间步长，如果为None则自动计算
            use_basic_advection: 是否使用基础对流计算
            
        返回:
            u_new, v_new, p_new: 更新后的速度和压力场
            dt: 使用的时间步长
        """
        if dt is None:
            dt = calculate_cfl_dt(u, v, self.grid.dx, self.grid.dy, self.cfl)
        
        u_new, v_new, p_new = navier_stokes_step(
            u, v, p, 
            self.grid.dx, self.grid.dy, 
            dt, self.Re, self.force_fn,
            is_viscous=True,
            use_basic_advection=use_basic_advection
        )
        
        return u_new, v_new, p_new, dt 