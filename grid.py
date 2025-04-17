import jax
import jax.numpy as jnp

class StaggeredGrid:
    """
    交错网格类，用于二维不可压缩Navier-Stokes方程的有限体积法求解。
    速度分量定义在单元格的面上，压力定义在单元格中心。
    """
    def __init__(self, nx, ny, domain_size=(2*jnp.pi, 2*jnp.pi)):
        """
        初始化交错网格
        
        参数:
            nx, ny: 网格单元数量（x和y方向）
            domain_size: 计算域的尺寸，默认为 [0, 2π] × [0, 2π]
        """
        self.nx = nx
        self.ny = ny
        self.Lx, self.Ly = domain_size
        
        # 计算网格间距
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny
        
        # 创建坐标
        # 压力点 (单元格中心)
        self.p_x = jnp.linspace(0.5*self.dx, self.Lx - 0.5*self.dx, nx)
        self.p_y = jnp.linspace(0.5*self.dy, self.Ly - 0.5*self.dy, ny)
        self.p_grid_x, self.p_grid_y = jnp.meshgrid(self.p_x, self.p_y)
        
        # u速度点 (x-方向速度，定义在垂直于x的面上)
        self.u_x = jnp.linspace(0, self.Lx - self.dx, nx)
        self.u_y = jnp.linspace(0.5*self.dy, self.Ly - 0.5*self.dy, ny)
        self.u_grid_x, self.u_grid_y = jnp.meshgrid(self.u_x, self.u_y)
        
        # v速度点 (y-方向速度，定义在垂直于y的面上)
        self.v_x = jnp.linspace(0.5*self.dx, self.Lx - 0.5*self.dx, nx)
        self.v_y = jnp.linspace(0, self.Ly - self.dy, ny)
        self.v_grid_x, self.v_grid_y = jnp.meshgrid(self.v_x, self.v_y)
    
    def create_fields(self, u_init=None, v_init=None, p_init=None):
        """
        创建初始场
        
        参数:
            u_init, v_init, p_init: 初始场函数，如果为None则初始化为零
            
        返回:
            u, v, p: 初始速度和压力场
        """
        if u_init is None:
            u = jnp.zeros((self.ny, self.nx))
        else:
            u = u_init(self.u_grid_x, self.u_grid_y)
            
        if v_init is None:
            v = jnp.zeros((self.ny, self.nx))
        else:
            v = v_init(self.v_grid_x, self.v_grid_y)
            
        if p_init is None:
            p = jnp.zeros((self.ny, self.nx))
        else:
            p = p_init(self.p_grid_x, self.p_grid_y)
            
        return u, v, p
    
    def create_kolmogorov_flow(self, A=1.0, k=4):
        """
        创建Kolmogorov流的初始条件
        
        参数:
            A: 振幅
            k: 波数
            
        返回:
            u, v, p: 初始速度和压力场
        """
        # 初始速度场: u = (A*sin(k*y), 0)
        u = A * jnp.sin(k * self.u_grid_y)
        v = jnp.zeros((self.ny, self.nx))
        p = jnp.zeros((self.ny, self.nx))
        
        return u, v, p
    
    def interpolate_to_cell_centers(self, u, v):
        """
        将面上的速度插值到单元格中心
        
        参数:
            u, v: 面上的速度分量
            
        返回:
            uc, vc: 中心点的速度分量
        """
        # 简单平均插值到中心
        uc = 0.5 * (u + jnp.roll(u, -1, axis=1))
        vc = 0.5 * (v + jnp.roll(v, -1, axis=0))
        
        return uc, vc

def downsample_velocity_field(u_high, v_high, factor):
    """
    对高分辨率速度场进行降采样，保持无散性
    
    参数:
        u_high, v_high: 高分辨率速度场
        factor: 降采样因子
        
    返回:
        u_low, v_low: 降采样后的速度场
    """
    ny, nx = u_high.shape
    
    # 确保降采样因子能整除网格尺寸
    assert nx % factor == 0 and ny % factor == 0, "网格尺寸必须能被降采样因子整除"
    
    # 新的网格尺寸
    nx_low = nx // factor
    ny_low = ny // factor
    
    # 为交错网格上的速度执行特殊的降采样
    # 对u（x方向速度）：在y方向先平均，然后在x方向上采样
    u_temp = jnp.mean(u_high.reshape(ny, nx_low, factor), axis=2)
    u_low = jnp.mean(u_temp.reshape(ny_low, factor, nx_low), axis=1)
    
    # 对v（y方向速度）：在x方向先平均，然后在y方向上采样
    v_temp = jnp.mean(v_high.reshape(ny, nx_low, factor), axis=2)
    v_low = jnp.mean(v_temp.reshape(ny_low, factor, nx_low), axis=1)
    
    return u_low, v_low 