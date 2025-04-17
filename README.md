# 基于学习型插值的二维不可压缩Navier-Stokes方程求解器

这个代码库实现了一个基于JAX的二维不可压缩Navier-Stokes方程求解器，以及使用机器学习（学习型插值，Learned Interpolation）来加速求解过程。该实现主要基于论文 "Machine Learning Accelerated Computational Fluid Dynamics" (Kochkov et al., 2021)。

## 功能特点

- 基础CFD求解器：使用有限体积法在交错网格上求解二维不可压缩Navier-Stokes方程
- 学习型插值 (LI) 模块：使用全卷积神经网络替换基础求解器中的对流项计算
- 端到端训练：利用JAX的自动微分功能进行端到端训练
- 可视化工具：包含用于可视化速度场和涡量场的函数，以及创建动画的功能

## 安装依赖

本代码需要以下Python依赖：

```bash
pip install jax jaxlib flax optax numpy matplotlib
```

如果要创建动画，还需要安装FFmpeg：

```bash
# 对于Ubuntu/Debian
apt-get install ffmpeg

# 对于macOS
brew install ffmpeg
```

## 文件结构

- `grid.py`: 交错网格实现和数据处理函数
- `solver.py`: 基础CFD求解器实现
- `learned_interpolation.py`: 学习型插值模块
- `training.py`: 模型训练流程
- `simulation.py`: 主程序，用于运行模拟和可视化结果

## 使用方法

### 训练模型

```bash
python simulation.py --mode train
```

这将运行高分辨率DNS模拟，生成训练数据，并训练LI模型。训练好的模型将保存在`./models`目录下。

### 运行模拟

```bash
python simulation.py --mode simulate --nx 64 --ny 64 --Re 1000 --steps 1000 --output ./results
```

这将使用基础求解器或LI增强求解器（如果存在训练好的模型）运行模拟，并将结果保存在`./results`目录下。

### 性能比较

```bash
python simulation.py --mode compare --nx 64 --ny 64 --steps 1000 --output ./comparison
```

这将同时运行基础求解器和LI增强求解器，比较它们的性能和结果，并将对比结果保存在`./comparison`目录下。

## 参数说明

- `--mode`: 运行模式，可选 `train`、`simulate` 或 `compare`
- `--nx`, `--ny`: 网格尺寸
- `--Re`: 雷诺数
- `--steps`: 模拟步数
- `--model`: 模型文件路径
- `--output`: 输出目录

## 示例：Kolmogorov流

默认的模拟设置是二维Kolmogorov流（Re=1000），具有以下特性：

- 计算域: [0, 2π] × [0, 2π]，使用周期性边界条件
- 外力项: f=(sin(4y), 0) - 0.1*u
- 初始条件: u=(sin(4y), 0) + 小扰动

## 扩展

可以通过以下方式扩展代码：

1. 实现更多的流动类型和初始条件
2. 尝试不同的神经网络架构
3. 实现其他数值方法或边界条件
4. 添加3D支持

## 引用

如果您在研究中使用了这段代码，请引用相关论文：

```
@article{kochkov2021machine,
  title={Machine learning accelerated computational fluid dynamics},
  author={Kochkov, Dmitrii and Smith, Jamie A and Alieva, Ayya and Wang, Qing and Brenner, Michael P and Hoyer, Stephan},
  journal={Proceedings of the National Academy of Sciences},
  volume={118},
  number={21},
  year={2021},
  publisher={National Acad Sciences}
}
``` 