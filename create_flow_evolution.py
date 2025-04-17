import matplotlib.pyplot as plt
import numpy as np
import os

def create_flow_evolution_plot(input_dir='./basic_results', output_dir='./flow_evolution'):
    """
    创建流场演化的拼图
    
    参数:
        input_dir: 输入目录，包含之前模拟生成的图像
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择特定时刻
    time_steps = [0, 50, 100, 150, 190]
    
    # 创建速度场演化图
    plt.figure(figsize=(20, 4))
    
    for i, step in enumerate(time_steps):
        plt.subplot(1, 5, i+1)
        img = plt.imread(os.path.join(input_dir, f'velocity_{step:04d}.png'))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Step {step}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_evolution.png'), dpi=200)
    plt.close()
    
    # 创建涡量场演化图
    plt.figure(figsize=(20, 4))
    
    for i, step in enumerate(time_steps):
        plt.subplot(1, 5, i+1)
        img = plt.imread(os.path.join(input_dir, f'vorticity_{step:04d}.png'))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Step {step}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vorticity_evolution.png'), dpi=200)
    plt.close()
    
    print(f"流场演化图像已保存至 {output_dir}")

def create_animation_from_images(input_dir='./basic_results', output_file='flow_animation.mp4', pattern='vorticity_*.png', fps=10):
    """
    从一系列图像创建动画
    
    参数:
        input_dir: 输入目录，包含图像
        output_file: 输出动画文件
        pattern: 文件名模式
        fps: 帧率
    """
    import glob
    from moviepy.editor import ImageSequenceClip
    
    # 获取所有匹配的图像文件
    image_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not image_files:
        print(f"未找到匹配 {pattern} 的图像文件")
        return
    
    # 创建动画
    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(output_file, fps=fps)
    
    print(f"动画已保存至 {output_file}")

def main():
    # 创建流场演化图
    create_flow_evolution_plot()
    
    # 尝试创建动画（如果有moviepy库）
    try:
        import moviepy.editor
        create_animation_from_images(output_file='vorticity_animation.mp4')
        create_animation_from_images(output_file='velocity_animation.mp4', pattern='velocity_*.png')
    except ImportError:
        print("未安装moviepy库，跳过动画创建")

if __name__ == "__main__":
    main() 