import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib import font_manager

# 设置中文字体
try:
    # 尝试设置微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
except:
    try:
        # 尝试设置黑体
        plt.rcParams['font.sans-serif'] = ['SimHei']
    except:
        print("警告：未能找到合适的中文字体，图表中的中文可能无法正确显示")

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_loss_log(file_path):
    epochs = []
    g_gan_losses = []
    g_l1_losses = []
    g_ssim_losses = []
    d_real_losses = []
    d_fake_losses = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if 'epoch:' in line and 'G_GAN:' in line:
                try:
                    # 使用正则表达式提取数值
                    epoch_match = re.search(r'epoch: (\d+)', line)
                    g_gan_match = re.search(r'G_GAN: ([\d.-]+)', line)
                    g_l1_match = re.search(r'G_L1: ([\d.-]+)', line)
                    g_ssim_match = re.search(r'G_SSIM: ([\d.-]+)', line)
                    d_real_match = re.search(r'D_real: ([\d.-]+)', line)
                    d_fake_match = re.search(r'D_fake: ([\d.-]+)', line)
                    
                    if all([epoch_match, g_gan_match, g_l1_match, g_ssim_match, d_real_match, d_fake_match]):
                        epoch = int(epoch_match.group(1))
                        g_gan = float(g_gan_match.group(1))
                        g_l1 = float(g_l1_match.group(1))
                        g_ssim = float(g_ssim_match.group(1))
                        d_real = float(d_real_match.group(1))
                        d_fake = float(d_fake_match.group(1))
                        
                        epochs.append(epoch)
                        g_gan_losses.append(g_gan)
                        g_l1_losses.append(g_l1)
                        g_ssim_losses.append(g_ssim)
                        d_real_losses.append(d_real)
                        d_fake_losses.append(d_fake)
                except Exception as e:
                    print(f"跳过无效行: {line.strip()}")
                    continue
    
    return epochs, g_gan_losses, g_l1_losses, g_ssim_losses, d_real_losses, d_fake_losses

def plot_losses(file_path):
    epochs, g_gan, g_l1, g_ssim, d_real, d_fake = parse_loss_log(file_path)
    
    # 创建迭代次数数组
    iters = np.arange(len(epochs))
    
    plt.figure(figsize=(15, 10))
    
    # 绘制生成器损失
    plt.subplot(2, 2, 1)
    plt.plot(iters, g_gan, label='G_GAN Loss', alpha=0.7)
    plt.plot(iters, g_l1, label='G_L1 Loss', alpha=0.7)
    plt.plot(iters, g_ssim, label='G_SSIM Loss', alpha=0.7)
    plt.title('生成器损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 绘制判别器损失
    plt.subplot(2, 2, 2)
    plt.plot(iters, d_real, label='D_Real Loss', alpha=0.7)
    plt.plot(iters, d_fake, label='D_Fake Loss', alpha=0.7)
    plt.title('判别器损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 绘制G_GAN和G_L1的对比
    plt.subplot(2, 2, 3)
    plt.plot(iters, g_gan, label='G_GAN Loss', alpha=0.7)
    plt.plot(iters, g_l1, label='G_L1 Loss', alpha=0.7)
    plt.title('G_GAN vs G_L1 损失对比')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 绘制SSIM指标
    plt.subplot(2, 2, 4)
    plt.plot(iters, g_ssim, label='G_SSIM', color='green', alpha=0.7)
    plt.title('SSIM度量')
    plt.xlabel('迭代次数')
    plt.ylabel('SSIM值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'training_losses.png'")
    plt.show()

if __name__ == '__main__':
    file_path = 'checkpoints/dem_experiment_s1/loss_log.txt'
    plot_losses(file_path) 