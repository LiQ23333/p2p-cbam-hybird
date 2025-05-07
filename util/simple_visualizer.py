import numpy as np
import os
import time
from . import util

class SimpleVisualizer():
    """简化版可视化器，不使用visdom，只保存图像和损失到文件"""

    def __init__(self, opt):
        self.opt = opt
        self.name = opt.name
        self.current_epoch = 0
        
        # 创建保存图像的目录
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        print('create image directory %s...' % self.img_dir)
        util.mkdirs([self.img_dir])
        
        # 创建损失日志文件
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """重置保存状态"""
        pass

    def display_current_results(self, visuals, epoch, save_result):
        """保存当前结果到图像文件

        Parameters:
            visuals (OrderedDict) - - 要保存的图像字典
            epoch (int) - - 当前epoch
            save_result (bool) - - 是否保存结果
        """
        if save_result:
            # 保存图像到磁盘
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """将当前损失写入日志文件

        Parameters:
            epoch (int)           -- 当前epoch
            counter_ratio (float) -- 当前epoch的进度（0到1之间）
            losses (OrderedDict)  -- 训练损失字典
        """
        # 将损失写入日志文件
        with open(self.log_name, "a") as log_file:
            message = '(epoch: %d, iters: %d) ' % (epoch, int(counter_ratio * 100))
            for k, v in losses.items():
                message += '%s: %.3f ' % (k, v)
            log_file.write(message + '\n')

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """打印当前损失到控制台

        Parameters:
            epoch (int) -- 当前epoch
            iters (int) -- 当前训练迭代次数
            losses (OrderedDict) -- 训练损失字典
            t_comp (float) -- 每个数据点的计算时间
            t_data (float) -- 每个数据点的数据加载时间
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)  # 打印消息到控制台 