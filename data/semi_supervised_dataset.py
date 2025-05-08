import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .base_dataset import BaseDataset
from .image_folder import make_dataset
from util.util import calculate_ssim

class SemiSupervisedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--labeled_ratio', type=float, default=0.75, help='ratio of labeled data')
        parser.add_argument('--pseudo_threshold', type=float, default=0.6, help='SSIM threshold for pseudo labels')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join('datasets', opt.dataroot, 'train', 'trainA')
        self.dir_B = os.path.join('datasets', opt.dataroot, 'train', 'trainB')
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        
        # 划分有标签和无标签数据
        random.shuffle(self.A_paths)
        split_idx = int(len(self.A_paths) * opt.labeled_ratio)
        self.labeled_paths = self.A_paths[:split_idx]
        self.unlabeled_paths = self.A_paths[split_idx:]
        
        self.pseudo_labels = {}  # 存储伪标签
        self.pseudo_ssim = {}    # 存储SSIM值
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        try:
            if index < len(self.labeled_paths):
                # 有标签数据
                A_path = self.labeled_paths[index]
                B_path = self.B_paths[index]
                A_img = Image.open(A_path).convert('RGB')
                B_img = Image.open(B_path).convert('RGB')
                A = self.transform(A_img)
                B = self.transform(B_img)
                return {
                    'A': A, 
                    'B': B, 
                    'A_paths': A_path, 
                    'B_paths': B_path, 
                    'is_labeled': torch.tensor(1.0),  # 使用张量而不是布尔值
                    'pseudo_label': torch.zeros_like(A),
                    'ssim': torch.tensor(0.0)
                }
            else:
                # 无标签数据
                idx = index - len(self.labeled_paths)
                A_path = self.unlabeled_paths[idx]
                A_img = Image.open(A_path).convert('RGB')
                A = self.transform(A_img)
                
                # 如果有伪标签，则使用伪标签
                if A_path in self.pseudo_labels:
                    pseudo_label = self.pseudo_labels[A_path]
                    ssim = self.pseudo_ssim[A_path]
                    return {
                        'A': A, 
                        'B': torch.zeros_like(A),  # 空标签
                        'A_paths': A_path, 
                        'B_paths': '',  # 空路径
                        'is_labeled': torch.tensor(0.0),  # 使用张量而不是布尔值
                        'pseudo_label': pseudo_label,
                        'ssim': torch.tensor(ssim)
                    }
                else:
                    return {
                        'A': A, 
                        'B': torch.zeros_like(A),  # 空标签
                        'A_paths': A_path, 
                        'B_paths': '',  # 空路径
                        'is_labeled': torch.tensor(0.0),  # 使用张量而不是布尔值
                        'pseudo_label': torch.zeros_like(A),
                        'ssim': torch.tensor(0.0)
                    }
        except Exception as e:
            print(f"Error loading data at index {index}: {str(e)}")
            # 返回一个有效的默认数据点
            A = torch.zeros((3, 256, 256))  # 假设图像大小为256x256
            return {
                'A': A,
                'B': A,
                'A_paths': '',
                'B_paths': '',
                'is_labeled': torch.tensor(0.0),
                'pseudo_label': A,
                'ssim': torch.tensor(0.0)
            }

    def __len__(self):
        return len(self.labeled_paths) + len(self.unlabeled_paths)

    def update_pseudo_labels(self, model, epoch):
        if epoch % 5 != 0:  # 更频繁地更新伪标签
            return
            
        model.eval()
        with torch.no_grad():
            for path in self.unlabeled_paths:
                try:
                    A_img = Image.open(path).convert('RGB')
                    A = self.transform(A_img).unsqueeze(0).to(model.device)
                    
                    # 使用集成方法生成伪标签
                    predictions = []
                    for _ in range(3):
                        fake_B = model.netG(A)
                        predictions.append(fake_B)
                    
                    # 取平均作为最终预测
                    fake_B = torch.mean(torch.stack(predictions), dim=0)
                    
                    # 计算SSIM
                    ssim = calculate_ssim(A.cpu(), fake_B.cpu())
                    
                    if isinstance(ssim, torch.Tensor):
                        ssim = ssim.item()
                    
                    # 动态调整SSIM阈值
                    base_threshold = 0.7
                    epoch_factor = min(epoch / 100, 1.0)  # 限制在0-1之间
                    threshold = base_threshold + 0.1 * epoch_factor
                    
                    if ssim > threshold:
                        self.pseudo_labels[path] = fake_B.cpu().squeeze(0)
                        self.pseudo_ssim[path] = ssim
                    elif path in self.pseudo_labels:
                        del self.pseudo_labels[path]
                        del self.pseudo_ssim[path]
                except Exception as e:
                    print(f"Error updating pseudo label for {path}: {str(e)}")
                    continue
            model.train() 