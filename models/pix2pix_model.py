import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='semi_supervised')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_pseudo', type=float, default=50.0, help='weight for pseudo label loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_pseudo']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'D']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.lambda_L1 = opt.lambda_L1
        self.lambda_pseudo = opt.lambda_pseudo
        self.opt = opt
        
        # 初始化损失属性
        self.loss_D_real = torch.tensor(0.0, device=self.device)
        self.loss_D_fake = torch.tensor(0.0, device=self.device)
        self.loss_D = torch.tensor(0.0, device=self.device)
        self.loss_G_GAN = torch.tensor(0.0, device=self.device)
        self.loss_G_L1 = torch.tensor(0.0, device=self.device)
        self.loss_G_pseudo = torch.tensor(0.0, device=self.device)
        self.loss_G = torch.tensor(0.0, device=self.device)
        
        # 初始化伪标签
        self.pseudo_label = None
        self.pseudo_label_ema = None
        self.ema_alpha = 0.9  # 指数移动平均的权重

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']
        
        # 处理批处理中的is_labeled值
        if isinstance(input.get('is_labeled', True), torch.Tensor):
            # 如果所有样本都是已标记的，则is_labeled为True
            self.is_labeled = bool(input['is_labeled'].all().item())
        else:
            self.is_labeled = bool(input.get('is_labeled', True))
        
        if self.is_labeled:
            self.real_B = input['B'].to(self.device)
        else:
            self.pseudo_label = input.get('pseudo_label', None)
            if self.pseudo_label is not None:
                self.pseudo_label = self.pseudo_label.to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        if self.is_labeled:
            # 有标签数据的判别器损失
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()

    def backward_G(self):
        if self.is_labeled:
            # 有标签数据的生成器损失
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
            
            # 添加L2正则化
            l2_reg = 0.0
            for param in self.netG.parameters():
                l2_reg += torch.norm(param)
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + 0.0001 * l2_reg
        else:
            # 无标签数据的生成器损失
            self.loss_G_pseudo = torch.tensor(0.0, device=self.device)
            if hasattr(self, 'pseudo_label') and self.pseudo_label is not None:
                self.loss_G_pseudo = self.criterionL1(self.fake_B, self.pseudo_label) * self.lambda_pseudo
            self.loss_G = self.loss_G_pseudo
        
        self.loss_G.backward()

    def update_learning_rate(self):
        """更新学习率"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        new_lr = old_lr * 0.95  # 每5个epoch衰减5%
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def update_loss_weights(self, epoch):
        """动态调整损失权重"""
        # 动态调整L1损失权重
        self.lambda_L1 = self.opt.lambda_L1 * (1.0 - 0.1 * (epoch / self.opt.n_epochs))
        
        # 动态调整伪标签损失权重
        if epoch < 20:
            self.lambda_pseudo = self.opt.lambda_pseudo
        else:
            self.lambda_pseudo = self.opt.lambda_pseudo * (1.0 - 0.5 * ((epoch - 20) / (self.opt.n_epochs - 20)))

    def update_pseudo_labels(self, epoch):
        """更新伪标签"""
        if epoch % 5 == 0:  # 每5个epoch更新一次伪标签
            with torch.no_grad():
                if self.pseudo_label is None or self.pseudo_label_ema is None:
                    # 确保fake_B已经被生成
                    if not hasattr(self, 'fake_B'):
                        self.forward()
                    # 初始化伪标签
                    self.pseudo_label = self.fake_B.clone()
                    self.pseudo_label_ema = self.fake_B.clone()
                else:
                    # 使用指数移动平均更新伪标签
                    self.pseudo_label_ema = self.ema_alpha * self.pseudo_label_ema + (1 - self.ema_alpha) * self.fake_B
                    self.pseudo_label = self.pseudo_label_ema.clone()

    def optimize_parameters(self):
        # 生成器前向传播
        self.forward()                   # compute fake images: G(A)
        
        # 判别器训练
        if self.is_labeled:  # 只在有标签数据时更新判别器
            self.optimizer_D.zero_grad()
            
            # 添加噪声到真实和生成图像
            real_noise = torch.randn_like(self.real_B) * 0.1
            fake_noise = torch.randn_like(self.fake_B) * 0.1
            
            # 真实图像
            real_AB = torch.cat((self.real_A, self.real_B + real_noise), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            
            # 生成图像
            fake_AB = torch.cat((self.real_A, self.fake_B.detach() + fake_noise), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            
            # 添加梯度惩罚
            alpha = torch.rand(self.real_B.size(0), 1, 1, 1).to(self.device)
            interpolated = alpha * self.real_B + (1 - alpha) * self.fake_B.detach()
            interpolated_AB = torch.cat((self.real_A, interpolated), 1)
            interpolated_AB.requires_grad_(True)
            pred_interpolated = self.netD(interpolated_AB)
            gradients = torch.autograd.grad(
                outputs=pred_interpolated,
                inputs=interpolated_AB,
                grad_outputs=torch.ones_like(pred_interpolated),
                create_graph=True
            )[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            # 计算总损失
            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5 + 10.0 * gradient_penalty
            
            # 反向传播
            self.loss_D.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
            
            self.optimizer_D.step()

        # 生成器训练
        self.optimizer_G.zero_grad()
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        
        # 计算生成器损失
        if self.is_labeled:
            # 有标签数据的生成器损失
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        else:
            # 无标签数据的生成器损失
            self.loss_G_pseudo = torch.tensor(0.0, device=self.device)
            if hasattr(self, 'pseudo_label') and self.pseudo_label is not None:
                self.loss_G_pseudo = self.criterionL1(self.fake_B, self.pseudo_label) * self.lambda_pseudo
            self.loss_G = self.loss_G_pseudo
        
        # 反向传播
        self.loss_G.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        
        self.optimizer_G.step()
