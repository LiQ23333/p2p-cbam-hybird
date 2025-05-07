import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_image_range(image):
    """检测图像数据范围"""
    if image.dtype == np.uint8:
        return 255
    elif image.dtype == np.float32 or image.dtype == np.float64:
        return 1.0 if image.max() <= 1.0 else 255.0
    else:
        return image.max() - image.min()

def to_grayscale(image, data_range):
    """根据数据范围转换为灰度图"""
    if image.ndim == 3:
        if data_range == 255:  # 8-bit图像
            return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:  # 浮点图像
            return np.mean(image, axis=2)
    return image

# 加载图像
fake_B = plt.imread('C:/Users/23308/Desktop/myproject/code/pytorch-CycleGAN-and-pix2pix-master/results/dem_experiment_s2/test_40/images/CE2_GRAS_DOM_20m_N001_87S000W_A_12032_2816_fake_B.png')
real_B = plt.imread('C:/Users/23308/Desktop/myproject/code/pytorch-CycleGAN-and-pix2pix-master/results/dem_experiment_s2/test_40/images/CE2_GRAS_DOM_20m_N001_87S000W_A_12032_2816_real_B.png')

# 检测数据范围
data_range = max(get_image_range(fake_B), get_image_range(real_B))

# 转换为灰度图
fake_B_gray = to_grayscale(fake_B, data_range)
real_B_gray = to_grayscale(real_B, data_range)

# 计算指标
mse = mean_squared_error(real_B_gray, fake_B_gray)
mae = mean_absolute_error(real_B_gray, fake_B_gray)
rmse = np.sqrt(mse)  # RMSE是MSE的平方根
ssim_value = ssim(real_B_gray, fake_B_gray, data_range=data_range)

# 绘制对比图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Generated DEM')
plt.imshow(fake_B, cmap='terrain')
plt.subplot(1, 2, 2)
plt.title('Real DEM')
plt.imshow(real_B, cmap='terrain')
plt.show()

# 输出结果
print(f"MSE: {mse:.5f}")
print(f"MAE: {mae:.5f}")
print(f"RMSE: {rmse:.5f}")
print(f"SSIM: {ssim_value:.5f}")