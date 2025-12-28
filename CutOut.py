import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Cutout(object):
    """
    # 随机遮挡图像中的多个方形区域
    - n_holes (int): 遮挡区域数量
    - length (int): 每个方形区域的边长（像素）
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes  # 遮挡区域数量
        self.length = length  # 每个方形区域的边长（像素）

    def __call__(self, img):
        h = img.size(1)  # 图像高度（CxHxW）
        w = img.size(2)  # 图像宽度（CxHxW）

        mask = np.ones((h, w), np.float32)  # 初始化全1掩码（1表示保留，0表示遮挡）

        for n in range(self.n_holes):
        	# 随机生成遮挡区域中心坐标(x,y)，确保补丁不超出图像边界
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)  # 计算补丁顶部边界，确保不超出图像顶部
            y2 = np.clip(y + self.length // 2, 0, h)  # 计算补丁底部边界，确保不超出图像底部
            x1 = np.clip(x - self.length // 2, 0, w)  # 计算补丁左侧边界，确保不超出图像左侧
            x2 = np.clip(x + self.length // 2, 0, w)  # 计算补丁右侧边界，确保不超出图像右侧

            mask[y1: y2, x1: x2] = 0.  # 将补丁区域在掩码中设为0，实现遮挡

        # 转换为Tensor并扩展为与图像同尺寸（CxHxW），确保与图像维度匹配
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)  # 扩展掩码维度，与输入图像匹配（CxHxW）
        img = img * mask  # 对图像应用掩码，实现遮挡效果

        return img
