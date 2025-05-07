import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# 假设你的 PyTorch 张量存储在变量 tensor 中，形状为 (3, 128, 128)
file_path = './pcd03.pth'  # 设置你的.pth文件路径
tensor = torch.load(file_path)
#tensor = torch.randn(3, 128, 128)  # 随机生成一个张量

# 将张量转换为 PIL 图像
image0 = to_pil_image(tensor)

# 显示图像
plt.imshow(image0)
plt.axis('off')  # 关闭坐标轴
plt.show()


#import matplotlib.pyplot as plt; from torchvision.transforms.functional import to_pil_image