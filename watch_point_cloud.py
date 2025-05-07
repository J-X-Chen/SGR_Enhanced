import matplotlib.pyplot as plt
import numpy as np
import torch

file_path = './xyz.pth'  # 设置你的.pth文件路径
points = torch.load(file_path)[10].permute(1,0)
points=points.cpu().detach().numpy()
# 可视化点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[0], points[1], points[2])
plt.show()