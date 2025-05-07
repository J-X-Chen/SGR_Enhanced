import torch

# 设置随机种子，确保结果可复现
torch.manual_seed(0)

# 创建一个简单的线性模型
# y = w * x + b
w = torch.tensor([1.0], requires_grad=True)  # 权重
b = torch.tensor([0.0], requires_grad=True)  # 偏置

# 定义输入和目标输出
x = torch.tensor([2.0])  # 输入
y_true = torch.tensor([5.0])  # 真实的目标值

# 前向传播
y_pred = w * x + b  # 预测值
loss = (y_pred - y_true) ** 2  # 损失函数，使用平方误差

# 打印前向传播的结果
print("前向传播结果：")
print("预测值 y_pred:", y_pred.item())
print("损失 loss:", loss.item())

# 反向传播
loss.backward()

# 打印梯度
print("\n反向传播后梯度：")
print("权重 w 的梯度:", w.grad.item())
print("偏置 b 的梯度:", b.grad.item())

# 验证梯度是否正确
# 损失函数为 (w*x + b - y_true)^2
# 求导：
# d(loss)/dw = 2 * (w*x + b - y_true) * x
# d(loss)/db = 2 * (w*x + b - y_true)
grad_w = 2 * (w.item() * x.item() + b.item() - y_true.item()) * x.item()
grad_b = 2 * (w.item() * x.item() + b.item() - y_true.item())

print("\n手动计算的梯度：")
print("权重 w 的梯度:", grad_w)
print("偏置 b 的梯度:", grad_b)

# 检查是否一致
print("\n梯度是否一致：")
print("权重梯度一致:", w.grad.item() == grad_w)
print("偏置梯度一致:", b.grad.item() == grad_b)