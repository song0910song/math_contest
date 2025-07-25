import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from PIL import Image

# 1. 读取图片并转为灰度
img = Image.open(r'E:\cv_learn\img\1.png').convert('L')
img_arr = np.array(img)
img_tensor = torch.tensor(img_arr, dtype=torch.float32)

# 2. 构造标签（假设目标是区分黑/白像素）
target = (img_tensor < 128).float().unsqueeze(1)
print(img_tensor.shape)

# 3. 构建模型
class ImageNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.hidden(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)
        return x

# 4. 数据预处理：每行作为一个样本
x = img_tensor
x = x.view(-1, 1)  # shape [H*W, 1]
target = target.view(-1, 1)

model = ImageNet(1, 20, 1)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. 训练
output = None
for epoch in range(1000):
    output = model(x)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 6. 预测与可视化
with torch.no_grad():
    pred = (model(x) > 0.5).float().view(img_tensor.shape)
plt.imshow(pred, cmap='gray')
plt.title('Predicted Mask')
plt.show()
print(output)