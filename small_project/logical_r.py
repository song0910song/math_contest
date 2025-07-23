import torch
from torch import nn, optim
import matplotlib.pylab as plt

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

def draw(x, y, output, loss):
    plt.cla()
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], c=output.detach().numpy(), cmap='coolwarm', marker='x')
    plt.text(0, 0, f'Loss: {loss.item()}', fontsize=12, ha='center')
    plt.pause(0.1)

def train(model, criterion, optimizer, x, y, epochs):
    for epoch in range(epochs):
        output = model(x)
        loss = criterion(output, y.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward() # 反向传播
        optimizer.step()

        if epoch % 30 == 0:
            draw(x, y, output, loss)

    return output, loss
    

def main():
    sample = torch.ones(500, 2)

    # 红色样本
    data0 = torch.normal(sample*4, 2)
    label0 = torch.ones(500)

    # 蓝色样本
    data1 = torch.normal(-4*sample, 2)
    label1 = torch.zeros(500)

    # 合并
    x = torch.cat((data0, data1))
    y = torch.cat((label0, label1))

    model = LogisticRegression()

    criterion = nn.BCELoss() # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.02) # 优化器

    output, loss = train(model, criterion, optimizer, x, y, epochs=500)
    pre = (output >= 0.5).float()
    print((pre == y.unsqueeze(1).float()).float().mean())
    plt.show()

main()