import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.hidden(x)
        # x = F.relu(x)
        x = self.linear(x)
        x = F.softmax(x)
        return x

def draw(x, y, output, loss):
    predict = torch.max(output, 1)[1]
    print((predict == y).float().mean())
    plt.cla()
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], c=output.detach().numpy(), cmap='coolwarm', marker='x')
    # plt.text(0, 0, f'Loss: {loss.item()}', fontsize=12, ha='center')
    # plt.text(0, 0, f'Accuracy: {accuracy.item():.5f}', fontsize=20)
    plt.pause(0.1)

def train(model, criterion, optimizer, x, y, epochs):
    output = None
    loss = None
    for epoch in range(epochs):
        output = model(x)
        loss = criterion(output, y)

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

    # 绿色样本
    data2 = torch.normal(sample*8, 2)
    label2 = label0*2

    # 合并
    x = torch.cat((data0, data1, data2))
    y = torch.cat((label0, label1, label2), dim=0).long()

    model = LogisticRegression(2, 2, 3)

    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.05) # 优化器

    output, loss = train(model, criterion, optimizer, x, y, epochs=10000)
    print(output)
    plt.show()

if __name__ == '__main__':
    main()