import torch
import matplotlib.pyplot as plt
from torch import nn, optim

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        # 输入特征数为1，输出特征数为1
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def draw(x, y, output, loss):
    """Plot data and model prediction."""
    plt.cla()
    plt.scatter(x.numpy(), y.numpy())
    plt.plot(x.numpy(), output.detach().numpy(), 'r-', lw=2)
    plt.text(float(x.min()) + 0.5, float(y.min()),
             f'loss={loss.item()}', fontdict={'size': 14, 'color': 'red'})
    plt.draw()
    plt.pause(0.05)

def train(model, criterion, optimizer, x, y, epochs):
    """Train the model."""
    for epoch in range(epochs):
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 80 == 0:
            draw(x, y, output, loss)
    return model, loss

def main():
    x = torch.linspace(-3, 2, 1000).unsqueeze(1)
    y = x + 1.2 * torch.rand(x.size())
    model = LR()
    criterion = nn.MSELoss() # 损失函数
    optimizer = optim.SGD(model.parameters(), lr=1e-4) # 优化器
    model, loss = train(model, criterion, optimizer, x, y, epochs=100000)
    print(f"Final loss: {loss.item():.4f}")
    print(f"w1: {model.linear.weight.item():.4f}, w0: {model.linear.bias.item():.4f}")
    plt.show()

if __name__ == "__main__":
    main()
