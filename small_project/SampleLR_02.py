'''
线性回归：使用矩阵X和向量w
'''

import torch
import matplotlib.pyplot as plt

def produce_x(x):
    '''
    生成矩阵X
    '''
    x0 = torch.ones_like(x)
    X = torch.stack([x, x0], dim=1)
    return X


def draw(x, y, output, loss):
    plt.cla()
    plt.scatter(x.numpy(), y.numpy())
    plt.plot(x.numpy(), output.detach().numpy(), 'r-', lw=5)
    plt.text(min(x.numpy()) + 0.5, min(y.numpy()),
             f'loss={loss.item()}', fontdict={'size': 18, 'color': 'red'})
    plt.pause(0.005)

def train(input, target, w, epochs=1, learning_rate=0.01):
    for epoch in range(epochs):
        output = input.mv(w)
        loss = (output - target).pow(2).sum() / (2 * len(input))
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
        w.grad.zero_()

        if epoch % 40 == 0:
            draw(input[:, 0], target, output, loss)

    return w, loss

x = torch.linspace(-4, 4, 1000)
y = x + 1.3 * torch.rand(x.size())
w = torch.rand(2, requires_grad=True)
X = produce_x(x)

w, loss = train(X, y, w, epochs=100000, learning_rate=1e-3)

print(f"最终损失函数值：{loss.item():.4f}")
print(f"权重w：{w.data}")

plt.show()