import torch
import matplotlib.pyplot as plt


def draw(output, loss):
    '''
    绘制图形
    output: 模型输出
    loss: 损失值
    '''
    plt.cla()
    plt.scatter(x.numpy(), y.numpy())
    plt.plot(x.numpy(), output.data.numpy(), 'r-', lw=5)
    plt.text(min(x.numpy())+0.5, min(y.numpy()), 
             'loss=%s' % (loss.item()), fontdict={'size':20, 'color':'red'})
    plt.pause(0.055)

def train(epochs=1, learning_rate=0.01):
    '''
    epochs: 训练次数
    learning_rate: 学习率
    '''
    for epoch in range(epochs):
        output = w1*x + w0
        loss = (output - target).pow(2).sum()/(2*len(x))
        
        loss.backward() # 向后传播计算梯度

        # 梯度下降算法
        w1.data = w1 - learning_rate*w1.grad
        w0.data = w0 - learning_rate*w0.grad

        # 避免梯度累积
        w1.grad.zero_()
        w0.grad.zero_()

        if epoch % 80 == 0:
            draw(output, loss)

    return w1, w0, loss


x = torch.linspace(-3, 2, 1000)
y = x + 1.2*torch.rand(x.size())

input = x
target = y

w1 = torch.rand(1, requires_grad=True)
w0 = torch.rand(1, requires_grad=True)

w1, w0, loss = train(100000, learning_rate=1e-4)

print(f"最终损失值：{loss.item()}")
print(f"w1: {w1.item()}, w0: {w0.item()}")
plt.show()
    