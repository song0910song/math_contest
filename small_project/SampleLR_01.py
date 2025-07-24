'''
直接根据y = w1*x + w0
运用梯度下降算法计算w1, w0的值
'''

import torch
import matplotlib.pyplot as plt

def draw(input, target, output, loss):
    '''
    绘制图形
    output: 模型输出
    loss: 损失值
    '''
    plt.cla()
    plt.scatter(input.numpy(), target.numpy())
    plt.plot(input.numpy(), output.data.numpy(), 'r-', lw=5)
    plt.text(min(input.numpy())+0.5, min(target.numpy()), 
             'loss=%s' % (loss.item()), fontdict={'size':20, 'color':'red'})
    plt.pause(0.055)

def train(input, target, w1, w0, epochs=1, learning_rate=0.01):
    '''
    epochs: 训练次数
    learning_rate: 学习率
    '''
    loss = None
    for epoch in range(epochs):
        output = w1*input + w0
        loss = (output - target).pow(2).sum()/(2*len(input))

        loss.backward() # 向后传播计算梯度

        # 梯度下降算法
        w1.data = w1 - learning_rate*w1.grad
        w0.data = w0 - learning_rate*w0.grad

        # 避免梯度累积
        w1.grad.zero_()
        w0.grad.zero_()

        if epoch % 80 == 0:
            draw(input, target,output, loss)

    return w1, w0, loss


def main():
    x = torch.linspace(-3, 2, 1000)
    y = x + 1.2*torch.rand(x.size())

    input = x
    target = y

    w1 = torch.rand(1, requires_grad=True)
    w0 = torch.rand(1, requires_grad=True)

    w1, w0, loss = train(input, target, w1, w0, 10000, learning_rate=1e-4)

    print(f"最终损失值：{loss.item()}")
    print(f"w1: {w1.item()}, w0: {w0.item()}")
    plt.show()
    
main()