# PyTorch基础- Day03~04

> **目标**：
> 	1、了解机器学习
>
> ​	2、线性回归
>
> ​	3、非线性回归

### 1、机器学习

概述：机器学习就是让机器去观察样本，求解规律函数$g$的过程，使其无限接近目标函数$f$的过程。

![](E:\cv_learn\机器学习模型.jpg)

### 2、线性回归

> 形式类似于高中学过的直线方程 
> $$
> y = kx+b
> $$
> 

其中**$k$**和**$b$**在机器学习中用 $w_1$和$w_0$，称它们为**参数**或**权重**。表示为：
$$
y = w_1x+w_0
$$
这里的$y$表示的是实际数据的值，若想实现预测，要给定一个**目标函数**，可表示为：
$$
\hat{y} = w_1x+w_0
$$
以**房价**和**房间平方**数据为例：

![](E:\cv_learn\线性回归.png)

由图可以看出每个数据样本${\LARGE x^{(i)}}$ 对应的${\LARGE \hat{y}^{(i)}}$和${\LARGE y^{(i)}}$不是完全相等的，所以要用一个函数来衡量它们之间的误差，叫做**损失函数**

#### **损失函数**:

- 均方误差（MSE）:
  $$
  {\LARGE L(w_1, w_0)=\sum_{i=1}^{n}(\hat{y}^{(i)} - y^{(i)})^2 = \sum_{i=1}^{n}(w_1x^{(i)}+w_0-y^{(i)})^2}
  $$

- 平均平方误差：
  $$
  {\LARGE L(w_1, w_0) = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m}\sum_{i=1}^{m}(w_1x^{(i)}+w_0-y^{(i)})^2}
  $$

损失函数$L$实际上是关于参数$(w_1, w_0)$的函数，因此，只要找到一组合适的$(w_1, w_0)$使得$\hat{y}^{(i)}$和${y^{(i)}}$之间的误差最小即可。

**损失函数的简单理解**：

假设$w_0$=0，则损失函数变为(这里使用平均平方误差)：
$$
{\LARGE L(w_1) = \frac{1}{2m} \sum_{i=1}^{m} (w_1x^{(i)}-y^{(i)})^2}
$$


假设有数据集：

|  X   |  Y   |
| :--: | :--: |
|  1   |  1   |
|  2   |  2   |
|  3   |  3   |

接下来**改变**$w_1$，观察损失函数$L(w_1)$的变化：

|  $w_1$   | $\hat{y}$ | $L(w_1)$ |
| :------: | :-------: | :------: |
|    1     |    $x$    |    0     |
|    0     |     0     |   2.33   |
|   0.5    |  $0.5x$   |   0.58   |
| $\cdots$ | $\cdots$  | $\cdots$ |

可以得出，损失函数$L(w_1)$是关于$w_1$的图像类似**抛物线**

若$w_1$和$w_0$都改变，则损失函数$L(w_1, w_0)$图像如下图所示：

![](E:\cv_learn\三维损失函数.jpg)

#### 优化

为了找到$w_1$和$w_0$使$L(w_1, w_0)$最小，可以使用**梯度下降**的算法，梯度向量表示函数增长最快的方向，它的反方向是函数下降最快的方向，最终到达最低点，即**梯度下降**，梯度可以表示为：
$$
{\LARGE \nabla L = (\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_0})}
$$
据此可以算出$L$对$w_1$和$w_0$的偏导，以$L$对$w_1$求偏导为例：
$$
{\LARGE \frac{\partial L}{\partial w_1} = \frac{1}{m}\sum_{i=1}^{m}(w_1x^{(i)}+w_0-y^{(i)})x^{(i)}}
$$
可以用以下方式更新$w_1$和$w_0$的值，使$L$最小：
$$
{\LARGE w_1 = w_1 - \frac{\partial L}{\partial w_1} \times \delta}
$$
$$
{\LARGE w_0 = w_0 - \frac{\partial L}{\partial w_0} \times \delta}
$$

#### 训练

最后进行**训练**:

训练就是不断进行前向传播和反向传播，对参数进行调优，最终让损失函数$L$达到最小值。

一般流程为：

![](\cv_learn\线性回归一般步骤.jpg)

如果把$(w_1, w_0)$看作向量$\vec{w}$，则可以把式子再精简一点，变成向量形式：
$$
{\LARGE \vec{w} = \vec{w} - \nabla L(\vec{w}) \times \delta}
$$
再把$w_0$看作$w_0 \times x_0 (其中 x_0= 1)$，就可以得到更精简的公式：
$$
{\LARGE \hat{y} = \vec{x} \cdot \vec{w}}
$$
其中$\vec{x} = (x_1, x_0)$，$\vec{w} = (w_1, w_0)$

为了让多个数据同时出现在一个公式中出现，$\vec{x}$和$\hat{y}$都要增加一个维度，即$\vec{x}$变为矩阵$\mathbf{X} $，$\hat{y}$变为$\vec{\hat{y}}$，

则公式变为：
$$
{\LARGE \vec{\hat{y}} = \mathbf{X} \cdot \vec{w}}
$$
简单线性回归代码如下：

```python
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
```

#### 使用人工神经元进行线性回归

神经网络的结构图：

![](\cv_learn\神经网络模型.jpg)

神经网络接受多个输入，不同的输入有不同的权重，通过**线性累加**和**激活函数**$\mathbf{f}$得到输出值

在使用人工神经元进行线性回归时，可以省略激活函数，所以可以将模型简化为：
$$
{\LARGE y = x_0w_0 + x_1w_1 + \cdots + x_nw_n = \sum_{i=0}^{n}x_iw_i = \vec{x_i} \cdot \vec{w_i}}
$$
PyTorch内置了损失函数和优化函数，下面用神经网络将上面的代码再编写一遍：

```python
import torch
import matplotlib.pyplot as plt
from torch import nn, optim

# 构造线性模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
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

```

其中：

```python
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

`nn.Linear(1, 1)`的含义是：

- 输入特征数为 1（即每一行有 1 个特征）
- 输出特征数为 1（即每一行输出 1 个值）

`self.linear(x)`的含义为对**x**进行前向运算:
$$
{\LARGE out = \mathbf{X} \cdot x.weight.T + x.bias}
$$


nn模块中预设有均方误差函数`MSELoss()`:

```python
criterion = nn.MSELoss() # 损失函数
```

其数学表达式：
$$
{\LARGE L(x, y) = \frac{1}{n}(x_i-y_i)^2}
$$
使用PyTorch预设的随机梯度下降函数`SGD()`进行更新：

```python
optimizer = optim.SGD(model,paramenters(), learning_rate)
```

优化神经网络中的参数$\vec{w}$

### 3、非线性回归

在神经网络结构图中，我们使用**激活函数**来拟合非线性函数。

#### 激活函数

激活函数是一个非常简单的非线性函数，只要多个激活函数叠加在一起，就可以拟合出复杂的非线性函数，常用的激活函数有sigmoid、tanh、ReLU和Maxout等。

所以，整个人工神经元的数据计算过程如下：
$$
y = f(\vec{x} \cdot \vec{w})
$$
![](E:\cv_learn\神经网络2.jpg)
