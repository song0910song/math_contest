# PyTorch基础-Day02

---

## Autograd自动微分

#### 1、基本原理

- 简单的计算可以把计算抽象为图像

  ![](E:\cv_learn\autograd.jpg)

  - 复杂计算图

    <img src="E:\cv_learn\2.jpg" alt="2" style="zoom:50%;" />

    一张复杂的图可以分为4部分：**叶子节点、中间节点、输出节点和信息流**

    **叶子节点**是图的末端，在神经网络模型中就是输入值和神经网络参数。

    Tensor在自动微分方面有3个重要属性：

    	- requires_grad
    	- grad 
    	- grad_fn

    **requires_grad** 是一个布尔值 ，默认为False, 为True时表示自动微分

    **grad**储存Tensor的微分值

    **grad_fn** 储存Tensor的微分函数

    只要在*输出节点*调用反向传播函数`backward()`,PyTorch就会自动求出叶子节点的微分值并储存在叶子节点grad中。

#### 2、前向传播

从叶子节点开始追踪信息流，记下整个过程使用的函数，直到输出节点。

### 3、反向传播

调用输出节点的`backward()`函数对整个图进行反向传播，求出微分值。

#### 4、非标量输出

当输出节点为非标量值时，`backward()`需要增加一个参数`gradient`,其形状应该与输出节点的形状保持一致且元素值均为1.