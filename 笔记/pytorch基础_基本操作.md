# PyTorch基础（Day-01)

---

> **目标** ：
>
> - 理解张量的概念
> - 使用pytorch实现张量的基本操作

---

## 1、张量（Tensor)

> Tensor之于pytorch，相当于Array之于Numpy

Pytorch里面的张量相当于Numpy中的Array，**张量（Tensor）**是Pytorch里的基本数据类型。

标量是零维张量，向量是一维张量，矩阵是二维张量。

## 2、PyTorch基本操作

---

#### 基本创建方法

| 方法         | 说明                                     |
| ------------ | ---------------------------------------- |
| Tensor()     | 随机创建指定维度的Tensor                 |
| eye()        | 创建对角线全为1的Tensor                  |
| from_numpy() | 将ndarray对象转为Ternsor                 |
| linspace()   | 创建一个区间被均匀划分的一维Tensor       |
| arange()     | 创建一个区间内以固定步长递增的一维Tensor |
| ones()       | 创建**全1张量**                          |
| zeros()      | 创建**全0张量**                          |
| rand()       | 创建**[0,1)区间的均匀分布**随机张量      |
| randn()      | 创建**标准正态分布**随机张量             |

#### Tensor数据类型

| 数据类型      | torch 类型              | 字符串表示   | 说明                                | 示例值                |
| :------------ | :---------------------- | :----------- | :---------------------------------- | :-------------------- |
| **32位浮点**  | `torch.float32`         | `'float32'`  | 标准单精度浮点数 (默认类型)         | `3.141592`            |
| **64位浮点**  | `torch.float64`         | `'float64'`  | 双精度浮点数                        | `3.141592653589793`   |
| **16位浮点**  | `torch.float16`         | `'float16'`  | 半精度浮点数 (GPU加速常用)          | `3.14`                |
| **BFloat16**  | `torch.bfloat16`        | `'bfloat16'` | 脑浮点数 (专为AI设计)               | `3.14`                |
| **8位整数**   | `torch.int8`            | `'int8'`     | 有符号8位整数 (-128~127)            | `-42`                 |
| **16位整数**  | `torch.int16` / `short` | `'int16'`    | 有符号16位整数                      | `30000`               |
| **32位整数**  | `torch.int32` / `int`   | `'int32'`    | 有符号32位整数                      | `2147483647`          |
| **64位整数**  | `torch.int64` / `long`  | `'int64'`    | 有符号64位整数 (索引默认类型)       | `9223372036854775807` |
| **8位无符号** | `torch.uint8`           | `'uint8'`    | 无符号8位整数 (0~255, 图像数据常用) | `255`                 |
| **布尔型**    | `torch.bool`            | `'bool'`     | 布尔类型 (True/False)               | `True`                |

#### 重要运算函数

`torch.dot()`向量与向量的**点积**
$$
\vec{a} \cdot \vec{b} = a_{1}b_{1} +  a_{2}b_{2} + ... + a_{n}b_{n} =  {\textstyle \sum_{i=1}^{n}a_{i}b_{i}} 
$$


`torch.mv()`向量与矩阵的乘法
$$
\left[\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 m} \\
a_{21} & a_{22} & \cdots & a_{2 m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n 1} & a_{n 2} & \cdots & a_{n m}
\end{array}\right]\left[\begin{array}{c}
b_{1} \\
b_{2} \\
\vdots \\
b_{m}
\end{array}\right]=\left[\begin{array}{c}
a_{11} b_{1}+a_{12} b_{2}+\cdots+a_{1 m} b_{m} \\
a_{21} b_{1}+a_{22} b_{2}+\cdots+a_{2 m} b_{m} \\
\vdots \\
a_{n 1} b_{1}+a_{n 2} b_{2}+\cdots+a_{n m} b_{m}
\end{array}\right]
$$
`torch.mm()`矩阵与矩阵的乘法
$$
\begin{bmatrix}
  &1  &2  &3 \\
  &2  & 3 &4 \\
  &3  &4  &5 \\
\end{bmatrix}
\begin{bmatrix}
  &2  & 3 &4 \\
  &3  &4  &5 \\
  &4  &5  &6
\end{bmatrix}
=
\begin{bmatrix}
  &20  &26 &32 \\
  &29  &38  &47 \\
  &38  &50  &62
\end{bmatrix}
$$
以矩阵中20为例，展示计算过程：

![](E:\cv_learn\1.jpg)
