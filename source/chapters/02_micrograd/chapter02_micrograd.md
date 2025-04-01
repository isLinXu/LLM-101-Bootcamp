# 第02章：Micrograd（机器学习，反向传播）

## 1. 机器学习基础

### 监督学习、无监督学习与强化学习

机器学习是人工智能的一个核心分支，它研究如何让计算机系统从数据中学习并改进其性能，而无需显式编程。根据学习方式和任务类型，机器学习可以分为三大类：监督学习、无监督学习和强化学习。

**监督学习**是最常见的机器学习范式，它使用带有标签的训练数据。在监督学习中，算法通过分析训练样本（输入）及其对应的目标值（输出）来学习输入与输出之间的映射关系。一旦学习完成，算法就能够对新的、未见过的输入数据做出预测。

监督学习的典型应用包括：
- 分类问题：如垃圾邮件检测、图像识别、情感分析等
- 回归问题：如房价预测、股票价格预测、温度预测等

在语言模型的背景下，预测下一个词的任务可以看作是一个监督学习问题，其中输入是前面的词序列，输出是下一个词的概率分布。

**无监督学习**使用的是没有标签的数据。算法需要自行发现数据中的模式、结构或规律，而不依赖于预定义的目标值。无监督学习的主要目标是理解数据的内在结构，而非做出预测。

无监督学习的典型应用包括：
- 聚类：如客户分群、社区发现等
- 降维：如主成分分析(PCA)、t-SNE等
- 异常检测：如信用卡欺诈检测、网络入侵检测等

在语言模型中，词嵌入（如Word2Vec、GloVe）的学习过程可以看作是一种无监督学习，它从大量文本中学习词的分布式表示，而不需要人工标注。

**强化学习**是一种通过与环境交互来学习的方法。在强化学习中，智能体（agent）通过执行动作并观察环境的反馈（奖励或惩罚）来学习最优策略，以最大化长期累积奖励。

强化学习的典型应用包括：
- 游戏AI：如AlphaGo、OpenAI Five等
- 机器人控制：如自主导航、机械臂操作等
- 推荐系统：如新闻推荐、广告投放等

在语言模型的微调阶段，特别是基于人类反馈的强化学习（RLHF）中，强化学习被用来使模型生成的文本更符合人类偏好。

### 损失函数与优化

在机器学习中，我们需要一种方法来衡量模型的预测与真实值之间的差距，这就是**损失函数**（Loss Function）的作用。损失函数将模型的预测与真实标签作为输入，输出一个非负实数，表示预测的"错误程度"。我们的目标是通过调整模型参数，使损失函数的值最小化。

常见的损失函数包括：

1. **均方误差（Mean Squared Error, MSE）**：主要用于回归问题
   $$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
   其中，$y_i$是真实值，$\hat{y}_i$是预测值，$n$是样本数量。

2. **交叉熵损失（Cross-Entropy Loss）**：主要用于分类问题
   $$CE = -\sum_{i=1}^{n}y_i\log(\hat{y}_i)$$
   其中，$y_i$是真实标签（通常是one-hot编码），$\hat{y}_i$是预测的概率分布。

3. **负对数似然（Negative Log-Likelihood, NLL）**：常用于语言模型
   $$NLL = -\sum_{i=1}^{n}\log(P(w_i|w_1, w_2, ..., w_{i-1}))$$
   其中，$P(w_i|w_1, w_2, ..., w_{i-1})$是模型预测的下一个词$w_i$的条件概率。

一旦定义了损失函数，我们需要一种方法来调整模型参数，使损失函数最小化。这个过程称为**优化**（Optimization）。

最常用的优化算法是**梯度下降法**（Gradient Descent）及其变体。梯度下降法的基本思想是沿着损失函数的负梯度方向更新参数，因为负梯度方向是函数值下降最快的方向。

### 梯度下降法

梯度下降法是一种迭代优化算法，用于找到函数的局部最小值。在机器学习中，我们使用梯度下降法来最小化损失函数，从而找到最优的模型参数。

梯度下降法的基本步骤如下：

1. 初始化模型参数（通常是随机初始化）
2. 计算损失函数关于参数的梯度
3. 沿着负梯度方向更新参数
4. 重复步骤2和3，直到收敛（梯度接近零或达到预定的迭代次数）

数学表示为：
$$\theta_{t+1} = \theta_t - \alpha \nabla_{\theta} J(\theta_t)$$

其中，$\theta_t$是当前参数，$\alpha$是学习率（一个控制更新步长的超参数），$\nabla_{\theta} J(\theta_t)$是损失函数$J$关于参数$\theta$的梯度。

梯度下降法有几种变体：

1. **批量梯度下降（Batch Gradient Descent）**：使用所有训练样本计算梯度
   - 优点：每次更新使用所有数据，梯度估计准确
   - 缺点：计算成本高，内存需求大，更新慢

2. **随机梯度下降（Stochastic Gradient Descent, SGD）**：每次只使用一个随机样本计算梯度
   - 优点：更新快，可能跳出局部最小值
   - 缺点：梯度估计噪声大，收敛波动

3. **小批量梯度下降（Mini-batch Gradient Descent）**：使用一小批样本计算梯度
   - 优点：结合了前两者的优点，计算效率和收敛性的良好平衡
   - 缺点：需要调整批量大小这一额外超参数

在实践中，我们通常使用小批量梯度下降及其改进版本，如动量法（Momentum）、AdaGrad、RMSProp和Adam等。这些改进算法通过自适应学习率、加入动量等机制，使优化过程更加稳定和高效。

## 2. 计算图与自动微分

### 前向传播

在神经网络中，**前向传播**（Forward Propagation）是指从输入层到输出层的计算过程。在这个过程中，数据沿着网络的前向方向流动，经过各层的变换，最终产生预测输出。

前向传播可以用**计算图**（Computational Graph）来表示。计算图是一种有向无环图，其中节点表示操作（如加法、乘法、激活函数等），边表示数据流动的方向。

以一个简单的神经网络为例，假设我们有一个具有一个隐藏层的网络，其数学表示为：

$$z = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$$

其中，$x$是输入，$W_1$和$b_1$是第一层的权重和偏置，$\sigma$是激活函数，$W_2$和$b_2$是第二层的权重和偏置，$z$是输出。

前向传播的计算步骤为：
1. 计算第一层的线性变换：$a_1 = W_1 \cdot x + b_1$
2. 应用激活函数：$h_1 = \sigma(a_1)$
3. 计算第二层的线性变换：$z = W_2 \cdot h_1 + b_2$

这个过程可以用计算图表示，其中每个操作都是图中的一个节点，数据沿着边流动。

### 反向传播算法详解

**反向传播**（Backpropagation）是训练神经网络的核心算法，它用于计算损失函数关于网络参数的梯度。反向传播的名称来源于梯度信息从输出层向输入层反向流动的特性。

反向传播算法基于链式法则，它允许我们计算复合函数的导数。在神经网络中，损失函数通常是网络参数的复合函数，我们需要计算损失函数关于每个参数的偏导数，以便使用梯度下降法更新参数。

反向传播的基本步骤如下：

1. **前向传播**：计算网络的输出和损失
2. **计算输出层的梯度**：计算损失函数关于输出层的梯度
3. **反向传播梯度**：使用链式法则，将梯度从输出层反向传播到每一层
4. **更新参数**：使用计算得到的梯度，通过梯度下降法更新网络参数

以上面的简单神经网络为例，假设我们使用均方误差作为损失函数：$L = \frac{1}{2}(z - y)^2$，其中$y$是真实标签。

反向传播的计算步骤为：

1. 计算损失关于输出的梯度：$\frac{\partial L}{\partial z} = z - y$
2. 计算损失关于第二层参数的梯度：
   - $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z} \cdot h_1^T$
   - $\frac{\partial L}{\partial b_2} = \frac{\partial L}{\partial z}$
3. 计算损失关于隐藏层输出的梯度：$\frac{\partial L}{\partial h_1} = W_2^T \cdot \frac{\partial L}{\partial z}$
4. 计算损失关于隐藏层激活前的梯度：$\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial h_1} \odot \sigma'(a_1)$，其中$\odot$表示元素wise乘法
5. 计算损失关于第一层参数的梯度：
   - $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_1} \cdot x^T$
   - $\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial a_1}$

### 链式法则

**链式法则**（Chain Rule）是微积分中的一个基本原理，用于计算复合函数的导数。在神经网络中，链式法则是反向传播算法的数学基础。

对于复合函数$f(g(x))$，其导数可以表示为：
$$\frac{d}{dx}f(g(x)) = \frac{df}{dg} \cdot \frac{dg}{dx}$$

在多变量情况下，如果$y = f(u)$且$u = g(x)$，则：
$$\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

在神经网络中，损失函数通常是网络参数的复杂复合函数。通过链式法则，我们可以将这个复杂的导数计算分解为一系列简单的导数计算，从而高效地计算梯度。

例如，对于一个三层神经网络，损失函数关于第一层权重的梯度可以表示为：
$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial h_2} \cdot \frac{\partial h_2}{\partial a_2} \cdot \frac{\partial a_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial a_1} \cdot \frac{\partial a_1}{\partial W_1}$$

通过链式法则，我们可以从输出层开始，逐层反向计算梯度，最终得到损失函数关于每个参数的梯度。

## 3. Micrograd框架介绍

### Micrograd的设计理念

Micrograd是由Andrej Karpathy创建的一个微型自动微分引擎，它的设计理念是通过最小化的代码实现神经网络的核心功能，包括前向计算和反向传播。Micrograd的目标是帮助人们理解深度学习的基本原理，特别是自动微分和反向传播算法。

Micrograd的主要设计理念包括：

1. **简洁性**：Micrograd的核心代码非常简洁，只有几百行，便于理解和学习。
2. **教育性**：Micrograd的设计目的是教育而非性能，它清晰地展示了自动微分和神经网络的工作原理。
3. **纯Python实现**：Micrograd完全用Python实现，不依赖于其他深度学习库，使得代码易于阅读和理解。
4. **动态计算图**：Micrograd使用动态计算图，这意味着计算图是在运行时构建的，而非预先定义。
5. **标量操作**：为了简化实现，Micrograd主要处理标量操作，而非向量或矩阵操作。

### 核心组件与架构

Micrograd的核心组件是`Value`类，它代表计算图中的一个节点，封装了一个标量值及其梯度。`Value`类支持基本的算术操作（如加法、乘法）和激活函数（如tanh），并能够通过这些操作构建计算图。

Micrograd的架构主要包括以下几个部分：

1. **Value类**：表示计算图中的节点，包含值、梯度和反向传播函数。
2. **操作符重载**：通过重载Python的算术操作符（如+、*），使`Value`对象能够参与算术表达式。
3. **反向传播**：通过拓扑排序和链式法则，实现梯度的反向传播。
4. **神经网络模块**：基于`Value`类构建的简单神经网络组件，如神经元和层。

Micrograd的工作流程如下：

1. 创建`Value`对象，表示输入和参数。
2. 通过算术操作和激活函数，构建计算图，得到输出。
3. 调用输出的`.backward()`方法，触发反向传播，计算梯度。
4. 使用计算得到的梯度，通过梯度下降法更新参数。

## 4. 从零实现Micrograd

### 实现Value类

`Value`类是Micrograd的核心，它封装了一个标量值及其梯度，并支持自动微分。下面是`Value`类的基本实现：

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

这个实现包含了`Value`类的基本功能：
- 初始化方法，设置数据、梯度和反向传播函数
- 加法和乘法操作的重载，支持构建计算图
- tanh激活函数，用于引入非线性
- backward方法，实现反向传播

### 实现基本运算操作

为了使`Value`类更加完整，我们需要实现更多的基本运算操作，如减法、除法、幂运算等。下面是这些操作的实现：

```python
def __neg__(self):
    return self * -1

def __sub__(self, other):
    return self + (-other)

def __rsub__(self, other):
    return other + (-self)

def __truediv__(self, other):
    return self * other**-1

def __rtruediv__(self, other):
    return other * self**-1

def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')
    
    def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward
    
    return out
```

这些方法使`Value`类支持更多的算术操作，从而能够构建更复杂的计算图。

### 实现反向传播

反向传播是自动微分的核心，它通过链式法则计算梯度。在Micrograd中，反向传播通过`backward`方法实现，该方法首先对计算图进行拓扑排序，然后从输出节点开始，反向传播梯度。

拓扑排序确保在计算一个节点的梯度之前，已经计算了所有依赖于该节点的节点的梯度。这是因为根据链式法则，一个节点的梯度依赖于所有使用该节点的节点的梯度。

```python
def backward(self):
    # 拓扑排序
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    
    # 反向传播梯度
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

在这个实现中，我们首先通过深度优先搜索对计算图进行拓扑排序，然后从输出节点开始，按照拓扑排序的逆序反向传播梯度。输出节点的梯度初始化为1.0，表示损失函数关于输出的导数。

## 5. 使用Micrograd构建简单神经网络

### 实现神经网络层

有了`Value`类，我们可以构建简单的神经网络组件，如神经元和层。下面是这些组件的实现：

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

这个实现包括三个类：
- `Neuron`：表示一个神经元，包含权重、偏置和激活函数
- `Layer`：表示一层神经元
- `MLP`（多层感知器）：表示一个多层神经网络

### 训练过程实现

有了神经网络模型，我们可以实现训练过程，包括前向传播、计算损失、反向传播和参数更新。下面是一个简单的训练循环：

```python
# 创建模型
model = MLP(3, [4, 4, 1])

# 训练数据
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # 目标值

# 训练参数
learning_rate = 0.1
epochs = 100

# 训练循环
for epoch in range(epochs):
    # 前向传播
    ypred = [model(x)[0] for x in xs]
    
    # 计算损失
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # 反向传播
    model.zero_grad()  # 清零梯度
    loss.backward()
    
    # 更新参数
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    # 打印损失
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data}')
```

在这个训练循环中，我们首先创建一个多层感知器模型，然后定义训练数据和参数。在每个训练周期，我们执行以下步骤：
1. 前向传播，计算模型的预测输出
2. 计算损失，这里使用均方误差
3. 反向传播，计算梯度
4. 更新参数，使用梯度下降法

### 案例：使用Micrograd解决简单分类问题

下面是一个完整的例子，展示如何使用Micrograd解决一个简单的二分类问题：

```python
import math
import random
import matplotlib.pyplot as plt

# 完整的Value类实现（包括之前的所有方法）
class Value:
    # ... （之前的实现）

# 神经网络组件
class Neuron:
    # ... （之前的实现）

class Layer:
    # ... （之前的实现）

class MLP:
    # ... （之前的实现）

# 生成螺旋数据
def generate_spiral_data(n_points=100, n_classes=2):
    X = []
    y = []
    for i in range(n_classes):
        for j in range(n_points):
            r = j / n_points * 5
            t = 1.25 * j / n_points * 2 * math.pi + i * math.pi
            X.append([r * math.sin(t), r * math.cos(t)])
            y.append(1.0 if i == 0 else -1.0)
    return X, y

# 生成数据
X, y = generate_spiral_data(100, 2)

# 可视化数据
plt.figure(figsize=(5, 5))
plt.scatter([x[0] for i, x in enumerate(X) if y[i] > 0], 
            [x[1] for i, x in enumerate(X) if y[i] > 0], 
            c='r', marker='o', label='Class 1')
plt.scatter([x[0] for i, x in enumerate(X) if y[i] < 0], 
            [x[1] for i, x in enumerate(X) if y[i] < 0], 
            c='b', marker='x', label='Class 2')
plt.legend()
plt.title('Spiral Dataset')
plt.savefig('spiral_data.png')
plt.close()

# 创建模型
model = MLP(2, [16, 16, 1])

# 训练参数
learning_rate = 0.1
epochs = 1000

# 训练循环
losses = []
for epoch in range(epochs):
    # 前向传播
    ypred = [model(x)[0] for x in X]
    
    # 计算损失
    loss = sum((yout - ygt)**2 for ygt, yout in zip(y, ypred)) / len(y)
    losses.append(loss.data)
    
    # 反向传播
    for p in model.parameters():
        p.grad = 0.0  # 清零梯度
    loss.backward()
    
    # 更新参数
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    # 打印损失
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.data:.4f}')

# 可视化损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
plt.close()

# 可视化决策边界
h = 0.01
x_min, x_max = min(x[0] for x in X) - 1, max(x[0] for x in X) + 1
y_min, y_max = min(x[1] for x in X) - 1, max(x[1] for x in X) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.array([[model([x, y])[0].data > 0 for x, y in zip(xx_row, yy_row)] for xx_row, yy_row in zip(xx, yy)])

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter([x[0] for i, x in enumerate(X) if y[i] > 0], 
            [x[1] for i, x in enumerate(X) if y[i] > 0], 
            c='r', marker='o', label='Class 1')
plt.scatter([x[0] for i, x in enumerate(X) if y[i] < 0], 
            [x[1] for i, x in enumerate(X) if y[i] < 0], 
            c='b', marker='x', label='Class 2')
plt.legend()
plt.title('Decision Boundary')
plt.savefig('decision_boundary.png')
plt.close()
```

在这个例子中，我们生成了一个螺旋形的二分类数据集，然后使用Micrograd构建了一个多层感知器模型来解决这个分类问题。我们训练模型1000个周期，并可视化了训练损失和最终的决策边界。

这个例子展示了Micrograd的强大功能：尽管它是一个微型库，但它能够实现完整的神经网络训练过程，并解决实际的机器学习问题。

## 总结

在本章中，我们深入探讨了机器学习的基础概念，包括监督学习、无监督学习和强化学习，以及损失函数和优化算法。我们详细讲解了计算图和自动微分的原理，特别是前向传播和反向传播算法。

我们介绍了Micrograd，一个微型自动微分引擎，并从零开始实现了它的核心功能，包括`Value`类、基本运算操作和反向传播算法。最后，我们使用Micrograd构建了简单的神经网络组件，并展示了如何使用它们解决实际的机器学习问题。

Micrograd的实现虽然简单，但它包含了深度学习的核心原理，为我们理解更复杂的深度学习框架（如PyTorch、TensorFlow）奠定了基础。在接下来的章节中，我们将基于这些基础知识，逐步构建更强大的语言模型。

在下一章中，我们将学习N-gram模型，这是一种更高级的语言模型，它使用多层感知器和矩阵乘法来捕捉更复杂的语言模式。我们还将介绍GELU激活函数，这是现代语言模型中常用的非线性函数。
