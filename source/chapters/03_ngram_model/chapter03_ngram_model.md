---
file_format: mystnb
kernelspec:
  name: python3
---
# 第03章：N-gram模型（多层感知器，矩阵乘法，GELU激活函数）

## 1. 从Bigram到N-gram

### N-gram模型的数学定义

在第一章中，我们详细介绍了Bigram模型，它基于一阶马尔可夫假设，认为一个词的出现只与其前一个词相关。然而，这种简化假设限制了模型捕捉更长距离依赖关系的能力。为了克服这一限制，我们可以扩展到更高阶的N-gram模型。

N-gram模型是一种基于(N-1)阶马尔可夫假设的语言模型，它假设一个词的出现只与其前面的N-1个词相关。形式化地，N-gram模型计算的条件概率为：

$$P(w_i|w_{i-(N-1)}, w_{i-(N-2)}, ..., w_{i-1})$$

例如：
- Unigram (N=1): $P(w_i)$
- Bigram (N=2): $P(w_i|w_{i-1})$
- Trigram (N=3): $P(w_i|w_{i-2}, w_{i-1})$
- 4-gram (N=4): $P(w_i|w_{i-3}, w_{i-2}, w_{i-1})$

在N-gram模型中，一个句子的概率可以表示为：

$$P(w_1, w_2, ..., w_m) = \prod_{i=1}^{m} P(w_i|w_{i-(N-1)}, w_{i-(N-2)}, ..., w_{i-1})$$

其中，对于$i < N$的情况，我们通常使用特殊的开始标记（如`<s>`）来填充序列的开始部分。

与Bigram模型类似，N-gram模型的条件概率通常使用最大似然估计来计算：

$$P(w_i|w_{i-(N-1)}, ..., w_{i-1}) = \frac{count(w_{i-(N-1)}, ..., w_{i-1}, w_i)}{count(w_{i-(N-1)}, ..., w_{i-1})}$$

其中，$count(w_{i-(N-1)}, ..., w_{i-1}, w_i)$是N个词的序列在语料库中出现的次数，$count(w_{i-(N-1)}, ..., w_{i-1})$是N-1个词的序列在语料库中出现的次数。

### 高阶N-gram的优势与挑战

高阶N-gram模型相比Bigram模型有以下优势：

1. **捕捉更长距离的依赖关系**：高阶N-gram模型考虑了更多的上下文信息，能够捕捉更长距离的词间依赖关系。例如，Trigram模型可以考虑前两个词的影响，而不仅仅是前一个词。

2. **更准确的概率估计**：通过考虑更多的上下文，高阶N-gram模型通常能够提供更准确的下一个词的概率估计，从而生成更流畅、更符合语法的文本。

3. **更好的语言理解**：高阶N-gram模型能够更好地理解语言的结构和模式，特别是对于那些依赖于较长上下文的语言现象（如一些固定搭配、习语等）。

然而，高阶N-gram模型也面临一些挑战：

1. **数据稀疏性问题加剧**：随着N的增加，可能的N-gram组合数量呈指数级增长，而大多数组合在语料库中可能很少出现或根本不出现。这导致了更严重的数据稀疏性问题。

2. **存储和计算开销增加**：高阶N-gram模型需要存储和处理更多的N-gram统计信息，增加了存储和计算的开销。

3. **泛化能力有限**：尽管高阶N-gram模型考虑了更多的上下文，但它们仍然基于离散的词序列统计，无法像神经网络模型那样学习词的分布式表示和更复杂的语言模式。

为了解决这些挑战，研究者提出了各种改进方法，如更复杂的平滑技术、回退模型、插值模型等。然而，这些方法仍然无法从根本上解决N-gram模型的局限性。随着深度学习的发展，基于神经网络的语言模型逐渐取代了传统的N-gram模型，成为语言建模的主流方法。

## 2. 多层感知器(MLP)基础

### 感知器模型

感知器（Perceptron）是神经网络的基本构建块，由Frank Rosenblatt在1957年提出。它是一种简单的二分类线性分类器，可以看作是一个单个神经元的模型。

感知器的基本结构包括：
- 输入特征 $x = (x_1, x_2, ..., x_n)$
- 权重 $w = (w_1, w_2, ..., w_n)$
- 偏置 $b$
- 激活函数 $f$

感知器的输出计算如下：
$$y = f(w \cdot x + b) = f(\sum_{i=1}^{n} w_i x_i + b)$$

其中，$f$通常是一个阶跃函数（step function），如：
$$f(z) = \begin{cases} 
1, & \text{if } z \geq 0 \\
0, & \text{if } z < 0
\end{cases}$$

感知器可以学习线性可分的问题，通过调整权重和偏置，使得模型的预测与真实标签尽可能一致。然而，感知器无法解决非线性可分的问题，如经典的XOR问题。

### 多层网络结构

为了克服单层感知器的局限性，研究者提出了多层感知器（Multi-Layer Perceptron, MLP），也称为前馈神经网络（Feedforward Neural Network）。MLP由多层神经元组成，包括：
- 输入层：接收输入特征
- 隐藏层：一个或多个中间层，执行非线性变换
- 输出层：产生最终预测

在MLP中，每一层的神经元接收前一层所有神经元的输出作为输入，并将其输出传递给下一层的所有神经元。这种结构允许网络学习更复杂的非线性映射。

形式化地，对于一个具有L层的MLP，第l层的输出可以表示为：
$$h^{(l)} = f^{(l)}(W^{(l)} h^{(l-1)} + b^{(l)})$$

其中，$h^{(l)}$是第l层的输出，$W^{(l)}$是第l层的权重矩阵，$b^{(l)}$是第l层的偏置向量，$f^{(l)}$是第l层的激活函数。特别地，$h^{(0)} = x$是输入特征。

MLP的关键特性是引入了非线性激活函数，如sigmoid、tanh或ReLU，使网络能够学习非线性映射。没有这些非线性函数，多层网络将等价于单层线性模型。

### 前向传播算法

前向传播（Forward Propagation）是神经网络中从输入到输出的计算过程。在MLP中，前向传播按照从输入层到输出层的顺序，逐层计算每一层的输出。

对于一个L层的MLP，前向传播的步骤如下：

1. 初始化：$h^{(0)} = x$（输入特征）
2. 对于每一层 $l = 1, 2, ..., L$：
   a. 计算线性变换：$z^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$
   b. 应用激活函数：$h^{(l)} = f^{(l)}(z^{(l)})$
3. 输出：$y = h^{(L)}$

在语言模型中，MLP可以用来学习词嵌入之间的关系，或者作为更复杂网络架构（如Transformer）的组件。例如，在Transformer的前馈网络部分，就使用了MLP来处理注意力机制的输出。

## 3. 矩阵乘法在深度学习中的应用

### 矩阵乘法基础

矩阵乘法是深度学习中最基本、最常用的操作之一。在神经网络中，权重通常表示为矩阵，输入和激活值表示为向量，它们之间的线性变换通过矩阵乘法实现。

给定一个权重矩阵 $W \in \mathbb{R}^{m \times n}$ 和一个输入向量 $x \in \mathbb{R}^{n}$，线性变换的结果是一个向量 $z \in \mathbb{R}^{m}$，计算如下：
$$z = Wx$$

其中，$z_i = \sum_{j=1}^{n} W_{ij} x_j$，即矩阵的第i行与向量x的点积。

在深度学习框架中，矩阵乘法通常通过高度优化的线性代数库（如BLAS、cuBLAS）实现，以充分利用现代CPU和GPU的并行计算能力。

### 向量化计算的优势

向量化计算是指使用矩阵和向量操作代替循环的编程范式。在深度学习中，向量化计算有以下优势：

1. **计算效率**：现代硬件（特别是GPU）针对矩阵运算进行了优化，向量化计算可以充分利用这些优化，大大提高计算效率。

2. **代码简洁**：向量化计算使代码更加简洁、易读，减少了显式循环的使用。

3. **并行处理**：矩阵运算可以自然地并行化，使得模型能够高效地处理大批量数据。

4. **数值稳定性**：优化的线性代数库通常实现了数值稳定的算法，减少了浮点误差的累积。

例如，考虑一个具有n个输入特征和m个输出特征的全连接层，非向量化的实现可能需要两层嵌套循环：

```python
# 非向量化实现
z = np.zeros(m)
for i in range(m):
    for j in range(n):
        z[i] += W[i, j] * x[j]
    z[i] += b[i]
```

而向量化实现只需要一行代码：

```python
# 向量化实现
z = np.dot(W, x) + b
```

向量化实现不仅代码更简洁，而且计算效率通常要高出几个数量级。

### 批处理技术

在深度学习中，我们通常不是一次处理一个样本，而是同时处理一批（batch）样本。批处理可以进一步提高计算效率，并且有助于稳定训练过程。

给定一个批次的输入 $X \in \mathbb{R}^{n \times b}$，其中b是批次大小，n是特征维度，线性变换的结果是 $Z \in \mathbb{R}^{m \times b}$，计算如下：
$$Z = WX$$

其中，$W \in \mathbb{R}^{m \times n}$ 是权重矩阵。

批处理的优势包括：

1. **计算效率**：批处理允许更高的计算并行度，特别是在GPU上，可以显著提高吞吐量。

2. **内存访问效率**：批处理可以更有效地利用缓存和内存带宽，减少内存访问的开销。

3. **训练稳定性**：批处理使用多个样本的平均梯度更新参数，减少了梯度的方差，使训练过程更加稳定。

4. **批归一化**：批处理使得批归一化（Batch Normalization）等技术成为可能，这些技术可以加速训练并提高模型性能。

在语言模型中，批处理通常涉及同时处理多个序列。为了处理不同长度的序列，我们通常使用填充（padding）和掩码（masking）技术。

## 4. 激活函数详解

### 常见激活函数比较

激活函数是神经网络中引入非线性的关键组件。没有激活函数，多层神经网络将等价于单层线性模型，无法学习复杂的非线性映射。以下是几种常见的激活函数及其特点：

1. **Sigmoid函数**：
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
   - 输出范围：(0, 1)
   - 优点：平滑、可微、输出可解释为概率
   - 缺点：存在梯度消失问题、输出不是零中心的、计算开销较大

2. **Tanh函数**：
   $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
   - 输出范围：(-1, 1)
   - 优点：平滑、可微、输出是零中心的
   - 缺点：仍然存在梯度消失问题、计算开销较大

3. **ReLU（Rectified Linear Unit）**：
   $$\text{ReLU}(x) = \max(0, x)$$
   - 输出范围：[0, +∞)
   - 优点：计算简单、收敛快、缓解梯度消失问题
   - 缺点：存在"死亡ReLU"问题（某些神经元可能永远不会激活）、输出不是零中心的

4. **Leaky ReLU**：
   $$\text{LeakyReLU}(x) = \begin{cases} 
   x, & \text{if } x \geq 0 \\
   \alpha x, & \text{if } x < 0
   \end{cases}$$
   其中，$\alpha$是一个小正数（如0.01）。
   - 输出范围：(-∞, +∞)
   - 优点：解决了"死亡ReLU"问题、其他优点与ReLU类似
   - 缺点：引入了额外的超参数$\alpha$

5. **ELU（Exponential Linear Unit）**：
   $$\text{ELU}(x) = \begin{cases} 
   x, & \text{if } x \geq 0 \\
   \alpha (e^x - 1), & \text{if } x < 0
   \end{cases}$$
   其中，$\alpha$是一个正数（通常为1）。
   - 输出范围：(-α, +∞)
   - 优点：可以产生负输出、缓解"死亡ReLU"问题、导数平滑
   - 缺点：计算开销较大、引入了额外的超参数

6. **GELU（Gaussian Error Linear Unit）**：
   $$\text{GELU}(x) = x \cdot \Phi(x)$$
   其中，$\Phi(x)$是标准正态分布的累积分布函数。
   - 输出范围：(-∞, +∞)，但主要集中在(-1, +∞)
   - 优点：平滑、可微、在负值区域有非零梯度、性能优越
   - 缺点：计算开销较大

### GELU激活函数的特点

GELU（Gaussian Error Linear Unit）激活函数由Dan Hendrycks和Kevin Gimpel在2016年提出，它在现代语言模型（如BERT、GPT等）中被广泛使用。GELU的数学定义为：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

其中，$\Phi(x)$是标准正态分布的累积分布函数，$\text{erf}$是误差函数。

GELU可以看作是ReLU和Leaky ReLU的平滑版本，它具有以下特点：

1. **平滑性**：GELU是一个平滑函数，在所有点都可微，这有助于梯度的稳定传播。

2. **非线性**：GELU在正值区域近似于恒等函数，在负值区域有非零输出，但随着输入变得更负，输出趋近于零。

3. **自正则化**：GELU可以看作是一种自正则化的激活函数，它根据输入的值随机"丢弃"一些神经元的激活，类似于Dropout的效果。

4. **理论基础**：GELU的设计基于高斯误差函数，有着良好的理论基础。

在实践中，由于精确计算GELU的计算开销较大，通常使用以下近似公式：

$$\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$

或者更简单的近似：

$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$

其中，$\sigma$是sigmoid函数。

### 为什么现代语言模型选择GELU

现代语言模型（如BERT、GPT、T5等）普遍选择GELU作为激活函数，主要有以下原因：

1. **性能优越**：在多项实验中，GELU表现出比ReLU和其他激活函数更好的性能，特别是在语言模型任务上。

2. **平滑性**：GELU是一个平滑函数，这有助于梯度的稳定传播，减少训练过程中的波动。

3. **自正则化特性**：GELU的自正则化特性有助于防止过拟合，特别是在大型模型中。

4. **与注意力机制的兼容性**：GELU与Transformer架构中的注意力机制配合良好，有助于模型学习复杂的语言模式。

5. **经验证明**：大量实践表明，使用GELU的模型通常收敛更快，最终性能也更好。

值得注意的是，虽然GELU在语言模型中表现优越，但在其他类型的神经网络中，ReLU及其变体仍然是常用的选择。激活函数的选择应该根据具体任务和模型架构来决定。

## 5. 实现基于MLP的N-gram模型

### 模型架构设计

传统的N-gram模型基于计数统计，面临数据稀疏性和泛化能力有限的问题。我们可以使用神经网络，特别是多层感知器（MLP），来构建更强大的N-gram语言模型。

基于MLP的N-gram模型的基本思想是：使用前面N-1个词的分布式表示（词嵌入）作为输入，通过多层神经网络预测下一个词的概率分布。

模型架构包括以下组件：

1. **词嵌入层**：将离散的词转换为连续的向量表示。
   - 输入：词的one-hot编码或词索引
   - 输出：词的嵌入向量（通常是50-300维）

2. **上下文表示层**：将前面N-1个词的嵌入向量组合成一个上下文表示。
   - 最简单的方法是将这些向量拼接起来
   - 也可以使用平均、加权和等方法

3. **隐藏层**：一个或多个全连接层，用于学习上下文表示与下一个词之间的映射关系。
   - 每个隐藏层包括线性变换和非线性激活函数（如GELU）

4. **输出层**：一个全连接层，输出词汇表大小的向量，表示下一个词的概率分布。
   - 通常使用softmax函数将输出转换为概率分布

形式化地，模型的前向传播过程如下：

1. 获取前面N-1个词的嵌入向量：$e_1, e_2, ..., e_{N-1}$
2. 拼接这些向量得到上下文表示：$c = [e_1; e_2; ...; e_{N-1}]$
3. 通过隐藏层：$h = \text{GELU}(W_h c + b_h)$
4. 通过输出层：$o = W_o h + b_o$
5. 应用softmax函数：$p = \text{softmax}(o)$

其中，$W_h$和$b_h$是隐藏层的权重和偏置，$W_o$和$b_o$是输出层的权重和偏置，$p$是下一个词的概率分布。

### 训练与评估

基于MLP的N-gram模型的训练过程包括以下步骤：

1. **数据准备**：
   - 将文本分割成词序列
   - 构建训练样本：每个样本包括N-1个输入词和1个目标词
   - 将词转换为索引或one-hot编码

2. **模型初始化**：
   - 随机初始化词嵌入矩阵和网络参数
   - 选择合适的超参数（如嵌入维度、隐藏层大小、学习率等）

3. **训练循环**：
   - 前向传播：计算模型预测的下一个词的概率分布
   - 计算损失：通常使用交叉熵损失或负对数似然损失
   - 反向传播：计算梯度
   - 参数更新：使用优化算法（如SGD、Adam）更新模型参数

4. **评估**：
   - 困惑度（Perplexity）：语言模型最常用的评估指标，定义为平均每个词的负对数似然的指数：
     $$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(w_i|w_{i-n+1}, ..., w_{i-1})\right)$$
     其中，$p(w_i|w_{i-n+1}, ..., w_{i-1})$是模型预测的第i个词的条件概率。
   - 准确率：预测正确的词的比例
   - 生成质量：使用模型生成文本，并评估其流畅度和连贯性

下面是一个使用PyTorch实现基于MLP的N-gram模型的示例代码：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MLPNgramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size):
        super(MLPNgramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(inputs.shape[0], -1)
        hidden = self.gelu(self.linear1(embeds))
        output = self.linear2(hidden)
        log_probs = nn.functional.log_softmax(output, dim=1)
        return log_probs

# 超参数
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
CONTEXT_SIZE = 3  # 使用前3个词预测下一个词
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 128

# 创建模型
model = MLPNgramModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, CONTEXT_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_inputs, batch_targets in data_loader:  # 假设data_loader已定义
        model.zero_grad()
        log_probs = model(batch_inputs)
        loss = loss_function(log_probs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss}')

# 评估
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            log_probs = model(batch_inputs)
            loss = loss_function(log_probs, batch_targets)
            total_loss += loss.item()
            
            _, predicted = log_probs.max(1)
            total += batch_targets.size(0)
            correct += predicted.eq(batch_targets).sum().item()
    
    perplexity = np.exp(total_loss / len(data_loader))
    accuracy = 100. * correct / total
    return perplexity, accuracy
```

### 与传统N-gram模型的比较

基于MLP的N-gram模型与传统的统计N-gram模型相比有以下优势：

1. **更好的泛化能力**：
   - 传统N-gram模型基于离散的词序列统计，对未见过的序列泛化能力有限
   - 基于MLP的模型使用词的分布式表示，能够捕捉词之间的语义相似性，对未见过的序列有更好的泛化能力

2. **缓解数据稀疏性问题**：
   - 传统N-gram模型面临严重的数据稀疏性问题，需要复杂的平滑技术
   - 基于MLP的模型通过学习词的分布式表示和非线性映射，能够更好地处理稀疏数据

3. **捕捉更复杂的模式**：
   - 传统N-gram模型只能捕捉词序列的表面统计特性
   - 基于MLP的模型能够学习更复杂的非线性模式，捕捉词之间的深层语义关系

4. **可扩展性**：
   - 传统N-gram模型的性能受限于N的大小，增大N会导致数据稀疏性问题加剧
   - 基于MLP的模型可以通过增加网络深度和宽度来提高模型容量，而不会显著增加数据稀疏性问题

5. **与现代深度学习技术的兼容性**：
   - 基于MLP的模型可以与现代深度学习技术（如Dropout、Batch Normalization等）结合，进一步提高性能
   - 基于MLP的模型可以作为更复杂神经网络架构的基础或组件

然而，基于MLP的N-gram模型也有一些局限性：

1. **计算开销**：
   - 基于MLP的模型通常需要更多的计算资源和训练时间
   - 传统N-gram模型在小规模应用中可能更加高效

2. **解释性**：
   - 传统N-gram模型基于简单的计数统计，结果更容易解释
   - 基于MLP的模型是"黑盒"模型，内部工作机制不易理解

3. **上下文长度限制**：
   - 基于MLP的N-gram模型仍然受限于固定的上下文窗口大小
   - 增大上下文窗口会导致输入维度增加，增加模型复杂度和过拟合风险

尽管基于MLP的N-gram模型相比传统模型有显著改进，但它仍然无法有效处理长距离依赖关系。为了解决这个问题，研究者提出了循环神经网络（RNN）、长短期记忆网络（LSTM）和注意力机制等更先进的技术，这些将在后续章节中详细介绍。

## 总结

在本章中，我们从传统的Bigram模型扩展到更一般的N-gram模型，探讨了高阶N-gram模型的优势和挑战。我们介绍了多层感知器（MLP）的基础知识，包括感知器模型、多层网络结构和前向传播算法。

我们详细讨论了矩阵乘法在深度学习中的应用，强调了向量化计算和批处理技术的重要性。我们比较了各种激活函数，特别关注了GELU激活函数的特点及其在现代语言模型中的应用。

最后，我们设计并实现了一个基于MLP的N-gram语言模型，讨论了其训练与评估方法，并与传统的统计N-gram模型进行了比较。

基于MLP的N-gram模型是从传统统计语言模型向现代神经网络语言模型过渡的重要一步。它引入了词的分布式表示和非线性映射，显著提高了模型的泛化能力和性能。然而，它仍然受限于固定的上下文窗口大小，无法有效处理长距离依赖关系。

在下一章中，我们将学习注意力机制，这是现代语言模型的核心组件之一，它能够动态地关注输入序列的不同部分，有效地处理长距离依赖关系。
