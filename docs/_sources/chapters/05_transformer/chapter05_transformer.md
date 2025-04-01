# 第05章：Transformer（transformer架构，残差连接，层归一化，GPT-2）

## 1. Transformer架构概览

### 编码器-解码器结构

Transformer架构是由Vaswani等人在2017年的论文《Attention is All You Need》中提出的，它彻底改变了自然语言处理领域。与之前依赖循环或卷积结构的模型不同，Transformer完全基于注意力机制，摒弃了循环结构，实现了更高的并行性和更好的长距离依赖建模能力。

Transformer的原始架构采用了编码器-解码器（Encoder-Decoder）结构，这是一种在序列到序列学习任务（如机器翻译）中常用的框架。在这种结构中：

1. **编码器（Encoder）**：负责将输入序列（如源语言句子）转换为连续表示（通常是一系列向量）。
2. **解码器（Decoder）**：基于编码器的输出和之前生成的输出，逐步生成目标序列（如目标语言句子）。

Transformer的编码器和解码器都由多个相同层堆叠而成，每层包含两个主要子层：

1. **多头注意力机制（Multi-head Attention）**：允许模型关注输入序列的不同部分。
2. **前馈神经网络（Feed-Forward Network）**：对每个位置独立应用相同的全连接网络。

此外，每个子层都使用残差连接（Residual Connection）和层归一化（Layer Normalization）。

编码器和解码器的主要区别在于：

1. 编码器中的注意力层允许每个位置关注输入序列的所有位置（自注意力）。
2. 解码器包含两个注意力层：
   - 第一个是掩码自注意力层，只允许关注当前位置及其之前的位置，以防止信息泄露。
   - 第二个是编码器-解码器注意力层，允许解码器关注编码器的输出。

这种架构设计使得Transformer能够高效地处理序列到序列的任务，如机器翻译、文本摘要等。

### 多头注意力机制

多头注意力机制（Multi-head Attention）是Transformer的核心创新之一，它扩展了基本的注意力机制，允许模型同时关注不同表示子空间中的信息。

在基本的注意力机制中，我们使用查询（Q）、键（K）和值（V）计算加权和。多头注意力机制并行地执行多个这样的注意力计算，每个称为一个"头"（head）。具体来说，它首先将查询、键和值线性投影到不同的子空间，然后对每个投影执行注意力计算，最后将所有头的输出拼接并再次线性变换。

形式化地，多头注意力的计算过程如下：

1. 线性投影：将查询、键和值投影到h个不同的子空间
   $$Q_i = QW_i^Q, \quad K_i = KW_i^K, \quad V_i = VW_i^V$$
   其中，$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$，$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$是可学习的参数矩阵，$d_k = d_v = d_{model}/h$。

2. 对每个头计算注意力：
   $$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i$$

3. 拼接所有头的输出并线性变换：
   $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O$$
   其中，$W^O \in \mathbb{R}^{hd_v \times d_{model}}$是可学习的参数矩阵。

多头注意力机制的优势在于：

1. **增强表示能力**：不同的头可以关注不同类型的模式，如语法关系、语义关系等。
2. **并行计算**：所有头可以并行计算，提高效率。
3. **稳定训练**：多头机制可以减少方差，使训练更加稳定。

在实践中，头的数量（h）通常设置为8或16。每个头的维度（$d_k$和$d_v$）相应地减小，使得总计算复杂度与单头注意力相近。

### 前馈神经网络

在Transformer的每个编码器和解码器层中，多头注意力子层之后是前馈神经网络（Feed-Forward Network, FFN）子层。这个前馈网络对序列中的每个位置独立应用，因此也被称为位置前馈网络（Position-wise Feed-Forward Network）。

前馈网络由两个线性变换组成，中间有一个非线性激活函数（通常是ReLU或GELU）：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

或者使用GELU激活函数：

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

其中，$W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$b_1 \in \mathbb{R}^{d_{ff}}$，$b_2 \in \mathbb{R}^{d_{model}}$是可学习的参数。$d_{ff}$是内部维度，通常设置为$d_{model}$的4倍（例如，如果$d_{model} = 512$，则$d_{ff} = 2048$）。

前馈网络的作用是引入非线性变换，增强模型的表示能力。由于它对每个位置独立应用，因此可以看作是一种特征转换，将注意力机制捕获的上下文信息进一步处理。

值得注意的是，尽管前馈网络对每个位置独立应用，但由于之前的注意力层已经融合了上下文信息，因此模型仍然能够捕获序列中的依赖关系。

## 2. 残差连接

### 残差连接的动机

残差连接（Residual Connection）是由He等人在2015年的论文《Deep Residual Learning for Image Recognition》中提出的，最初用于解决深度卷积神经网络中的梯度消失问题。在Transformer中，残差连接被用于连接每个子层的输入和输出，有助于训练更深的网络。

残差连接的核心思想是，不直接学习一个函数$F(x)$，而是学习一个残差函数$F(x) - x$，即网络实际学习的是$H(x) = F(x) + x$，其中$x$是输入，$F(x)$是子层的输出。

残差连接的主要动机包括：

1. **缓解梯度消失问题**：在深度网络中，梯度在反向传播过程中可能会变得非常小，导致网络难以训练。残差连接提供了一条"捷径"，使梯度可以直接流回较早的层，缓解梯度消失问题。

2. **简化优化过程**：学习残差函数通常比学习原始函数更容易。如果最优函数接近于恒等映射，那么残差部分就会接近于零，这比直接学习恒等映射要容易得多。

3. **提高模型性能**：残差连接使得网络可以更容易地保留低层特征，同时学习高层特征，从而提高模型的表示能力和性能。

在Transformer中，残差连接被应用于每个子层（多头注意力和前馈网络）的前后，形式为：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

其中，$\text{Sublayer}(x)$是子层的函数（如多头注意力或前馈网络），$\text{LayerNorm}$是层归一化。

### 梯度流与信息流

残差连接对梯度流和信息流有重要影响，这是它能够有效缓解深度网络训练问题的关键原因。

**梯度流**：在反向传播过程中，梯度需要从输出层流回到输入层。在没有残差连接的深度网络中，梯度需要通过每一层的权重矩阵和激活函数的导数，这可能导致梯度消失或爆炸。残差连接提供了一条绕过这些层的路径，使梯度可以直接流回较早的层。

具体来说，对于残差块$y = x + F(x)$，反向传播时梯度为：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial F(x)}{\partial x}\right)$$

其中，$\mathcal{L}$是损失函数。可以看到，即使$\frac{\partial F(x)}{\partial x}$很小，梯度仍然可以通过恒等映射（即1）流回。

**信息流**：在前向传播过程中，残差连接使得较早层的信息可以直接传递到较后层，而不必通过所有中间层的变换。这有助于保留原始输入的信息，同时允许网络学习更复杂的特征。

在Transformer中，残差连接使得每一层都可以访问之前所有层的信息，这对于捕获不同层次的语言特征非常重要。例如，较低层可能捕获词法和句法特征，而较高层可能捕获语义和上下文特征。残差连接使得这些不同层次的特征可以有效地组合，提高模型的表示能力。

### 实现细节

在Transformer中，残差连接的实现相对简单，但有一些重要的细节需要注意：

1. **维度匹配**：残差连接要求输入和输出的维度相同，这样才能进行元素wise的加法。在Transformer中，每个子层（多头注意力和前馈网络）都保持输入和输出维度相同（$d_{model}$），因此可以直接应用残差连接。

2. **与层归一化的结合**：在原始Transformer中，残差连接后立即应用层归一化：

   ```python
   def sublayer_connection(self, x, sublayer):
       return self.layer_norm(x + sublayer(x))
   ```

   这种"先残差后归一化"（Post-LN）的方式在实践中可能导致训练不稳定，特别是对于深层模型。因此，一些后续工作（如GPT-2）采用了"先归一化后残差"（Pre-LN）的方式：

   ```python
   def sublayer_connection(self, x, sublayer):
       return x + sublayer(self.layer_norm(x))
   ```

   Pre-LN通常更容易训练，因为它确保了残差路径上没有归一化操作，使梯度流更加稳定。

3. **Dropout**：为了防止过拟合，通常在子层输出上应用Dropout，然后再进行残差连接：

   ```python
   def sublayer_connection(self, x, sublayer):
       return self.layer_norm(x + self.dropout(sublayer(x)))
   ```

   Dropout率通常设置为0.1或0.2。

4. **初始化**：残差连接对初始化也有影响。为了保持前向传播时信号的方差稳定，通常对残差块中的权重进行特殊初始化，如使用较小的初始值。

下面是一个完整的Transformer编码器层的PyTorch实现，展示了残差连接的使用：

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer_norm1 = nn.LayerNorm(d_model)
        self.sublayer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 先归一化后残差 (Pre-LN)
        attn_output = self.self_attn(self.sublayer_norm1(x), mask=mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.sublayer_norm2(x))
        x = x + self.dropout(ff_output)
        
        return x
```

在这个实现中，我们使用了Pre-LN方式，即先对输入进行层归一化，然后应用子层（多头注意力或前馈网络），最后进行残差连接。这种方式在实践中通常更加稳定，特别是对于深层Transformer模型。

## 3. 层归一化

### 批归一化vs层归一化

归一化技术是深度学习中用于加速训练和提高模型性能的重要工具。在Transformer中，层归一化（Layer Normalization, LN）是一个关键组件。为了理解层归一化的作用，我们首先比较它与批归一化（Batch Normalization, BN）的区别。

**批归一化（Batch Normalization）**：
- 在批次维度上归一化，即对每个特征在批次中的所有样本上计算均值和方差。
- 形式：$\text{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$，其中$\mu_B$和$\sigma_B^2$是批次中每个特征的均值和方差。
- 优点：有效减少内部协变量偏移，加速训练，允许使用更高的学习率。
- 缺点：依赖于批次大小，对小批次效果不佳；在推理时需要使用运行时统计信息；不适合RNN等序列模型。

**层归一化（Layer Normalization）**：
- 在特征维度上归一化，即对每个样本的所有特征计算均值和方差。
- 形式：$\text{LN}(x) = \gamma \cdot \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta$，其中$\mu_L$和$\sigma_L^2$是每个样本中所有特征的均值和方差。
- 优点：不依赖于批次大小，适用于序列模型；训练和推理行为一致。
- 缺点：在某些任务（如计算机视觉）上可能不如批归一化有效。

在Transformer中选择层归一化而非批归一化的主要原因是：

1. **序列长度变化**：自然语言处理任务中，不同样本的序列长度可能不同，这使得批归一化难以应用。
2. **批次大小限制**：Transformer模型通常较大，训练时批次大小受限，而批归一化在小批次上效果不佳。
3. **位置独立性**：层归一化对每个位置独立应用，这与Transformer的位置独立设计理念一致。

### 数学原理

层归一化的数学原理相对简单，它对每个样本的特征进行归一化，使其均值为0，方差为1，然后应用可学习的缩放和偏移参数。

给定一个输入向量$x \in \mathbb{R}^d$（在Transformer中，这通常是一个词嵌入或隐藏状态），层归一化的计算步骤如下：

1. 计算均值：$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
2. 计算方差：$\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$
3. 归一化：$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$，其中$\epsilon$是一个小常数，用于数值稳定性。
4. 缩放和偏移：$y = \gamma \cdot \hat{x} + \beta$，其中$\gamma$和$\beta$是可学习的参数，初始值通常为$\gamma = 1$和$\beta = 0$。

在Transformer中，层归一化应用于每个位置的隐藏状态，对序列中的每个位置独立计算均值和方差。对于一个形状为$[batch\_size, seq\_length, d_{model}]$的张量，层归一化在最后一个维度（$d_{model}$）上进行。

层归一化的作用包括：

1. **加速训练**：通过归一化输入，减少了内部协变量偏移，使得优化过程更加稳定，允许使用更高的学习率。
2. **减少梯度消失/爆炸**：归一化后的值通常在一个合理的范围内，有助于防止梯度消失或爆炸。
3. **增强模型鲁棒性**：归一化使得模型对输入尺度的变化不那么敏感，增强了模型的鲁棒性。

### 实现与优化

层归一化在PyTorch等深度学习框架中已有内置实现，使用起来非常简单。以下是一个基本的PyTorch层归一化实现：

```python
import torch
import torch.nn as nn

# 使用内置的LayerNorm
layer_norm = nn.LayerNorm(d_model)
normalized_x = layer_norm(x)

# 或者手动实现
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

在实践中，层归一化的实现和使用有一些优化技巧：

1. **计算效率**：直接使用均值和标准差函数通常比手动计算更高效，因为它们有优化的实现。

2. **数值稳定性**：使用足够大的$\epsilon$值（如1e-5或1e-6）来确保数值稳定性，特别是在使用低精度（如fp16）训练时。

3. **初始化**：$\gamma$和$\beta$的初始化对模型性能有影响。通常，$\gamma$初始化为1，$\beta$初始化为0，但在某些情况下，使用较小的$\gamma$初始值（如0.1）可能有助于训练稳定性。

4. **与残差连接的结合**：如前所述，层归一化与残差连接的结合方式有两种：Post-LN（先残差后归一化）和Pre-LN（先归一化后残差）。在实践中，Pre-LN通常更容易训练，特别是对于深层模型。

5. **RMSNorm变体**：RMSNorm是层归一化的一个简化变体，它只归一化方差而不归一化均值。在某些任务上，RMSNorm可能比标准层归一化更有效，同时计算成本更低。

```python
class RMSNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.eps = eps
        
    def forward(self, x):
        # 只计算RMS（均方根），不减去均值
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

6. **性能优化**：在高性能实现中，可以使用融合操作（fused operations）来减少内存访问和提高计算效率。例如，NVIDIA的Apex库提供了融合的层归一化实现。

层归一化是Transformer架构的关键组件，它与残差连接一起，使得深层Transformer模型能够有效训练。在实践中，正确实现和配置层归一化对模型性能至关重要。

## 4. GPT-2模型详解

### GPT-2架构特点

GPT-2（Generative Pre-trained Transformer 2）是由OpenAI在2019年发布的一个大型语言模型，它基于Transformer架构，但与原始Transformer相比有一些重要的修改和改进。GPT-2的成功奠定了后续GPT系列模型的基础，并对大型语言模型的发展产生了深远影响。

GPT-2的主要架构特点包括：

1. **仅使用解码器**：与原始Transformer的编码器-解码器结构不同，GPT-2只使用解码器部分，这使得它更适合于生成任务。这种仅解码器的架构也被称为"自回归Transformer"或"单向Transformer"。

2. **掩码自注意力**：GPT-2使用掩码自注意力机制，确保每个位置只能关注其自身及之前的位置，这与语言模型的自回归性质一致。

3. **层归一化位置**：GPT-2采用了"先归一化后残差"（Pre-LN）的方式，即在每个子层之前应用层归一化，然后再应用残差连接。这与原始Transformer的"先残差后归一化"（Post-LN）不同，有助于训练更深的网络。

4. **更大的模型规模**：GPT-2有多个版本，最大的版本（GPT-2 XL）有15亿参数，这在当时是非常大的模型。

5. **字节对编码（BPE）分词**：GPT-2使用改进的BPE分词算法，词汇表大小为50,257。

6. **位置编码**：GPT-2使用学习的位置嵌入，而不是原始Transformer中的正弦余弦位置编码。

7. **激活函数**：GPT-2使用GELU激活函数，而不是ReLU。

GPT-2的基本结构可以表示为以下几个组件的堆叠：

```
Input Embeddings + Positional Embeddings
↓
for each layer:
    Layer Normalization
    Masked Multi-head Self-attention
    Residual Connection
    Layer Normalization
    Feed-forward Network
    Residual Connection
↓
Layer Normalization
Linear Layer + Softmax (for next token prediction)
```

GPT-2有四种不同大小的版本：

1. **GPT-2 Small**：1.17亿参数，12层，768维隐藏状态，12个注意力头
2. **GPT-2 Medium**：3.45亿参数，24层，1024维隐藏状态，16个注意力头
3. **GPT-2 Large**：7.74亿参数，36层，1280维隐藏状态，20个注意力头
4. **GPT-2 XL**：15亿参数，48层，1600维隐藏状态，25个注意力头

这种通过增加层数和模型维度来扩展模型的方法，为后续更大规模语言模型的发展奠定了基础。

### 预训练与生成策略

GPT-2的训练过程分为两个阶段：预训练和微调（虽然OpenAI最初只发布了预训练模型）。

**预训练**：
GPT-2在一个名为WebText的大型文本语料库上进行预训练，该语料库包含约40GB的文本数据，来自互联网上的各种来源。预训练的目标是标准的语言模型目标，即最大化给定前面所有词的条件下，下一个词的概率：

$$\max_{\theta} \sum_{i} \log P(w_i | w_1, w_2, ..., w_{i-1}; \theta)$$

其中，$\theta$是模型参数，$w_i$是第i个词。

预训练过程使用了以下技术：

1. **大批量训练**：使用分布式训练和梯度累积来实现大批量。
2. **学习率调度**：使用余弦学习率调度，逐渐减小学习率。
3. **权重衰减**：应用权重衰减正则化，防止过拟合。
4. **梯度裁剪**：限制梯度范数，防止梯度爆炸。

**生成策略**：
GPT-2生成文本的基本方法是自回归生成，即一次生成一个词，然后将生成的词添加到输入中，继续生成下一个词。然而，简单的贪心解码（每次选择概率最高的词）通常会导致重复和缺乏多样性的输出。为了生成更高质量的文本，GPT-2使用了几种解码策略：

1. **温度采样**：通过调整softmax的温度参数来控制生成的随机性：

   $$P(w_i | w_{<i}) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

   其中，$z_i$是词$w_i$的logit，$T$是温度参数。较低的温度（如0.7）使生成更加确定性，较高的温度（如1.0或更高）增加随机性。

2. **Top-k采样**：在每一步，只从概率最高的k个词中采样，而不是整个词汇表：

   $$P(w_i | w_{<i}) \propto \begin{cases} 
   \exp(z_i), & \text{if } w_i \in \text{top-k}(z) \\
   0, & \text{otherwise}
   \end{cases}$$

   这有助于避免低概率词的干扰，同时保持一定的多样性。

3. **Top-p（核）采样**：选择概率总和达到阈值p的最小词集合，从中采样：

   $$P(w_i | w_{<i}) \propto \begin{cases} 
   \exp(z_i), & \text{if } w_i \in V_p \\
   0, & \text{otherwise}
   \end{cases}$$

   其中，$V_p$是使得$\sum_{w_j \in V_p} P(w_j | w_{<i}) \geq p$的最小词集合。这种方法比Top-k更加动态，在不同上下文中可以选择不同数量的候选词。

4. **重复惩罚**：降低已生成词的概率，防止重复：

   $$P(w_i | w_{<i}) \propto \exp(z_i - \alpha \cdot \mathbb{1}(w_i \in w_{<i}))$$

   其中，$\alpha$是惩罚系数，$\mathbb{1}(w_i \in w_{<i})$是指示函数，如果$w_i$已经在生成的序列中出现，则为1，否则为0。

这些生成策略的组合使得GPT-2能够生成连贯、多样且高质量的文本。在实践中，通常需要根据具体任务调整这些策略的参数，以获得最佳效果。

### 解码器架构的优势

GPT-2采用的仅解码器架构（只使用Transformer的解码器部分）相比完整的编码器-解码器架构有几个重要优势，特别是对于生成任务：

1. **参数效率**：对于相同参数量，仅解码器架构可以使用更多层或更大的隐藏维度，因为它不需要分配参数给编码器部分。这使得模型能够学习更复杂的模式和更长的依赖关系。

2. **训练效率**：仅解码器架构的训练更加高效，因为它只需要一次前向传播，而不是编码器和解码器的两次前向传播。这使得模型能够在相同的计算资源下处理更多的训练数据。

3. **自回归生成的自然适应**：解码器的掩码自注意力机制天然适合语言模型的自回归性质，即根据前面的词预测下一个词。这使得模型在生成任务上表现出色。

4. **统一的预训练和微调目标**：仅解码器架构使用相同的目标函数（下一个词预测）进行预训练和微调，这简化了训练过程，并使得模型能够更好地适应各种下游任务。

5. **灵活的上下文学习**：通过自注意力机制，解码器能够灵活地关注输入序列的不同部分，这使得模型能够从少量示例中学习任务（少样本学习）或根据提示生成相关内容（提示工程）。

6. **可扩展性**：仅解码器架构更容易扩展到更大的模型规模，因为它的结构更加统一和规则。这为后续的GPT-3、GPT-4等更大模型奠定了基础。

然而，仅解码器架构也有一些局限性：

1. **单向上下文**：由于掩码自注意力的限制，模型只能关注前面的词，而不能关注后面的词。这在某些需要双向上下文的任务（如填空题）上可能表现不佳。

2. **生成偏差**：自回归生成可能导致一些偏差，如倾向于生成更短的序列或重复内容。

3. **推理效率**：自回归生成需要逐词生成，无法并行化，这使得推理速度相对较慢，特别是对于长文本生成。

尽管有这些局限性，GPT-2及其后续模型的成功表明，仅解码器架构对于大规模语言模型是一个非常有效的选择。它的简单性、可扩展性和在生成任务上的强大性能，使其成为现代大型语言模型的主流架构之一。

## 5. 实现简化版Transformer

### 构建Transformer块

下面我们将实现一个简化版的Transformer模型，重点关注其核心组件。我们将从构建基本的Transformer块开始，包括多头注意力机制、前馈网络、残差连接和层归一化。

首先，让我们实现多头注意力机制：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性投影并分割为多个头
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 重新整形并连接多个头的输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性投影
        output = self.output(attn_output)
        
        return output, attn_weights
```

接下来，实现前馈网络：

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))
```

现在，我们可以构建完整的Transformer编码器层：

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 先归一化后残差 (Pre-LN)
        attn_input = self.norm1(x)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, mask)
        x = x + self.dropout(attn_output)
        
        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)
        
        return x
```

类似地，我们可以实现Transformer解码器层：

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # 掩码自注意力
        attn_input = self.norm1(x)
        self_attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, tgt_mask)
        x = x + self.dropout(self_attn_output)
        
        # 交叉注意力
        attn_input = self.norm2(x)
        cross_attn_output, _ = self.cross_attn(attn_input, memory, memory, memory_mask)
        x = x + self.dropout(cross_attn_output)
        
        # 前馈网络
        ff_input = self.norm3(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)
        
        return x
```

### 实现完整模型

现在，我们可以使用上面定义的组件来实现完整的Transformer模型。我们将实现两个版本：一个是原始的编码器-解码器Transformer，另一个是类似GPT-2的仅解码器Transformer。

首先，让我们实现位置编码：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不是模型参数）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

接下来，实现完整的编码器-解码器Transformer：

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 编码器和解码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终输出层
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 源序列嵌入和位置编码
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        src = self.dropout(src)
        
        # 目标序列嵌入和位置编码
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)
        
        # 编码器前向传播
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, src_mask)
        
        # 解码器前向传播
        output = tgt
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, memory, tgt_mask, memory_mask)
        
        # 最终输出
        output = self.final_norm(output)
        output = self.output_projection(output)
        
        return output
```

现在，让我们实现类似GPT-2的仅解码器Transformer：

```python
class GPT2Model(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super(GPT2Model, self).__init__()
        
        # 词嵌入和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # 解码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化和输出投影
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.size(1)
        assert seq_length <= self.max_seq_length, f"Input sequence length ({seq_length}) exceeds maximum sequence length ({self.max_seq_length})"
        
        # 创建位置索引
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # 词嵌入和位置嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # 创建注意力掩码（确保只关注当前位置及之前的位置）
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device))
        
        # 前向传播
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # 最终输出
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits
```

### 训练与文本生成

最后，让我们实现训练和文本生成的函数。首先是训练函数：

```python
def train_transformer(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 获取输入和目标
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        # 创建掩码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_len]
        tgt_mask = tgt_mask & torch.tril(torch.ones((tgt.size(1), tgt.size(1)), device=device)).bool()
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        # 计算损失
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

对于GPT-2模型的训练：

```python
def train_gpt2(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 获取输入
        input_ids = batch.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # 计算损失（预测下一个词）
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

最后，实现文本生成函数：

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=0, top_p=0.9, device='cuda'):
    model.eval()
    
    # 对提示进行分词
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型预测
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # 应用top-k采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 应用top-p（核）采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的词
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = -float('Inf')
            
            # 采样下一个词
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到输入序列
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 如果生成了结束标记，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return generated_text
```

这个简化版的Transformer实现包含了原始Transformer和GPT-2的核心组件，包括多头注意力、前馈网络、残差连接和层归一化。虽然它省略了一些细节（如学习率调度、更复杂的初始化策略等），但已经足够用于理解Transformer的基本原理和实现方法。

在实际应用中，通常会使用成熟的库（如Hugging Face的Transformers）来实现和使用Transformer模型，这些库提供了更多优化和功能。但理解底层实现对于深入理解模型工作原理和进行自定义修改非常重要。

## 总结

在本章中，我们深入探讨了Transformer架构，这是现代大型语言模型的基础。我们首先介绍了Transformer的整体架构，包括编码器-解码器结构、多头注意力机制和前馈神经网络。然后，我们详细讲解了残差连接和层归一化，这两个组件对于训练深层Transformer模型至关重要。

我们还分析了GPT-2模型，它是基于Transformer解码器的一个重要变体，为后续GPT系列模型奠定了基础。我们讨论了GPT-2的架构特点、预训练与生成策略，以及仅解码器架构的优势。

最后，我们实现了一个简化版的Transformer模型，包括原始的编码器-解码器Transformer和类似GPT-2的仅解码器Transformer，并提供了训练和文本生成的函数。

Transformer架构的引入彻底改变了自然语言处理领域，它摒弃了循环结构，完全基于注意力机制，实现了更高的并行性和更好的长距离依赖建模能力。随着模型规模的不断扩大和训练数据的增加，基于Transformer的模型（如GPT系列、BERT、T5等）展现出了惊人的语言理解和生成能力，推动了大型语言模型的快速发展。

在下一章中，我们将学习分词技术，特别是字节对编码（BPE），这是现代语言模型处理文本输入的关键组件。
