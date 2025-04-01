---
file_format: mystnb
kernelspec:
  name: python3
---
# 第01章：Bigram语言模型（语言建模）

## 1. 语言模型基础概念

### 什么是语言模型

语言模型是自然语言处理（NLP）领域的核心技术，它的基本任务是预测文本序列中的下一个词或字符。从本质上讲，语言模型试图捕捉人类语言的统计规律，理解词与词之间的关联和依赖关系。一个优秀的语言模型能够生成流畅、连贯且符合语法规则的文本，就像是由人类撰写的一样。

语言模型可以形式化地定义为：给定一个词序列 $w_1, w_2, ..., w_{n-1}$，模型需要计算下一个词 $w_n$ 出现的概率分布 $P(w_n | w_1, w_2, ..., w_{n-1})$。这个条件概率表示了在已知前面所有词的情况下，下一个词可能是什么。

### 语言模型的历史发展

语言模型的发展历程可以大致分为以下几个阶段：

1. **统计语言模型时代（1980s-2000s）**：
   - **N-gram模型**：最早的语言模型主要基于N-gram统计，如Bigram（二元语法）和Trigram（三元语法）。这些模型假设一个词的出现只与其前面的N-1个词有关。
   - **平滑技术**：为了解决数据稀疏问题，研究者提出了各种平滑方法，如拉普拉斯平滑、Good-Turing平滑等。
   - **回退模型**：当高阶N-gram没有足够统计数据时，回退到低阶模型。

2. **神经网络语言模型时代（2003-2013）**：
   - **前馈神经网络语言模型**：Bengio等人在2003年提出了第一个基于神经网络的语言模型，使用词嵌入和前馈神经网络来预测下一个词。
   - **循环神经网络（RNN）**：Mikolov等人使用RNN构建语言模型，能够捕捉更长距离的依赖关系。
   - **长短期记忆网络（LSTM）和门控循环单元（GRU）**：这些改进的RNN架构解决了标准RNN的梯度消失问题，能够学习更长序列的依赖关系。

3. **Transformer时代（2017-至今）**：
   - **Transformer架构**：2017年，Vaswani等人提出的Transformer架构通过自注意力机制彻底改变了NLP领域。
   - **预训练语言模型**：GPT、BERT、T5等大型预训练模型的出现，将语言模型的能力提升到了新的高度。
   - **大型语言模型（LLM）**：如GPT-3、GPT-4、LLaMA、Claude等拥有数十亿到数千亿参数的模型，展现出了惊人的语言理解和生成能力。

### 语言模型的应用场景

语言模型在现代人工智能和自然语言处理中有着广泛的应用：

1. **文本生成**：
   - 故事创作和内容生成
   - 自动写作辅助
   - 对话系统和聊天机器人
   - 诗歌、歌词创作

2. **机器翻译**：
   - 语言模型可以帮助生成更流畅的翻译结果
   - 多语言翻译系统

3. **文本摘要**：
   - 自动生成长文档的摘要
   - 新闻摘要和报告生成

4. **问答系统**：
   - 基于知识的问答
   - 开放域问答

5. **语音识别**：
   - 提高语音转文字的准确性
   - 语音助手系统

6. **代码生成与补全**：
   - 编程辅助工具
   - 自动代码生成

7. **文本纠错与改写**：
   - 语法检查和纠正
   - 风格转换和文本改写

随着大型语言模型的发展，语言模型的应用场景还在不断扩展，正在改变人类与计算机交互的方式。

## 2. 概率论基础

要理解语言模型，特别是统计语言模型，我们需要掌握一些基本的概率论概念。

### 条件概率

条件概率是指在已知一个事件B发生的情况下，另一个事件A发生的概率，记作P(A|B)。

数学定义：
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

其中，P(A∩B)是事件A和B同时发生的概率，P(B)是事件B发生的概率。

在语言模型中，我们关心的是在已知前面的词的条件下，下一个词出现的概率。例如，P("学习"|"我喜欢")表示在"我喜欢"之后出现"学习"的概率。

### 联合概率

联合概率是指多个事件同时发生的概率，记作P(A,B)或P(A∩B)。

在语言模型中，一个句子的联合概率可以表示为：
$$P(w_1, w_2, ..., w_n) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_1,w_2) \times ... \times P(w_n|w_1,w_2,...,w_{n-1})$$

这个公式使用了概率的链式法则，将联合概率分解为条件概率的乘积。

### 马尔可夫假设

在实际应用中，计算完整的条件概率$P(w_n|w_1,w_2,...,w_{n-1})$非常困难，因为随着历史长度的增加，可能的组合数量呈指数级增长，导致数据稀疏问题。

为了简化计算，我们引入马尔可夫假设：假设当前状态只依赖于有限的前k个状态，而与更早的状态无关。

在语言模型中，k阶马尔可夫假设可以表示为：
$$P(w_n|w_1,w_2,...,w_{n-1}) \approx P(w_n|w_{n-k},...,w_{n-1})$$

特别地，当k=1时，我们得到一阶马尔可夫假设：
$$P(w_n|w_1,w_2,...,w_{n-1}) \approx P(w_n|w_{n-1})$$

这就是Bigram模型的基础，它假设一个词的出现只与其前一个词有关。

## 3. Bigram模型详解

### Bigram模型的数学定义

Bigram（二元语法）模型是最简单的N-gram模型之一，它基于一阶马尔可夫假设，认为一个词的出现只与其前一个词相关。

在Bigram模型中，一个句子的概率可以表示为：
$$P(w_1, w_2, ..., w_n) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_2) \times ... \times P(w_n|w_{n-1})$$

其中，$P(w_i|w_{i-1})$是条件概率，表示在词$w_{i-1}$之后出现词$w_i$的概率。

为了处理句子的开始，我们通常引入一个特殊的开始标记\<s>，并将$P(w_1)$表示为$P(w_1|\<s>)$。同样，为了处理句子的结束，我们引入结束标记\</s>。

### 从文本数据构建Bigram模型

构建Bigram模型的过程主要包括以下步骤：

1. **数据预处理**：
   - 分词：将文本分割成词或字符序列
   - 添加特殊标记：在每个句子的开始和结束添加\<s>和\</s>标记
   - 构建词汇表：统计所有不同的词，并为每个词分配一个唯一的索引

2. **统计频率**：
   - 统计每个词对$(w_{i-1}, w_i)$在语料库中出现的次数，记为$count(w_{i-1}, w_i)$
   - 统计每个词$w_{i-1}$在语料库中出现的次数，记为$count(w_{i-1})$

3. **计算条件概率**：
   - 使用最大似然估计（MLE）计算条件概率：
     $$P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i)}{count(w_{i-1})}$$

### Bigram模型的参数估计

最大似然估计是Bigram模型最基本的参数估计方法，但它存在一个严重问题：对于在训练数据中未出现的词对，其条件概率将为零，这会导致整个句子的概率为零，即使只有一个词对未在训练数据中出现。

为了解决这个问题，我们需要使用平滑技术。以下是几种常见的平滑方法：

1. **拉普拉斯平滑（加一平滑）**：
   $$P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i) + 1}{count(w_{i-1}) + V}$$
   其中，V是词汇表的大小。

2. **加k平滑**：
   $$P(w_i|w_{i-1}) = \frac{count(w_{i-1}, w_i) + k}{count(w_{i-1}) + k \times V}$$
   其中，k是一个小于1的正数。

3. **Good-Turing平滑**：
   调整频率计数，使得未见事件获得一些概率质量。

4. **插值平滑**：
   将高阶模型与低阶模型结合：
   $$P(w_i|w_{i-1}) = \lambda P_{ML}(w_i|w_{i-1}) + (1-\lambda)P(w_i)$$
   其中，λ是一个介于0和1之间的插值参数，$P_{ML}$是最大似然估计，$P(w_i)$是一元语法概率。

5. **回退平滑**：
   当高阶N-gram没有足够的统计数据时，回退到低阶模型。

## 4. 实现一个简单的Bigram模型

下面我们将实现一个简单的Bigram语言模型，包括数据预处理、模型训练和文本生成。

### 数据预处理

首先，我们需要对文本数据进行预处理，包括分词、添加特殊标记和构建词汇表。

```python
def preprocess_text(text):
    # 分词（这里简单地按空格分割）
    words = text.lower().split()
    
    # 添加特殊标记
    sentences = []
    current_sentence = ['<s>']
    
    for word in words:
        if word in ['.', '!', '?']:
            current_sentence.append('</s>')
            sentences.append(current_sentence)
            current_sentence = ['<s>']
        else:
            current_sentence.append(word)
    
    # 处理最后一个句子
    if len(current_sentence) > 1:  # 不只包含<s>
        current_sentence.append('</s>')
        sentences.append(current_sentence)
    
    # 构建词汇表
    vocab = set()
    for sentence in sentences:
        for word in sentence:
            vocab.add(word)
    
    return sentences, vocab
```

### 模型训练

接下来，我们统计词对频率并计算条件概率：

```python
def train_bigram_model(sentences, vocab, smoothing='laplace'):
    # 初始化计数器
    bigram_counts = {}
    unigram_counts = {}
    
    # 统计频率
    for sentence in sentences:
        for i in range(len(sentence) - 1):
            w_prev = sentence[i]
            w_curr = sentence[i + 1]
            
            # 更新二元语法计数
            if w_prev not in bigram_counts:
                bigram_counts[w_prev] = {}
            if w_curr not in bigram_counts[w_prev]:
                bigram_counts[w_prev][w_curr] = 0
            bigram_counts[w_prev][w_curr] += 1
            
            # 更新一元语法计数
            if w_prev not in unigram_counts:
                unigram_counts[w_prev] = 0
            unigram_counts[w_prev] += 1
    
    # 计算条件概率
    bigram_probs = {}
    vocab_size = len(vocab)
    
    for w_prev in bigram_counts:
        bigram_probs[w_prev] = {}
        for w_curr in vocab:
            if smoothing == 'laplace':
                # 拉普拉斯平滑
                count = bigram_counts[w_prev].get(w_curr, 0)
                bigram_probs[w_prev][w_curr] = (count + 1) / (unigram_counts[w_prev] + vocab_size)
            elif smoothing == 'none':
                # 无平滑（最大似然估计）
                if w_curr in bigram_counts[w_prev]:
                    bigram_probs[w_prev][w_curr] = bigram_counts[w_prev][w_curr] / unigram_counts[w_prev]
                else:
                    bigram_probs[w_prev][w_curr] = 0
    
    return bigram_probs
```

### 文本生成

有了训练好的Bigram模型，我们可以生成新的文本：

```python
import random

def generate_text(bigram_probs, max_length=20):
    # 从<s>开始
    current_word = '<s>'
    generated_text = []
    
    # 生成文本，直到遇到</s>或达到最大长度
    while current_word != '</s>' and len(generated_text) < max_length:
        # 获取当前词之后可能出现的所有词及其概率
        next_word_probs = bigram_probs.get(current_word, {})
        
        if not next_word_probs:
            break
        
        # 按概率随机选择下一个词
        words = list(next_word_probs.keys())
        probs = list(next_word_probs.values())
        
        # 归一化概率
        sum_probs = sum(probs)
        if sum_probs > 0:
            probs = [p / sum_probs for p in probs]
            next_word = random.choices(words, weights=probs, k=1)[0]
        else:
            # 如果所有概率都为0，随机选择
            next_word = random.choice(words)
        
        if next_word != '</s>':
            generated_text.append(next_word)
        
        current_word = next_word
    
    return ' '.join(generated_text)
```

### 完整示例

下面是一个完整的示例，展示如何使用上述代码训练Bigram模型并生成文本：

```python
# 示例文本
text = """
The quick brown fox jumps over the lazy dog. 
The dog barks at the fox. 
The fox runs away quickly.
"""

# 预处理文本
sentences, vocab = preprocess_text(text)
print(f"词汇表大小: {len(vocab)}")
print(f"句子数量: {len(sentences)}")

# 训练Bigram模型
bigram_probs = train_bigram_model(sentences, vocab, smoothing='laplace')

# 生成文本
for _ in range(5):
    generated_text = generate_text(bigram_probs)
    print(f"生成的文本: {generated_text}")
```

这个简单的Bigram模型可以生成基本的文本，但由于只考虑了前一个词的影响，生成的文本通常缺乏长距离的连贯性和语义一致性。

## 5. Bigram模型的局限性

尽管Bigram模型简单易实现，但它存在一些明显的局限性：

### 稀疏性问题

Bigram模型面临的主要挑战之一是数据稀疏性。即使在大型语料库中，也会有许多合法的词对从未出现过。这导致模型对这些未见词对的概率估计为零或接近零，影响模型的泛化能力。

例如，假设训练语料库中从未出现过"人工智能"后面跟着"革命"这个词对，但这是一个完全合理的组合。Bigram模型会给这个组合分配很低的概率，即使它在语义上是合理的。

### 上下文有限问题

Bigram模型只考虑前一个词的影响，忽略了更广泛的上下文。这导致生成的文本可能在局部上看起来合理，但整体上缺乏连贯性和一致性。

例如，在句子"我昨天去了_____"中，填空词的选择应该受到整个前文的影响，而不仅仅是"了"这个词。Bigram模型无法捕捉这种长距离依赖关系。

### 平滑技术简介

为了缓解数据稀疏性问题，我们介绍了几种平滑技术。这些技术通过从高频事件中"偷取"一些概率质量并重新分配给低频或未见事件，使模型能够更好地处理未见词对。

然而，平滑技术只能部分缓解问题，无法从根本上解决Bigram模型的局限性。为了构建更强大的语言模型，我们需要：

1. **考虑更长的上下文**：使用更高阶的N-gram模型（如Trigram、4-gram等）或循环神经网络（RNN）、Transformer等能够捕捉长距离依赖的模型。

2. **使用词嵌入**：将离散的词表示为连续的向量，使模型能够捕捉词之间的语义相似性。

3. **引入神经网络**：利用神经网络的强大表示能力，学习更复杂的语言模式。

在接下来的章节中，我们将逐步探索这些更先进的技术，最终构建一个强大的故事讲述AI大语言模型。

## 总结

在本章中，我们介绍了语言模型的基本概念、历史发展和应用场景，学习了概率论的基础知识，详细讲解了Bigram模型的原理和实现方法，并讨论了Bigram模型的局限性。

Bigram模型是理解更复杂语言模型的基础，它通过简单的统计方法捕捉词与词之间的局部关系。尽管它存在明显的局限性，但Bigram模型为我们提供了一个理解语言建模核心思想的起点。

在下一章中，我们将学习Micrograd，这是一个微型自动微分引擎，它将为我们构建神经网络语言模型奠定基础。通过Micrograd，我们将深入理解机器学习的核心概念和反向传播算法，为后续章节中更复杂模型的实现做好准备。
