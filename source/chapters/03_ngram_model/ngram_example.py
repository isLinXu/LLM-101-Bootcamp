import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter, defaultdict
import jieba
import random

# 示例文本数据
sample_text = """
人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
人工智能也可以被定义为使机器能够执行需要人类智能才能完成的任务的计算系统的理论与开发。
目前人工智能研究的主要领域包括自然语言处理、机器学习、计算机视觉和专家系统等。
"""

# 1. 传统N-gram模型实现
def preprocess_text(text, n=3):
    """对文本进行预处理，分词并生成n-gram序列"""
    tokens = list(jieba.cut(text))
    ngrams = []
    
    # 添加开始和结束标记
    tokens = ["<s>"] * (n-1) + tokens + ["</s>"]
    
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    
    return tokens, ngrams

def train_ngram_model(text, n=3):
    """训练传统的n-gram语言模型"""
    tokens, ngrams = preprocess_text(text, n)
    
    # 计算n-gram和(n-1)-gram的频率
    ngram_counts = Counter(ngrams)
    context_counts = Counter([gram[:-1] for gram in ngrams])
    
    # 计算条件概率
    model = defaultdict(lambda: defaultdict(float))
    for gram in ngram_counts:
        context, word = gram[:-1], gram[-1]
        model[context][word] = ngram_counts[gram] / context_counts[context]
    
    return model, tokens

def generate_text(model, context, max_length=20):
    """使用n-gram模型生成文本"""
    result = list(context)
    n = len(context) + 1
    
    for _ in range(max_length):
        # 获取当前上下文
        current_context = tuple(result[-(n-1):])
        
        # 如果上下文不在模型中，随机选择一个词
        if current_context not in model:
            break
        
        # 根据条件概率采样下一个词
        next_word_probs = model[current_context]
        words = list(next_word_probs.keys())
        probs = list(next_word_probs.values())
        next_word = random.choices(words, weights=probs)[0]
        
        # 如果生成结束标记，停止生成
        if next_word == "</s>":
            break
        
        result.append(next_word)
    
    return "".join(result)

# 测试传统N-gram模型
def test_traditional_ngram():
    print("=== 传统N-gram模型测试 ===")
    for n in range(2, 5):
        print(f"\n--- {n}-gram模型 ---")
        model, tokens = train_ngram_model(sample_text, n)
        
        # 随机选择一个上下文
        start_idx = random.randint(0, len(tokens) - n)
        context = tuple(tokens[start_idx:start_idx+n-1])
        
        print(f"上下文: {''.join(context)}")
        generated_text = generate_text(model, context)
        print(f"生成文本: {generated_text}")
        
        # 计算困惑度 (只是示例，实际应在测试集上计算)
        test_tokens, test_ngrams = preprocess_text(sample_text[:len(sample_text)//2], n)
        log_prob_sum = 0
        count = 0
        
        for gram in test_ngrams:
            context, word = gram[:-1], gram[-1]
            if context in model and word in model[context]:
                log_prob_sum += np.log(model[context][word])
                count += 1
        
        if count > 0:
            perplexity = np.exp(-log_prob_sum / count)
            print(f"困惑度: {perplexity:.4f}")

# 2. 基于MLP的N-gram模型
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

def prepare_data_for_mlp_ngram(text, n, word_to_ix=None):
    """准备用于MLP N-gram模型的数据"""
    tokens = list(jieba.cut(text))
    tokens = ["<s>"] * (n-1) + tokens + ["</s>"]
    
    # 创建词汇表
    if word_to_ix is None:
        word_to_ix = {word: i for i, word in enumerate(set(tokens))}
    
    # 生成训练样本
    X, y = [], []
    for i in range(len(tokens) - n + 1):
        X.append([word_to_ix[token] for token in tokens[i:i+n-1]])
        y.append(word_to_ix[tokens[i+n-1]])
    
    return torch.tensor(X), torch.tensor(y), word_to_ix

def test_mlp_ngram():
    print("\n=== 基于MLP的N-gram模型测试 ===")
    n = 3  # 使用trigram模型
    
    # 准备数据
    X, y, word_to_ix = prepare_data_for_mlp_ngram(sample_text, n)
    ix_to_word = {i: word for word, i in word_to_ix.items()}
    vocab_size = len(word_to_ix)
    
    # 超参数
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 64
    CONTEXT_SIZE = n - 1
    LEARNING_RATE = 0.01
    EPOCHS = 100
    
    # 创建模型
    model = MLPNgramModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, CONTEXT_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    print("开始训练...")
    losses = []
    for epoch in range(EPOCHS):
        model.zero_grad()
        log_probs = model(X)
        loss = loss_function(log_probs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('mlp_ngram_loss.png')
    plt.close()
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        log_probs = model(X)
        _, predicted = log_probs.max(1)
        total = y.size(0)
        correct = predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        
        # 计算困惑度
        test_loss = loss_function(log_probs, y).item()
        perplexity = np.exp(test_loss)
        
        print(f"准确率: {accuracy:.2f}%")
        print(f"困惑度: {perplexity:.4f}")
    
    # 生成一些文本
    print("\n生成文本示例:")
    for _ in range(3):
        # 随机选择一个上下文
        start_idx = random.randint(0, len(X) - 1)
        context = X[start_idx].tolist()
        
        # 生成文本
        result = [ix_to_word[idx] for idx in context]
        for _ in range(15):
            with torch.no_grad():
                input_tensor = torch.tensor([context])
                log_probs = model(input_tensor)
                _, next_word_idx = log_probs.max(1)
                next_word = ix_to_word[next_word_idx.item()]
                
                if next_word == "</s>":
                    break
                    
                result.append(next_word)
                context = context[1:] + [next_word_idx.item()]
        
        print("".join(result))

# 3. 激活函数比较
def plot_activation_functions():
    print("\n=== 激活函数比较 ===")
    x = np.linspace(-5, 5, 1000)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Tanh
    tanh = np.tanh(x)
    
    # ReLU
    relu = np.maximum(0, x)
    
    # Leaky ReLU
    leaky_relu = np.where(x > 0, x, 0.01 * x)
    
    # ELU
    elu = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
    
    # GELU (近似)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    # 绘图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(x, sigmoid)
    plt.grid(True)
    plt.title('Sigmoid')
    
    plt.subplot(2, 3, 2)
    plt.plot(x, tanh)
    plt.grid(True)
    plt.title('Tanh')
    
    plt.subplot(2, 3, 3)
    plt.plot(x, relu)
    plt.grid(True)
    plt.title('ReLU')
    
    plt.subplot(2, 3, 4)
    plt.plot(x, leaky_relu)
    plt.grid(True)
    plt.title('Leaky ReLU')
    
    plt.subplot(2, 3, 5)
    plt.plot(x, elu)
    plt.grid(True)
    plt.title('ELU')
    
    plt.subplot(2, 3, 6)
    plt.plot(x, gelu)
    plt.grid(True)
    plt.title('GELU')
    
    plt.tight_layout()
    plt.savefig('activation_functions.png')
    plt.close()
    print("已生成激活函数对比图: activation_functions.png")

if __name__ == "__main__":
    test_traditional_ngram()
    plot_activation_functions()
    test_mlp_ngram() 