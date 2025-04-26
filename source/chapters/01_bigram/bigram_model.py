# -*- coding: utf-8 -*-
"""
Bigram语言模型实现

这个模块包含了Bigram语言模型的核心功能，包括：
1. 数据预处理
2. 模型训练
3. 文本生成
4. 模型评估

这些函数可以被导入到其他Python脚本或Jupyter notebook中使用。
"""

import random
import numpy as np
from collections import defaultdict, Counter

# 设置随机种子，确保结果可重现
random.seed(42)
np.random.seed(42)


def preprocess_text(text):
    """
    对英文文本进行预处理，包括分词、添加特殊标记和构建词汇表
    
    参数:
        text (str): 输入的文本
        
    返回:
        tuple: (sentences, vocab)
            sentences: 处理后的句子列表，每个句子是一个词列表
            vocab: 词汇表，包含所有不同的词
    """
    # 分词（这里简单地按空格分割）
    words = text.lower().split()
    
    # 添加特殊标记
    sentences = []
    current_sentence = ['<s>']
    
    for word in words:
        # 检查单词是否以句号、感叹号或问号结尾
        if word.endswith('.') or word.endswith('!') or word.endswith('?'):
            # 将单词的主体部分添加到当前句子
            if len(word) > 1:  # 如果单词不只是标点符号
                current_sentence.append(word[:-1])
            current_sentence.append('</s>')
            sentences.append(current_sentence)
            current_sentence = ['<s>']
        elif word in ['.', '!', '?']:  # 单独的标点符号
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


def preprocess_chinese_text(text):
    """
    对中文文本进行预处理，按字符分割，添加特殊标记和构建词汇表
    
    参数:
        text (str): 输入的中文文本
        
    返回:
        tuple: (sentences, vocab)
            sentences: 处理后的句子列表，每个句子是一个字符列表
            vocab: 词汇表，包含所有不同的字符
    """
    # 按字符分割
    chars = list(text)
    
    # 添加特殊标记
    sentences = []
    current_sentence = ['<s>']
    
    for char in chars:
        if char in ['。', '！', '？']:
            current_sentence.append('</s>')
            sentences.append(current_sentence)
            current_sentence = ['<s>']
        elif char not in ['\n', ' ', '\t']:
            current_sentence.append(char)
    
    # 处理最后一个句子
    if len(current_sentence) > 1:  # 不只包含<s>
        current_sentence.append('</s>')
        sentences.append(current_sentence)
    
    # 构建词汇表
    vocab = set()
    for sentence in sentences:
        for char in sentence:
            vocab.add(char)
    
    return sentences, vocab


def train_bigram_model(sentences, vocab, smoothing='laplace', k=1.0):
    """
    训练Bigram语言模型
    
    参数:
        sentences (list): 句子列表，每个句子是一个词列表
        vocab (set): 词汇表
        smoothing (str): 平滑方法，可选值：'none', 'laplace', 'add_k', 'interpolation'
        k (float): 加k平滑的k值，默认为1.0
        
    返回:
        tuple: (bigram_probs, unigram_counts, bigram_counts)
            bigram_probs: 二元语法条件概率
            unigram_counts: 一元语法计数
            bigram_counts: 二元语法计数
    """
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
    
    # 计算一元语法概率（用于插值平滑）
    total_words = sum(unigram_counts.values())
    unigram_probs = {word: count/total_words for word, count in unigram_counts.items()}
    
    # 计算条件概率
    bigram_probs = {}
    vocab_size = len(vocab)
    
    for w_prev in bigram_counts:
        bigram_probs[w_prev] = {}
        for w_curr in vocab:
            if smoothing == 'laplace':
                # 拉普拉斯平滑（加一平滑）
                count = bigram_counts[w_prev].get(w_curr, 0)
                bigram_probs[w_prev][w_curr] = (count + 1) / (unigram_counts[w_prev] + vocab_size)
            elif smoothing == 'add_k':
                # 加k平滑
                count = bigram_counts[w_prev].get(w_curr, 0)
                bigram_probs[w_prev][w_curr] = (count + k) / (unigram_counts[w_prev] + k * vocab_size)
            elif smoothing == 'interpolation':
                # 插值平滑（这里使用固定的λ=0.9）
                lambda_val = 0.9
                count = bigram_counts[w_prev].get(w_curr, 0)
                ml_prob = count / unigram_counts[w_prev] if count > 0 else 0
                uni_prob = unigram_probs.get(w_curr, 0)
                bigram_probs[w_prev][w_curr] = lambda_val * ml_prob + (1 - lambda_val) * uni_prob
            elif smoothing == 'none':
                # 无平滑（最大似然估计）
                if w_curr in bigram_counts[w_prev]:
                    bigram_probs[w_prev][w_curr] = bigram_counts[w_prev][w_curr] / unigram_counts[w_prev]
                else:
                    bigram_probs[w_prev][w_curr] = 0
    
    return bigram_probs, unigram_counts, bigram_counts


def generate_text(bigram_probs, max_length=20):
    """
    使用训练好的Bigram模型生成文本
    
    参数:
        bigram_probs (dict): 二元语法条件概率
        max_length (int): 生成文本的最大长度
        
    返回:
        str: 生成的文本
    """
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


def calculate_perplexity(bigram_probs, test_sentences):
    """
    计算Bigram模型在测试集上的困惑度
    
    参数:
        bigram_probs (dict): 二元语法条件概率
        test_sentences (list): 测试句子列表
        
    返回:
        float: 困惑度
    """
    log_likelihood = 0
    token_count = 0
    
    for sentence in test_sentences:
        for i in range(1, len(sentence)):
            w_prev = sentence[i-1]
            w_curr = sentence[i]
            
            # 获取条件概率 P(w_curr | w_prev)
            if w_prev in bigram_probs and w_curr in bigram_probs[w_prev]:
                prob = bigram_probs[w_prev][w_curr]
            else:
                # 如果词对不在模型中，使用一个很小的概率
                prob = 1e-10
            
            if prob > 0:
                log_likelihood += np.log2(prob)
            else:
                log_likelihood += np.log2(1e-10)  # 避免log(0)
            
            token_count += 1
    
    # 计算平均负对数似然
    avg_log_likelihood = -log_likelihood / token_count
    
    # 计算困惑度
    perplexity = 2 ** avg_log_likelihood
    
    return perplexity


def visualize_bigram_probabilities(bigram_probs, word, top_n=10):
    """
    可视化给定词后面最可能出现的top_n个词及其概率
    
    参数:
        bigram_probs (dict): 二元语法条件概率
        word (str): 给定的词
        top_n (int): 显示的词数量
        
    返回:
        None
    """
    import matplotlib.pyplot as plt
    
    if word not in bigram_probs:
        print(f"'{word}'not in the model")
        return
    
    # 获取给定词后面最可能出现的top_n个词
    next_words = [(next_word, prob) for next_word, prob in bigram_probs[word].items() if prob > 0]
    next_words.sort(key=lambda x: x[1], reverse=True)
    next_words = next_words[:top_n]
    
    if not next_words:
        print(f"'{word}'after no words have a probability greater than 0")
        return
    
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    words = [item[0] for item in next_words]
    probs = [item[1] for item in next_words]
    
    plt.bar(words, probs)
    plt.xlabel('next word')
    plt.ylabel('probability')
    plt.title(f"'{word}'after most possible {top_n} words")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{word}_bigram.png")


# 如果直接运行这个文件，展示一个简单的示例
if __name__ == '__main__':
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
    bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing='laplace')
    
    # 生成文本
    print("\n生成的文本:")
    for _ in range(3):
        generated_text = generate_text(bigram_probs)
        print(f"  {generated_text}")
    
    # 计算困惑度
    perplexity = calculate_perplexity(bigram_probs, sentences)
    print(f"\n困惑度: {perplexity:.4f}")