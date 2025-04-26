#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bigram语言模型示例

这个脚本展示了如何使用Bigram语言模型进行文本生成和分析。
它是对第01章：Bigram语言模型（语言建模）的补充材料。

使用方法：
    python bigram_example.py
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from bigram_model import (
    preprocess_text,
    preprocess_chinese_text,
    train_bigram_model,
    generate_text,
    calculate_perplexity,
    visualize_bigram_probabilities
)

# 设置随机种子，确保结果可重现
random.seed(42)
np.random.seed(42)


def english_example():
    """英文Bigram模型示例"""
    print("\n===== 英文Bigram模型示例 =====")
    
    # 示例文本
    english_text = """
    The quick brown fox jumps over the lazy dog. 
    The dog barks at the fox. 
    The fox runs away quickly.
    A language model is a probability distribution over sequences of words.
    Bigram models consider only the previous word when predicting the next word.
    """
    
    # 预处理文本
    sentences, vocab = preprocess_text(english_text)
    print(f"词汇表大小: {len(vocab)}")
    print(f"句子数量: {len(sentences)}")
    
    # 训练Bigram模型（使用不同的平滑方法）
    smoothing_methods = ['none', 'laplace', 'add_k', 'interpolation']
    models = {}
    
    for method in smoothing_methods:
        if method == 'add_k':
            models[method] = train_bigram_model(sentences, vocab, smoothing=method, k=0.5)[0]
        else:
            models[method] = train_bigram_model(sentences, vocab, smoothing=method)[0]
        
        print(f"使用{method}平滑方法训练完成")
    
    # 生成文本
    print("\n使用不同平滑方法生成的文本：")
    for method, model in models.items():
        print(f"\n{method}平滑方法：")
        for _ in range(3):
            generated_text = generate_text(model)
            print(f"  生成的文本: {generated_text}")
    
    # 计算困惑度
    print("\n不同平滑方法的困惑度：")
    for method, model in models.items():
        perplexity = calculate_perplexity(model, sentences)
        print(f"{method}平滑方法的困惑度: {perplexity:.4f}")
    
    # 可视化某些词后面最可能出现的词
    print("\n可视化'the'后面最可能出现的词（请关闭图形窗口继续）：")
    visualize_bigram_probabilities(models['laplace'], 'the', top_n=5)


def chinese_example():
    """中文Bigram模型示例"""
    print("\n===== 中文Bigram模型示例 =====")
    
    # 中文示例文本
    chinese_text = """
    自然语言处理是人工智能的重要分支。
    大语言模型可以生成流畅的文本。
    Bigram模型是最简单的语言模型之一。
    """
    
    # 预处理中文文本
    chinese_sentences, chinese_vocab = preprocess_chinese_text(chinese_text)
    print(f"词汇表大小: {len(chinese_vocab)}")
    print(f"句子数量: {len(chinese_sentences)}")
    
    # 训练中文Bigram模型
    chinese_model, _, _ = train_bigram_model(chinese_sentences, chinese_vocab, smoothing='laplace')
    
    # 生成中文文本
    print("\n生成的中文文本：")
    for _ in range(5):
        generated_text = generate_text(chinese_model, max_length=30)
        print(f"  {generated_text}")
    
    # 计算困惑度
    perplexity = calculate_perplexity(chinese_model, chinese_sentences)
    print(f"\n困惑度: {perplexity:.4f}")
    
    # 可视化某些字后面最可能出现的字
    print("\n可视化'语'后面最可能出现的字（请关闭图形窗口继续）：")
    visualize_bigram_probabilities(chinese_model, '语', top_n=5)


def analyze_model_limitations():
    """分析Bigram模型的局限性"""
    print("\n===== Bigram模型局限性分析 =====")
    
    # 创建一个包含长距离依赖的文本
    long_dependency_text = """
    The man who lives next door works at the hospital. He is a doctor.
    The woman who lives upstairs works at the school. She is a teacher.
    """
    
    sentences, vocab = preprocess_text(long_dependency_text)
    bigram_model, _, _ = train_bigram_model(sentences, vocab, smoothing='laplace')
    
    print("\n原始文本：")
    print(long_dependency_text)
    
    print("\nBigram模型生成的文本：")
    for _ in range(5):
        generated_text = generate_text(bigram_model, max_length=30)
        print(f"  {generated_text}")
    
    print("\nBigram模型的局限性：")
    print("1. 只考虑前一个词的影响，无法捕捉长距离依赖关系")
    print("2. 生成的文本可能在局部上看起来合理，但整体上缺乏连贯性和一致性")
    print("3. 对于稀疏数据，即使使用平滑技术，模型的性能也会受到限制")
    print("4. 无法理解语义，只能基于统计规律生成文本")


if __name__ == "__main__":
    print("Bigram语言模型示例程序")
    print("=======================")
    
    english_example()
    chinese_example()
    analyze_model_limitations()
    
    print("\n示例程序结束")