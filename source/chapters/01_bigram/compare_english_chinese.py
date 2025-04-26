# -*- coding: utf-8 -*-
"""
比较英文和中文的Bigram语言模型

这个脚本比较Bigram模型在英文和中文文本上的表现差异。
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from bigram_model import (
    preprocess_text,
    preprocess_chinese_text,
    train_bigram_model,
    generate_text,
    calculate_perplexity
)

# 设置随机种子，确保结果可重现
random.seed(42)
np.random.seed(42)


def compare_models():
    """比较英文和中文Bigram模型的表现"""
    # 英文测试文本
    english_text = """
    The quick brown fox jumps over the lazy dog. 
    The dog barks at the fox. 
    The fox runs away quickly.
    Natural language processing is a branch of artificial intelligence.
    Language models can generate fluent text.
    """
    
    # 中文测试文本
    chinese_text = """
    自然语言处理是人工智能的重要分支。
    大语言模型可以生成流畅的文本。
    Bigram模型是最简单的语言模型之一。
    语言模型在机器翻译中有重要应用。
    """
    
    print("=== 比较英文和中文Bigram模型 ===\n")
    
    # 预处理文本
    english_sentences, english_vocab = preprocess_text(english_text)
    chinese_sentences, chinese_vocab = preprocess_chinese_text(chinese_text)
    
    print(f"英文词汇表大小: {len(english_vocab)}")
    print(f"英文句子数量: {len(english_sentences)}")
    print(f"中文词汇表大小: {len(chinese_vocab)}")
    print(f"中文句子数量: {len(chinese_sentences)}")
    
    # 训练模型
    english_model, _, _ = train_bigram_model(english_sentences, english_vocab, smoothing='laplace')
    chinese_model, _, _ = train_bigram_model(chinese_sentences, chinese_vocab, smoothing='laplace')
    
    # 生成文本比较
    print("\n=== 生成文本比较 ===")
    
    print("\n英文生成的文本:")
    for i in range(3):
        english_text = generate_text(english_model, max_length=15)
        print(f"  {i+1}. {english_text}")
    
    print("\n中文生成的文本:")
    for i in range(3):
        chinese_text = generate_text(chinese_model, max_length=15)
        print(f"  {i+1}. {chinese_text}")
    
    # 计算困惑度
    english_perplexity = calculate_perplexity(english_model, english_sentences)
    chinese_perplexity = calculate_perplexity(chinese_model, chinese_sentences)
    
    print("\n=== 困惑度比较 ===")
    print(f"英文模型困惑度: {english_perplexity:.4f}")
    print(f"中文模型困惑度: {chinese_perplexity:.4f}")
    
    # 可视化比较
    plt.figure(figsize=(10, 6))
    plt.bar(['英文', '中文'], [english_perplexity, chinese_perplexity])
    plt.ylabel('困惑度')
    plt.title('英文和中文Bigram模型困惑度比较')
    plt.savefig('english_chinese_perplexity.png')
    plt.close()
    
    print("\n困惑度比较图已保存为'english_chinese_perplexity.png'")
    
    # 分析结果
    print("\n=== 分析 ===")
    print("1. 中文和英文的Bigram模型表现有所不同，这主要是因为两种语言的结构差异:")
    print("   - 英文是以词为基本单位，词之间有空格分隔")
    print("   - 中文是以字符为基本单位，没有明显的词边界")
    print("2. 困惑度的差异也反映了两种语言的不同统计特性")
    print("3. 在实际应用中，中文可能需要先进行分词，再应用Bigram模型，以获得更好的效果")


if __name__ == "__main__":
    compare_models() 