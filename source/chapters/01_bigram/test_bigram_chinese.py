# -*- coding: utf-8 -*-
"""
中文Bigram语言模型测试

这个测试文件专注于测试Bigram模型对中文文本的处理能力。
"""

import unittest
import random
import numpy as np

# 导入Bigram模型的函数
from bigram_model import (
    preprocess_chinese_text,
    train_bigram_model,
    generate_text,
    calculate_perplexity
)

# 设置随机种子，确保测试结果可重现
random.seed(42)
np.random.seed(42)


class TestChineseBigramModel(unittest.TestCase):
    """测试中文Bigram语言模型"""

    def setUp(self):
        """设置测试环境"""
        # 中文测试文本
        self.chinese_text = """
        自然语言处理是人工智能的重要分支。
        大语言模型可以生成流畅的文本。
        Bigram模型是最简单的语言模型之一。
        语言模型在机器翻译中有重要应用。
        """
    
    def test_chinese_preprocessing(self):
        """测试中文文本预处理"""
        sentences, vocab = preprocess_chinese_text(self.chinese_text)
        
        # 检查句子数量
        self.assertEqual(len(sentences), 4, "应该有4个句子")
        
        # 检查词汇表大小
        self.assertGreater(len(vocab), 20, "词汇表应该包含至少20个字符")
        
        # 检查特殊标记
        self.assertIn('<s>', vocab, "词汇表应该包含开始标记<s>")
        self.assertIn('</s>', vocab, "词汇表应该包含结束标记</s>")
        
        # 检查常见汉字
        common_chars = ['自', '然', '语', '言', '处', '理', '人', '工', '智', '能']
        for char in common_chars:
            self.assertIn(char, vocab, f"词汇表应该包含常见汉字'{char}'")
    
    def test_chinese_text_generation(self):
        """测试中文文本生成"""
        sentences, vocab = preprocess_chinese_text(self.chinese_text)
        bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing='laplace')
        
        # 生成多个文本样本
        generated_texts = []
        for _ in range(5):
            text = generate_text(bigram_probs, max_length=20)
            generated_texts.append(text)
            
            # 检查生成的文本不为空
            self.assertGreater(len(text), 0, "生成的文本不应该为空")
        
        # 检查生成的文本彼此不同（随机性）
        self.assertGreater(len(set(generated_texts)), 1, "生成的多个文本样本应该彼此不同")
    
    def test_chinese_perplexity(self):
        """测试中文模型的困惑度计算"""
        sentences, vocab = preprocess_chinese_text(self.chinese_text)
        
        # 使用不同的平滑方法
        smoothing_methods = ['laplace', 'add_k', 'interpolation']
        
        for method in smoothing_methods:
            if method == 'add_k':
                bigram_probs, _, _ = train_bigram_model(
                    sentences, vocab, smoothing=method, k=0.5
                )
            else:
                bigram_probs, _, _ = train_bigram_model(
                    sentences, vocab, smoothing=method
                )
            
            # 计算困惑度
            perplexity = calculate_perplexity(bigram_probs, sentences)
            
            # 检查困惑度是否为正数
            self.assertGreater(perplexity, 0, "困惑度应该是正数")
            
            # 检查困惑度是否在合理范围内
            self.assertLess(perplexity, 100, "困惑度应该在合理范围内")
            
            print(f"{method}平滑方法的中文困惑度: {perplexity:.4f}")


if __name__ == '__main__':
    unittest.main() 