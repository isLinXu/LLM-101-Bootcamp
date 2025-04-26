# -*- coding: utf-8 -*-
"""
测试Bigram语言模型的功能

这个文件包含了一系列测试函数，用于验证Bigram语言模型的各个组件是否正常工作，
包括数据预处理、模型训练和文本生成功能。
"""

import unittest
import random
import numpy as np

# 导入notebook中定义的函数
# 注意：在实际使用时，你可能需要将这些函数放在一个单独的.py文件中
from bigram_model import (
    preprocess_text,
    preprocess_chinese_text,
    train_bigram_model,
    generate_text,
    calculate_perplexity
)

# 设置随机种子，确保测试结果可重现
random.seed(42)
np.random.seed(42)


class TestBigramModel(unittest.TestCase):
    """测试Bigram语言模型的各个组件"""

    def setUp(self):
        """设置测试环境"""
        # 英文测试文本
        self.english_text = """
        The quick brown fox jumps over the lazy dog. 
        The dog barks at the fox. 
        The fox runs away quickly.
        """
        
        # 中文测试文本
        self.chinese_text = """
        自然语言处理是人工智能的重要分支。
        大语言模型可以生成流畅的文本。
        Bigram模型是最简单的语言模型之一。
        """

    def test_preprocess_text(self):
        """测试英文文本预处理功能"""
        sentences, vocab = preprocess_text(self.english_text)
        
        # 检查句子数量
        self.assertEqual(len(sentences), 3, "应该有3个句子")
        
        # 检查词汇表大小
        self.assertGreater(len(vocab), 10, "词汇表应该包含至少10个词")
        
        # 检查特殊标记
        self.assertIn('<s>', vocab, "词汇表应该包含开始标记<s>")
        self.assertIn('</s>', vocab, "词汇表应该包含结束标记</s>")
        
        # 检查句子格式
        for sentence in sentences:
            self.assertEqual(sentence[0], '<s>', "每个句子应该以<s>开始")
            self.assertEqual(sentence[-1], '</s>', "每个句子应该以</s>结束")

    def test_preprocess_chinese_text(self):
        """测试中文文本预处理功能"""
        sentences, vocab = preprocess_chinese_text(self.chinese_text)
        
        # 检查句子数量
        self.assertEqual(len(sentences), 3, "应该有3个句子")
        
        # 检查词汇表大小
        self.assertGreater(len(vocab), 20, "词汇表应该包含至少20个字符")
        
        # 检查特殊标记
        self.assertIn('<s>', vocab, "词汇表应该包含开始标记<s>")
        self.assertIn('</s>', vocab, "词汇表应该包含结束标记</s>")

    def test_train_bigram_model(self):
        """测试模型训练功能"""
        sentences, vocab = preprocess_text(self.english_text)
        
        # 测试不同的平滑方法
        smoothing_methods = ['none', 'laplace', 'add_k', 'interpolation']
        
        for method in smoothing_methods:
            if method == 'add_k':
                bigram_probs, unigram_counts, bigram_counts = train_bigram_model(
                    sentences, vocab, smoothing=method, k=0.5
                )
            else:
                bigram_probs, unigram_counts, bigram_counts = train_bigram_model(
                    sentences, vocab, smoothing=method
                )
            
            # 检查bigram_probs是否包含所有的前导词
            for sentence in sentences:
                for i in range(len(sentence) - 1):
                    w_prev = sentence[i]
                    self.assertIn(w_prev, bigram_probs, f"bigram_probs应该包含词{w_prev}")
            
            # 检查概率总和
            for w_prev in bigram_probs:
                probs_sum = sum(bigram_probs[w_prev].values())
                if method != 'none':  # 'none'方法可能导致概率和不为1
                    self.assertAlmostEqual(
                        probs_sum, 1.0, delta=0.1,
                        msg=f"{method}平滑方法下，词{w_prev}后面所有词的概率和应该接近1"
                    )

    def test_generate_text(self):
        """测试文本生成功能"""
        sentences, vocab = preprocess_text(self.english_text)
        bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing='laplace')
        
        # 生成多个文本样本
        for _ in range(5):
            text = generate_text(bigram_probs, max_length=20)
            
            # 检查生成的文本不为空
            self.assertGreater(len(text), 0, "生成的文本不应该为空")
            
            # 检查生成的文本中的词都在词汇表中
            for word in text.split():
                self.assertIn(word, vocab, f"生成的词{word}应该在词汇表中")

    def test_calculate_perplexity(self):
        """测试困惑度计算功能"""
        sentences, vocab = preprocess_text(self.english_text)
        
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


if __name__ == '__main__':
    # 创建一个单独的bigram_model.py文件，包含notebook中的所有函数
    print("请确保已经创建了bigram_model.py文件，包含notebook中的所有函数")
    unittest.main()