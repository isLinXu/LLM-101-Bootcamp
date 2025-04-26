# -*- coding: utf-8 -*-
"""
扩展的Bigram语言模型测试用例

这个文件包含了一系列扩展测试，用于更全面地验证Bigram语言模型的各个组件，
包括边界条件测试、不同语言的处理、各种平滑方法的比较等。
"""

import unittest
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在无GUI环境中出错

# 导入Bigram模型的函数
from bigram_model import (
    preprocess_text,
    preprocess_chinese_text,
    train_bigram_model,
    generate_text,
    calculate_perplexity,
    visualize_bigram_probabilities
)

# 设置随机种子，确保测试结果可重现
random.seed(42)
np.random.seed(42)


class TestBigramModelExtended(unittest.TestCase):
    """扩展的Bigram语言模型测试类"""

    def setUp(self):
        """设置测试环境"""
        # 英文测试文本
        self.english_text = """
        The quick brown fox jumps over the lazy dog. 
        The dog barks at the fox. 
        The fox runs away quickly.
        A language model is a probability distribution over sequences of words.
        Bigram models consider only the previous word when predicting the next word.
        """
        
        # 中文测试文本
        self.chinese_text = """
        自然语言处理是人工智能的重要分支。
        大语言模型可以生成流畅的文本。
        Bigram模型是最简单的语言模型之一。
        语言模型在机器翻译中有重要应用。
        """
        
        # 空文本
        self.empty_text = ""
        
        # 单句文本
        self.single_sentence_text = "This is a single sentence."
        
        # 包含特殊字符的文本
        self.special_chars_text = "Text with special chars: @#$%^&*()!"
        
        # 包含数字的文本
        self.numeric_text = "Text with numbers: 123, 456, 789."
        
        # 包含重复词的文本
        self.repeated_words_text = "The the the fox fox jumps jumps over over the the dog dog."

    def test_preprocess_empty_text(self):
        """测试空文本的预处理"""
        sentences, vocab = preprocess_text(self.empty_text)
        self.assertEqual(len(sentences), 0, "空文本应该生成0个句子")
        self.assertEqual(len(vocab), 0, "空文本应该生成空词汇表")

    def test_preprocess_single_sentence(self):
        """测试单句文本的预处理"""
        sentences, vocab = preprocess_text(self.single_sentence_text)
        self.assertEqual(len(sentences), 1, "单句文本应该生成1个句子")
        self.assertGreater(len(vocab), 3, "词汇表应该包含至少3个词")
        self.assertIn('<s>', vocab, "词汇表应该包含开始标记<s>")
        self.assertIn('</s>', vocab, "词汇表应该包含结束标记</s>")

    def test_preprocess_special_chars(self):
        """测试包含特殊字符的文本预处理"""
        sentences, vocab = preprocess_text(self.special_chars_text)
        self.assertEqual(len(sentences), 1, "应该生成1个句子")
        self.assertIn('@#$%^&*()', vocab, "词汇表应该包含特殊字符")

    def test_preprocess_numeric_text(self):
        """测试包含数字的文本预处理"""
        sentences, vocab = preprocess_text(self.numeric_text)
        self.assertEqual(len(sentences), 1, "应该生成1个句子")
        self.assertIn('123', vocab, "词汇表应该包含数字")
        self.assertIn('456', vocab, "词汇表应该包含数字")
        self.assertIn('789', vocab, "词汇表应该包含数字")

    def test_preprocess_repeated_words(self):
        """测试包含重复词的文本预处理"""
        sentences, vocab = preprocess_text(self.repeated_words_text)
        self.assertEqual(len(sentences), 1, "应该生成1个句子")
        # 重复词在词汇表中只应该出现一次
        self.assertEqual(len([w for w in vocab if w == 'the']), 1, "重复词在词汇表中只应该出现一次")

    def test_train_model_different_smoothing(self):
        """测试不同平滑方法的模型训练"""
        sentences, vocab = preprocess_text(self.english_text)
        smoothing_methods = ['none', 'laplace', 'add_k', 'interpolation']
        
        for method in smoothing_methods:
            if method == 'add_k':
                bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing=method, k=0.5)
            else:
                bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing=method)
            
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

    def test_generate_text_max_length(self):
        """测试生成文本的最大长度限制"""
        sentences, vocab = preprocess_text(self.english_text)
        bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing='laplace')
        
        # 测试不同的最大长度
        for max_length in [5, 10, 20]:
            text = generate_text(bigram_probs, max_length=max_length)
            words = text.split()
            self.assertLessEqual(len(words), max_length, f"生成的文本长度应该不超过{max_length}")

    def test_perplexity_comparison(self):
        """比较不同平滑方法的困惑度"""
        sentences, vocab = preprocess_text(self.english_text)
        
        # 划分训练集和测试集
        train_sentences = sentences[:2]  # 前两个句子作为训练集
        test_sentences = sentences[2:]   # 剩余句子作为测试集
        
        # 使用不同的平滑方法
        smoothing_methods = ['laplace', 'add_k', 'interpolation']
        perplexities = {}
        
        for method in smoothing_methods:
            if method == 'add_k':
                bigram_probs, _, _ = train_bigram_model(
                    train_sentences, vocab, smoothing=method, k=0.5
                )
            else:
                bigram_probs, _, _ = train_bigram_model(
                    train_sentences, vocab, smoothing=method
                )
            
            # 计算困惑度
            perplexity = calculate_perplexity(bigram_probs, test_sentences)
            perplexities[method] = perplexity
            
            # 检查困惑度是否为正数
            self.assertGreater(perplexity, 0, "困惑度应该是正数")
        
        # 输出不同平滑方法的困惑度比较
        print("\n不同平滑方法的困惑度比较：")
        for method, perplexity in perplexities.items():
            print(f"{method}平滑方法的困惑度: {perplexity:.4f}")

    def test_chinese_text_processing(self):
        """测试中文文本处理"""
        sentences, vocab = preprocess_chinese_text(self.chinese_text)
        
        # 检查句子数量
        self.assertEqual(len(sentences), 4, "应该有4个句子")
        
        # 检查词汇表大小
        self.assertGreater(len(vocab), 20, "词汇表应该包含至少20个字符")
        
        # 训练中文Bigram模型
        bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing='laplace')
        
        # 生成中文文本
        for _ in range(3):
            text = generate_text(bigram_probs, max_length=30)
            self.assertGreater(len(text), 0, "生成的文本不应该为空")
        
        # 计算困惑度
        perplexity = calculate_perplexity(bigram_probs, sentences)
        self.assertGreater(perplexity, 0, "困惑度应该是正数")

    def test_visualize_bigram_probabilities(self):
        """测试Bigram概率可视化"""
        sentences, vocab = preprocess_text(self.english_text)
        bigram_probs, _, _ = train_bigram_model(sentences, vocab, smoothing='laplace')
        
        # 测试可视化函数（不显示图形，只检查函数是否正常运行）
        try:
            visualize_bigram_probabilities(bigram_probs, 'the', top_n=5)
            # 如果函数正常运行，测试通过
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"可视化函数抛出异常: {e}")

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空词汇表
        empty_sentences = []
        empty_vocab = set()
        
        # 应该能处理空输入而不崩溃
        try:
            bigram_probs, _, _ = train_bigram_model(empty_sentences, empty_vocab, smoothing='laplace')
            self.assertEqual(len(bigram_probs), 0, "空输入应该生成空模型")
        except Exception as e:
            self.fail(f"处理空输入时抛出异常: {e}")
        
        # 测试只有一个词的句子
        single_word_sentences = [['<s>', 'hello', '</s>']]
        single_word_vocab = {'<s>', 'hello', '</s>'}
        
        try:
            bigram_probs, _, _ = train_bigram_model(single_word_sentences, single_word_vocab, smoothing='laplace')
            self.assertIn('<s>', bigram_probs, "模型应该包含开始标记")
            self.assertIn('hello', bigram_probs['<s>'], "'<s>'后面应该可能出现'hello'")
        except Exception as e:
            self.fail(f"处理只有一个词的句子时抛出异常: {e}")


if __name__ == '__main__':
    unittest.main()